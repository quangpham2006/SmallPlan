# Habitat Object Search Task for SmallPlan
# Provides object search task compatible with Habitat-Lab/Habitat-Sim
# This is a complete rewrite without any iGibson dependencies

import logging
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple, Set

import cv2
import numpy as np

try:
    import pyastar
    PYASTAR_AVAILABLE = True
except ImportError:
    PYASTAR_AVAILABLE = False

from moma_llm.llm.habitat_llm import LLM_hugging, object_states
from moma_llm.navigation.habitat_navigation import (
    PyAstarHelper, 
    get_circular_kernel
)
from moma_llm.utils.habitat_constants import OCCUPANCY

log = logging.getLogger(__name__)


class HabitatObjectSearchTask:
    """
    Object Search Task for Habitat environment.
    The goal is to find a random target object in the scene.
    
    This is a Habitat-compatible implementation that doesn't depend on iGibson.
    """
    
    def __init__(self, env):
        """
        Initialize object search task.
        
        Args:
            env: HabitatEnv instance
        """
        self.env = env
        self.config = env.config if hasattr(env, 'config') else {}
        
        self.floor_num = 0
        self.initial_pos = np.array([0, 0, 0])
        self.initial_orn = np.array([0, 0, 0])
        self.target_category = ""
        self.shortest_dist = None
        self.reachable_targets = []
        
        # Track new objects and relations (for compatibility)
        self.new_objects = []
        self.new_obj_relations = {}
        self.new_parent_relations = {}
        
        # Get available categories from scene
        if hasattr(env, 'scene') and hasattr(env.scene, 'category_ids'):
            self.all_object_categories = list(env.scene.category_ids)
        else:
            self.all_object_categories = []
            
        self.scene = env.scene if hasattr(env, 'scene') else None
        
        # Categories to exclude from search
        self.excluded_categories = {
            "walls", "floors", "ceilings", "agent", "door", 
            "wall", "floor", "ceiling", "void", "unknown",
            "misc", "objects", "unlabeled"
        }
        
    @property
    def task_description(self) -> str:
        """Human-readable task description."""
        return f"find a {LLM_hugging.to_human_readable_object_name(self.target_category)}"
    
    @property
    def task_info(self) -> Dict[str, Any]:
        """Task information dictionary."""
        return {
            "target_category": self.target_category,
            "shortest_dist": self.shortest_dist
        }
    
    def get_valid_categories(self) -> Set[str]:
        """Get valid object categories for search task."""
        if self.scene is None:
            print("DEBUG: scene is None")
            return set()
        
        if hasattr(self.scene, 'objects_by_category'):
            all_cats = set(self.scene.objects_by_category.keys())
            valid = all_cats - self.excluded_categories
            print(f"DEBUG: All categories: {all_cats}")
            print(f"DEBUG: Valid categories (after exclusion): {valid}")
        else:
            print("DEBUG: scene has no objects_by_category")
            valid = set()
        return valid
    
    @staticmethod
    def get_random_point_inflated(env, floor: int, uniform_over_room: bool = True) -> Tuple[int, np.ndarray]:
        """
        Sample a random navigable point with inflation for safety.
        
        Args:
            env: Environment instance
            floor: Floor index
            uniform_over_room: Whether to sample uniformly over rooms
            
        Returns:
            Tuple of (floor index, sampled position)
        """
        # Get navigable area from SLAM map
        if hasattr(env, 'slam') and env.slam is not None:
            occupancy_map = env.slam.bev_map_occupancy
            can_trav = (occupancy_map == OCCUPANCY.FREE)
            
            # Inflate to keep robot away from obstacles
            kernel = get_circular_kernel(radius=5)
            occupied = (occupancy_map == OCCUPANCY.OCCUPIED).astype(np.uint8)
            inflated = cv2.dilate(occupied, kernel, iterations=1)
            can_trav = np.logical_and(can_trav, inflated == 0)
        else:
            # Fallback: use scene floor map if available
            if hasattr(env, 'scene') and hasattr(env.scene, 'floor_map') and len(env.scene.floor_map) > floor:
                trav = env.scene.floor_map[floor] == 0
                trav_inflated = cv2.dilate(trav.astype(np.uint8), get_circular_kernel(radius=5), iterations=1)
                can_trav = (trav_inflated == 0)
            else:
                # Return default position if no map available
                return floor, np.array([0, 0, 0])
        
        if not np.any(can_trav):
            # Fallback to any free space
            if hasattr(env, 'slam'):
                can_trav = (env.slam.bev_map_occupancy == OCCUPANCY.FREE)
            
        trav_space = np.where(can_trav)
        if trav_space[0].shape[0] == 0:
            # Return center if no traversable space found
            return floor, np.array([0, 0, 0])
            
        idx = env.np_random.integers(0, high=trav_space[0].shape[0])
        xy_map = np.array([trav_space[0][idx], trav_space[1][idx]])
        
        # Convert to world coordinates
        if hasattr(env, 'slam'):
            xy_world = env.slam.voxel2world(xy_map)
        elif hasattr(env, 'scene') and hasattr(env.scene, 'map_to_world'):
            xy_world = env.scene.map_to_world(xy_map)
        else:
            xy_world = xy_map * 0.05  # Default resolution
            
        z = env.scene.floor_heights[floor] if (hasattr(env, 'scene') and 
            hasattr(env.scene, 'floor_heights') and len(env.scene.floor_heights) > floor) else 0.0
        
        return floor, np.array([xy_world[0], xy_world[1], z])
    
    def _draw_target(self, env, valid_object_categories: Set[str]) -> Tuple[str, List]:
        """
        Draw a random target category and find instances.
        
        Args:
            env: Environment instance
            valid_object_categories: Set of valid categories
            
        Returns:
            Tuple of (target category, list of possible targets)
        """
        if not valid_object_categories:
            return "", []
            
        target_obj_category = env.np_random.choice(sorted(valid_object_categories))
        
        if self.scene and hasattr(self.scene, 'objects_by_category'):
            possible_targets = self.scene.objects_by_category.get(target_obj_category, [])
        else:
            possible_targets = []
        
        return target_obj_category, possible_targets
    
    def sample_initial_pose_and_target_category(self, env) -> Tuple[np.ndarray, np.ndarray, str, float, List]:
        """
        Sample initial robot pose and target object category.
        
        Args:
            env: Environment instance
            
        Returns:
            Tuple of (initial_pos, initial_orn, target_category, shortest_dist, reachable_targets)
        """
        _, initial_pos = self.get_random_point_inflated(env, self.floor_num)
        
        valid_object_categories = self.get_valid_categories()
        
        if not valid_object_categories:
            log.warning("No valid object categories found in scene")
            return initial_pos, np.array([0, 0, 0]), "unknown", 0.0, []
        
        max_trials = 100
        reachable_targets = []
        dists = []
        target_obj_category = ""
        
        for trial in range(max_trials):
            target_obj_category, possible_targets = self._draw_target(env, valid_object_categories)
            
            if not possible_targets:
                continue
                
            # Check reachability of each target
            for target in possible_targets:
                target_pos = target.get_position()
                
                # Simple Euclidean distance calculation
                dist = np.linalg.norm(initial_pos[:2] - target_pos[:2])
                
                # Add targets that are not too close to start
                if dist > 1.0:
                    reachable_targets.append(target)
                    dists.append(dist)
                    
            if reachable_targets:
                break
                
        if not reachable_targets:
            log.warning("Could not find reachable targets, using random target")
            if possible_targets:
                reachable_targets = possible_targets[:1]
                dists = [10.0]
            else:
                return initial_pos, np.array([0, 0, 0]), target_obj_category or "unknown", 0.0, []
                
        shortest_dist = np.min(dists) if dists else 0.0
        initial_orn = np.array([0, 0, env.np_random.uniform(0, np.pi * 2)])
        
        log.debug(f"Sampled initial pose: {initial_pos}, {initial_orn}")
        log.debug(f"Sampled target category: {target_obj_category}")
        
        return initial_pos, initial_orn, target_obj_category, shortest_dist, reachable_targets
    
    def reset_scene(self, env):
        """
        Reset scene for new episode.
        
        Args:
            env: Environment instance
        """
        self.floor_num = 0
        
        # Reset new objects tracking
        self.new_objects = []
        self.new_obj_relations = {}
        self.new_parent_relations = {}
        
        # Note: Habitat scenes are typically static, no dynamic object spawning
        
    def reset_agent(self, env):
        """
        Reset agent pose and sample new task.
        
        Args:
            env: Environment instance
        """
        # Sample initial pose and target
        (initial_pos, initial_orn, target_category, 
         shortest_dist, reachable_targets) = self.sample_initial_pose_and_target_category(env)
        
        # Set robot position
        from moma_llm.navigation.habitat_navigation import set_agent_state
        yaw = initial_orn[2]
        set_agent_state(env, initial_pos[:2], yaw, z_offset=0.0)
        
        # Store task parameters
        self.initial_pos = initial_pos
        self.initial_orn = initial_orn
        self.target_category = target_category
        self.shortest_dist = shortest_dist
        self.reachable_targets = reachable_targets
        
    def get_task_obs(self, env) -> Optional[Dict]:
        """
        Get task-specific observations.
        
        Args:
            env: Environment instance
            
        Returns:
            Task observations (None for this task)
        """
        return None
    
    def evaluate_success(self, env) -> bool:
        """
        Evaluate if task is completed successfully.
        
        The task is successful if the target object category has been observed.
        
        Args:
            env: Environment instance
            
        Returns:
            Whether task is successful
        """
        if not self.target_category:
            return False
            
        # Check if target category has been seen
        if hasattr(env, 'slam') and hasattr(env.slam, 'seen_instances'):
            for instance_id in env.slam.seen_instances:
                if self.scene and hasattr(self.scene, 'objects_by_id'):
                    obj = self.scene.objects_by_id.get(instance_id, None)
                    if obj is not None and hasattr(obj, 'category'):
                        if obj.category == self.target_category:
                            if hasattr(env, 'episode_info'):
                                env.episode_info["task_success"] = True
                            return True
        
        # Alternative: check if any object of target category is in view
        # (simplified for Habitat where we might not have instance tracking)
        
        if hasattr(env, 'episode_info'):
            env.episode_info["task_success"] = False
        return False


# Compatibility aliases
ObjectSearchTask = HabitatObjectSearchTask
