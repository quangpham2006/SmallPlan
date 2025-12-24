"""
Object Navigation Task

Defines the ObjectNav task where an agent must find an instance of a target object category.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

from ..utils.constants import EXCLUDED_OBJECT_CATEGORIES

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of a navigation task."""
    success: bool
    distance_to_target: float
    steps_taken: int
    distance_travelled: float
    shortest_path_distance: Optional[float]
    spl: float  # Success weighted by Path Length
    failure_reason: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "distance_to_target": self.distance_to_target,
            "steps_taken": self.steps_taken,
            "distance_travelled": self.distance_travelled,
            "shortest_path_distance": self.shortest_path_distance,
            "spl": self.spl,
            "failure_reason": self.failure_reason,
        }


@dataclass
class ObjectNavTask:
    """
    Object Goal Navigation Task.
    
    The agent must navigate to find an instance of a target object category.
    The task is successful when the agent is within a distance threshold of
    the target and the target is visible.
    """
    target_category: str
    success_distance: float = 1.5
    max_steps: int = 500
    require_visibility: bool = True
    
    # Task state
    start_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    shortest_path_distance: Optional[float] = None
    target_positions: List[np.ndarray] = field(default_factory=list)
    
    @property
    def description(self) -> str:
        """Human-readable task description."""
        return f"Find a {self.target_category}"
    
    @classmethod
    def sample_task(cls,
                    scene,
                    agent_position: np.ndarray,
                    rng: np.random.RandomState,
                    success_distance: float = 1.5,
                    max_steps: int = 500,
                    floor_tolerance: float = 1.5,
                    pathfinder=None) -> "ObjectNavTask":
        """
        Sample a random ObjectNav task for the scene.
        
        Args:
            scene: Scene wrapper with object information
            agent_position: Current agent position (x, y, z where y is up)
            rng: Random number generator
            success_distance: Distance threshold for success
            max_steps: Maximum episode steps
            floor_tolerance: Maximum height difference for same floor (meters)
            pathfinder: Optional pathfinder for geodesic distance computation
            
        Returns:
            ObjectNavTask instance
        """
        agent_height = agent_position[1]  # Y is up in Habitat
        
        # Get valid categories with objects on the same floor
        valid_categories = cls.get_valid_categories_on_floor(
            scene, agent_height, floor_tolerance
        )
        
        if not valid_categories:
            logger.warning("No valid target categories on same floor, using 'chair'")
            return cls(
                target_category="chair",
                success_distance=success_distance,
                max_steps=max_steps,
                start_position=agent_position.copy(),
            )
        
        # Sample target category
        target_category = rng.choice(list(valid_categories))
        
        # Get target positions (only on same floor)
        target_objects = scene.get_objects_by_category(target_category)
        target_positions = [
            obj.get_position() for obj in target_objects
            if cls._is_on_same_floor(obj.get_position(), agent_height, floor_tolerance)
        ]
        
        # Compute shortest path distance using geodesic distance
        if target_positions:
            distances = []
            for pos in target_positions:
                if pathfinder is not None:
                    try:
                        path = pathfinder.find_path(agent_position, pos)
                        if path.geodesic_distance < float('inf'):
                            distances.append(path.geodesic_distance)
                            continue
                    except Exception:
                        pass
                # Fall back to Euclidean
                distances.append(float(np.linalg.norm(agent_position - pos)))
            shortest_dist = min(distances) if distances else None
        else:
            shortest_dist = None
        
        logger.info(f"Sampled target '{target_category}' with {len(target_positions)} "
                    f"instance(s) on same floor (agent height: {agent_height:.2f}m, "
                    f"shortest path: {shortest_dist:.2f}m)" if shortest_dist else 
                    f"Sampled target '{target_category}' with {len(target_positions)} "
                    f"instance(s) on same floor (agent height: {agent_height:.2f}m)")
        
        return cls(
            target_category=target_category,
            success_distance=success_distance,
            max_steps=max_steps,
            start_position=agent_position.copy(),
            shortest_path_distance=shortest_dist,
            target_positions=target_positions,
        )
    
    @staticmethod
    def _is_on_same_floor(obj_position: np.ndarray, agent_height: float, 
                          floor_tolerance: float = 1.5) -> bool:
        """Check if an object is on the same floor as the agent."""
        obj_height = obj_position[1]  # Y coordinate (up)
        return abs(obj_height - agent_height) < floor_tolerance
    
    @staticmethod
    def get_valid_categories(scene) -> Set[str]:
        """
        Get valid object categories for navigation targets.
        
        Filters out structural elements and invalid categories.
        """
        if scene is None or not hasattr(scene, 'category_ids'):
            return set()
        
        valid = set()
        for category in scene.category_ids:
            cat_lower = category.lower()
            
            # Check exclusions
            if cat_lower in EXCLUDED_OBJECT_CATEGORIES:
                continue
            
            # Check for partial matches
            excluded = False
            for ex in ["wall", "floor", "ceiling", "void", "unknown"]:
                if ex in cat_lower:
                    excluded = True
                    break
            
            if not excluded:
                valid.add(category)
        
        return valid
    
    @classmethod
    def get_valid_categories_on_floor(cls, scene, agent_height: float, 
                                       floor_tolerance: float = 1.5) -> Set[str]:
        """
        Get valid object categories that have instances on the same floor as the agent.
        
        Args:
            scene: Scene wrapper with object information
            agent_height: Agent's Y coordinate
            floor_tolerance: Maximum height difference for same floor (meters)
            
        Returns:
            Set of valid category names with objects on the same floor
        """
        all_valid = cls.get_valid_categories(scene)
        
        if not all_valid or scene is None:
            return set()
        
        valid_on_floor = set()
        for category in all_valid:
            objects = scene.get_objects_by_category(category)
            for obj in objects:
                if cls._is_on_same_floor(obj.get_position(), agent_height, floor_tolerance):
                    valid_on_floor.add(category)
                    break  # Found at least one on same floor
        
        return valid_on_floor
    
    def _get_geodesic_distance(self, 
                               start_pos: np.ndarray, 
                               end_pos: np.ndarray,
                               pathfinder=None) -> float:
        """
        Compute geodesic distance between two positions.
        
        Args:
            start_pos: Starting position
            end_pos: Ending position
            pathfinder: Optional pathfinder for geodesic computation
            
        Returns:
            Geodesic distance (or Euclidean if pathfinder unavailable)
        """
        if pathfinder is not None:
            try:
                path = pathfinder.find_path(start_pos, end_pos)
                if path.geodesic_distance < float('inf'):
                    return path.geodesic_distance
            except Exception:
                pass
        
        # Fall back to Euclidean distance
        return float(np.linalg.norm(start_pos - end_pos))
    
    def check_success(self,
                      agent_position: np.ndarray,
                      visible_objects: Optional[Set[str]] = None,
                      pathfinder=None) -> bool:
        """
        Check if the task is successfully completed.
        
        Success requires target category to be visible.
        Distance condition removed - if agent sees the target, that's success.
        
        Args:
            agent_position: Current agent position
            visible_objects: Set of visible object categories
            pathfinder: Optional pathfinder for geodesic distance (unused)
            
        Returns:
            True if task is successful
        """
        # Check if target is visible (case-insensitive partial match)
        if visible_objects:
            target_lower = self.target_category.lower()
            for visible_obj in visible_objects:
                if target_lower in visible_obj.lower():
                    return True
        
        return False
    
    def get_distance_to_target(self, agent_position: np.ndarray, pathfinder=None) -> float:
        """Get geodesic distance to nearest target instance."""
        if not self.target_positions:
            return float('inf')
        
        distances = [
            self._get_geodesic_distance(agent_position, pos, pathfinder)
            for pos in self.target_positions
        ]
        return min(distances)
    
    def compute_spl(self, 
                    distance_travelled: float,
                    success: bool) -> float:
        """
        Compute Success weighted by Path Length.
        
        SPL = success * (shortest / max(shortest, actual))
        """
        if not success:
            return 0.0
        
        if self.shortest_path_distance is None or self.shortest_path_distance <= 0:
            return 0.0
        
        return self.shortest_path_distance / max(
            self.shortest_path_distance, distance_travelled
        )
    
    def get_result(self,
                   agent_position: np.ndarray,
                   steps_taken: int,
                   distance_travelled: float,
                   success: bool,
                   failure_reason: Optional[str] = None) -> TaskResult:
        """
        Get final task result.
        
        Args:
            agent_position: Final agent position
            steps_taken: Number of steps taken
            distance_travelled: Total distance travelled
            success: Whether task was successful
            failure_reason: Reason for failure if applicable
            
        Returns:
            TaskResult with all metrics
        """
        distance_to_target = self.get_distance_to_target(agent_position)
        spl = self.compute_spl(distance_travelled, success)
        
        return TaskResult(
            success=success,
            distance_to_target=distance_to_target,
            steps_taken=steps_taken,
            distance_travelled=distance_travelled,
            shortest_path_distance=self.shortest_path_distance,
            spl=spl,
            failure_reason=failure_reason,
        )

