"""
Object Navigation Environment

High-level environment for object navigation tasks.
Wraps the simulator and provides episode management.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from .simulator import HabitatSimulator, SimulatorConfig
from .observations import ObservationProcessor, SceneWrapper, ProcessedObservation
from ..utils.actions import Action, ActionSpace
from ..utils.constants import OCCUPANCY

logger = logging.getLogger(__name__)


def annotate_frame(rgb: np.ndarray, 
                   semantic: np.ndarray, 
                   scene, 
                   target_category: str,
                   task_info: Optional[str] = None) -> np.ndarray:
    """
    Annotate an RGB frame with visible object information.
    
    Args:
        rgb: RGB image (H, W, 3)
        semantic: Semantic segmentation (H, W) or (H, W, 1)
        scene: SceneWrapper for object lookup
        target_category: Target object category to highlight
        task_info: Optional task description to display
        
    Returns:
        Annotated RGB image
    """
    try:
        import cv2
    except ImportError:
        # If cv2 not available, return original frame
        return rgb
    
    # Make a copy to avoid modifying original
    frame = rgb.copy()
    h, w = frame.shape[:2]
    
    if semantic is not None:
        if semantic.ndim == 3:
            semantic = semantic.squeeze(-1)
        
        # Find unique objects in view
        unique_ids = np.unique(semantic)
        
        target_found = False
        
        for obj_id in unique_ids:
            if obj_id == 0:  # Skip background
                continue
            
            obj = scene.get_object_by_id(int(obj_id)) if hasattr(scene, 'get_object_by_id') else None
            if not obj or not hasattr(obj, 'category'):
                continue
            
            category = obj.category
            
            # Get bounding box from semantic mask
            mask = semantic == obj_id
            ys, xs = np.where(mask)
            
            if len(xs) == 0 or len(ys) == 0:
                continue
            
            # Skip very small detections
            if len(xs) < 100:
                continue
            
            x_min, x_max = int(np.min(xs)), int(np.max(xs))
            y_min, y_max = int(np.min(ys)), int(np.max(ys))
            
            # Determine if this is the target
            is_target = category.lower() == target_category.lower()
            
            if is_target:
                target_found = True
                color = (0, 255, 0)  # Green for target
                thickness = 3
            else:
                color = (200, 200, 200)  # Gray for other objects
                thickness = 1
            
            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
            
            # Draw label
            label = f"{category}" + (" [TARGET]" if is_target else "")
            font_scale = 0.5 if is_target else 0.4
            label_thickness = 2 if is_target else 1
            
            # Get text size for background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_thickness)
            
            # Draw text background
            cv2.rectangle(frame, (x_min, y_min - text_h - 6), (x_min + text_w + 4, y_min), color, -1)
            
            # Draw text
            text_color = (0, 0, 0) if is_target else (255, 255, 255)
            cv2.putText(frame, label, (x_min + 2, y_min - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, label_thickness)
        
        # Draw "TARGET FOUND!" banner if target is visible
        if target_found:
            banner_text = f"TARGET FOUND: {target_category}"
            font_scale = 0.8
            (text_w, text_h), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            
            # Green banner at top
            cv2.rectangle(frame, (0, 0), (w, text_h + 20), (0, 180, 0), -1)
            cv2.putText(frame, banner_text, ((w - text_w) // 2, text_h + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    
    # Draw task info at bottom
    if task_info:
        font_scale = 0.5
        (text_w, text_h), _ = cv2.getTextSize(task_info, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (0, h - text_h - 10), (w, h), (50, 50, 50), -1)
        cv2.putText(frame, task_info, (5, h - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    
    return frame


@dataclass
class EpisodeInfo:
    """Information about a navigation episode."""
    scene_id: str
    episode_id: int
    target_category: str
    target_description: str
    initial_position: np.ndarray
    target_position: Optional[np.ndarray] = None
    shortest_path_distance: Optional[float] = None
    
    # Episode state
    step_count: int = 0  # Low-level action steps
    high_level_step_count: int = 0  # LLM query count (for timeout)
    distance_travelled: float = 0.0
    success: bool = False
    done: bool = False
    
    # Track if target was reached during episode (for success on stop())
    target_reached: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "target_category": self.target_category,
            "target_description": self.target_description,
            "step_count": self.step_count,
            "high_level_step_count": self.high_level_step_count,
            "distance_travelled": self.distance_travelled,
            "success": self.success,
            "done": self.done,
            "shortest_path_distance": self.shortest_path_distance,
            "target_reached": self.target_reached,
        }


class ObjectNavEnv:
    """
    Environment for Object Goal Navigation (ObjectNav).
    
    The agent must navigate to find an instance of a target object category.
    
    Key features:
    - Nav-R1 compatible discrete action space
    - RGB-D observations with semantic segmentation
    - Episode-based task management
    - SPL (Success weighted by Path Length) metric support
    
    Example:
        env = ObjectNavEnv(scene_id="00800-TEEsavR23oF")
        obs, info = env.reset()
        
        while not info["done"]:
            action = agent.act(obs, info["target"])
            obs, reward, info = env.step(action)
        
        env.close()
    """
    
    def __init__(self,
                 scene_id: str,
                 config: Optional[Dict] = None,
                 max_episode_steps: int = 500,
                 success_distance: float = 1.5,
                 min_geodesic_distance: float = 5.0,
                 seed: int = 42,
                 annotate_videos: bool = True):
        """
        Initialize the ObjectNav environment.
        
        Args:
            scene_id: Scene identifier for Habitat
            config: Optional configuration dictionary
            max_episode_steps: Maximum steps per episode
            success_distance: Distance threshold for success (meters)
            min_geodesic_distance: Minimum geodesic distance to target (meters)
            seed: Random seed for reproducibility
            annotate_videos: Whether to annotate video frames with object info
        """
        self.scene_id = scene_id
        self.config = config or {}
        self.max_episode_steps = max_episode_steps
        self.success_distance = success_distance
        self.min_geodesic_distance = min_geodesic_distance
        self.seed = seed
        self._annotate_videos_config = annotate_videos
        
        # Random state
        self.np_random = np.random.RandomState(seed)
        
        # Create simulator config
        sim_config = SimulatorConfig.from_dict(self.config)
        
        # Initialize simulator
        self.simulator = HabitatSimulator(scene_id, sim_config)
        
        # Initialize observation processor
        self.obs_processor = ObservationProcessor(
            image_width=sim_config.image_width,
            image_height=sim_config.image_height,
            hfov=sim_config.hfov,
        )
        
        # Initialize scene wrapper
        self.scene = SceneWrapper(
            self.simulator.semantic_scene,
            scene_id
        )
        
        # Action space
        self.action_space = ActionSpace()
        
        # Episode tracking
        self.episode_info: Optional[EpisodeInfo] = None
        self.episode_count = 0
        self._previous_position: Optional[np.ndarray] = None
        
        # Observation history for videos
        self.rgb_frames: List[np.ndarray] = []
        self._annotate_videos: bool = self._annotate_videos_config  # Enable video annotations
        self._last_semantic: Optional[np.ndarray] = None  # Store for annotations
        
        # Simple frontier tracking (visited positions)
        self._visited_positions: List[np.ndarray] = []
        self._position_grid_size = 2.0  # meters
        
        logger.info(f"Initialized ObjectNavEnv for scene {scene_id}")
    
    def reset(self, 
              target_category: Optional[str] = None,
              episode_id: Optional[int] = None) -> Tuple[ProcessedObservation, Dict]:
        """
        Reset environment for a new episode.
        
        Args:
            target_category: Target object category (random if None)
            episode_id: Episode identifier (auto-increment if None)
            
        Returns:
            Tuple of (observation, episode_info_dict)
        """
        # Reset simulator
        raw_obs = self.simulator.reset()
        
        # Get initial agent state
        position, rotation = self.simulator.get_agent_state()
        
        # Sample target if not provided
        if target_category is None:
            target_category, sampled_dist = self._sample_target_category()
        else:
            sampled_dist = None
        
        # Get shortest path distance for SPL (recompute for accuracy)
        shortest_dist = self._compute_shortest_path_distance(
            position, target_category
        )
        if shortest_dist is None and sampled_dist is not None:
            shortest_dist = sampled_dist
        
        # Find target position for logging
        target_pos = self._get_nearest_target_position(target_category, position)
        
        # Create episode info
        episode_id = episode_id if episode_id is not None else self.episode_count
        self.episode_info = EpisodeInfo(
            scene_id=self.scene_id,
            episode_id=episode_id,
            target_category=target_category,
            target_description=f"Find a {target_category}",
            initial_position=position.copy(),
            target_position=target_pos,
            shortest_path_distance=shortest_dist,
        )
        
        self.episode_count += 1
        self._previous_position = position.copy()
        self.rgb_frames = []
        self._visited_positions = [position.copy()]  # Reset visited positions
        self._last_semantic = None
        
        # Process observation with pathfinder for geodesic distance
        obs = self.obs_processor.process(
            raw_obs, position, rotation, self.scene, self.simulator.pathfinder
        )
        
        # Store semantic for annotations
        self._last_semantic = obs.semantic
        
        # Store frame for video (with annotations)
        self._store_frame(obs)
        
        # Determine current room (simple heuristic based on visible objects)
        current_room = self._detect_current_room(obs)
        
        # Get frontier info (unexplored directions)
        frontier_info = self._compute_frontier_info(obs)
        
        info = {
            **self.episode_info.to_dict(),
            "target": target_category,
            "done": False,
            "current_room": current_room,
            "frontier_info": frontier_info,
        }
        
        # Log task initialization
        logger.info(f"=== TASK INIT: Episode {episode_id} ===")
        logger.info(f"  Scene: {self.scene_id}")
        logger.info(f"  Target: {target_category}")
        logger.info(f"  Agent position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
        if target_pos is not None:
            logger.info(f"  Target position: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})")
        logger.info(f"  Geodesic distance: {shortest_dist:.2f}m" if shortest_dist else "  Geodesic distance: N/A")
        logger.info(f"  Min distance threshold: {self.min_geodesic_distance}m")
        
        return obs, info
    
    def _get_nearest_target_position(self, target_category: str, agent_pos: np.ndarray) -> Optional[np.ndarray]:
        """Get the position of the nearest target object."""
        all_targets = self.scene.get_objects_by_category(target_category)
        if not all_targets:
            return None
        
        agent_height = agent_pos[1]
        targets = [t for t in all_targets if self._is_on_same_floor(t.get_position(), agent_height)]
        
        if not targets:
            return None
        
        min_dist = float('inf')
        nearest_pos = None
        for t in targets:
            dist = self._get_geodesic_distance(agent_pos, t.get_position())
            if dist < min_dist:
                min_dist = dist
                nearest_pos = t.get_position()
        
        return nearest_pos
    
    def step(self, action: Action | int) -> Tuple[ProcessedObservation, float, Dict]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action to execute (Action enum or int 0-4)
            
        Returns:
            Tuple of (observation, reward, info_dict)
        """
        if isinstance(action, int):
            action = Action(action)
        
        # Store previous position for distance calculation
        prev_pos = self._previous_position.copy()
        
        # Check for STOP action first
        done = False
        if action == Action.STOP:
            done = True
        else:
            # Execute action in simulator
            raw_obs = self.simulator.step(action)
        
        # Get new state
        if action != Action.STOP:
            position, rotation = self.simulator.get_agent_state()
        else:
            position, rotation = prev_pos, self.simulator.get_agent_state()[1]
            raw_obs = self.simulator.get_observations()
        
        # Update distance travelled
        step_distance = np.linalg.norm(position - prev_pos)
        self.episode_info.distance_travelled += step_distance
        self.episode_info.step_count += 1
        self._previous_position = position.copy()
        
        # Track visited positions for frontier detection
        self._visited_positions.append(position.copy())
        
        # Process observation with pathfinder for geodesic distance
        obs = self.obs_processor.process(
            raw_obs, position, rotation, self.scene, self.simulator.pathfinder
        )
        
        # Store semantic for annotations
        self._last_semantic = obs.semantic
        
        # Store frame for video (with annotations if enabled)
        self._store_frame(obs)
        
        # Check success
        success = self._check_success(obs)
        
        # Note: Timeout is now checked in inference.py based on action_level
        # - High-level mode: timeout based on LLM queries
        # - Low-level mode: timeout based on step_count
        # This is just a safety fallback for low-level steps
        if self.episode_info.step_count >= self.max_episode_steps * 10:
            done = True
            logger.warning(f"Episode safety limit reached at {self.episode_info.step_count} steps")
        
        if success:
            done = True
            self.episode_info.success = True
            logger.info(f"Episode succeeded! Target '{self.episode_info.target_category}' found")
        
        self.episode_info.done = done
        
        # Compute reward
        reward = self._compute_reward(success, step_distance, done)
        
        # Determine current room
        current_room = self._detect_current_room(obs)
        
        # Get frontier info (unexplored directions)
        frontier_info = self._compute_frontier_info(obs)
        
        info = {
            **self.episode_info.to_dict(),
            "target": self.episode_info.target_category,
            "action_taken": action.name,
            "current_room": current_room,
            "frontier_info": frontier_info,
        }
        
        return obs, reward, info
    
    def get_info(self) -> Dict:
        """Get current episode info without taking an action."""
        position, rotation = self.simulator.get_agent_state()
        raw_obs = self.simulator.get_observations()
        obs = self.obs_processor.process(
            raw_obs, position, rotation, self.scene, self.simulator.pathfinder
        )
        current_room = self._detect_current_room(obs)
        frontier_info = self._compute_frontier_info(obs)
        
        return {
            **self.episode_info.to_dict(),
            "target": self.episode_info.target_category,
            "current_room": current_room,
            "frontier_info": frontier_info,
        }
    
    def _get_agent_floor_height(self) -> float:
        """Get the floor height (Y coordinate) of the agent's current position."""
        position, _ = self.simulator.get_agent_state()
        return position[1]  # Y is up in Habitat
    
    def _is_on_same_floor(self, obj_position: np.ndarray, agent_height: float, 
                          floor_tolerance: float = 1.5) -> bool:
        """
        Check if an object is on the same floor as the agent.
        
        Args:
            obj_position: Object position (x, y, z)
            agent_height: Agent's Y coordinate
            floor_tolerance: Maximum height difference for same floor (meters)
            
        Returns:
            True if object is on the same floor
        """
        obj_height = obj_position[1]  # Y coordinate
        return abs(obj_height - agent_height) < floor_tolerance
    
    def _sample_target_category(self) -> Tuple[str, float]:
        """
        Sample a random target category from available objects, ensuring minimum geodesic distance.
        
        Returns:
            Tuple of (target_category, geodesic_distance_to_nearest_target)
        """
        # Filter out structural categories
        excluded = {
            "wall", "floor", "ceiling", "void", "unknown", 
            "misc", "door", "window", "outdoor", "room"
        }
        
        # Get agent position and floor height
        agent_pos, _ = self.simulator.get_agent_state()
        agent_height = agent_pos[1]
        pathfinder = self.simulator.pathfinder
        
        # Build list of (category, min_geodesic_distance) for valid targets
        valid_targets = []
        for cat in self.scene.category_ids:
            cat_lower = cat.lower()
            
            # Skip excluded categories
            if cat_lower in excluded:
                continue
            if any(ex in cat_lower for ex in ["wall", "floor", "ceiling"]):
                continue
            
            # Get objects on the same floor with their geodesic distances
            objects = self.scene.get_objects_by_category(cat)
            min_dist = float('inf')
            for obj in objects:
                obj_pos = obj.get_position()
                if not self._is_on_same_floor(obj_pos, agent_height):
                    continue
                
                # Compute geodesic distance
                dist = self._get_geodesic_distance(agent_pos, obj_pos)
                if dist < min_dist:
                    min_dist = dist
            
            # Only include if min distance meets threshold
            if min_dist >= self.min_geodesic_distance and min_dist < float('inf'):
                valid_targets.append((cat, min_dist))
        
        if not valid_targets:
            # Fallback: relax distance constraint and pick any valid category
            logger.warning(f"No targets >= {self.min_geodesic_distance}m away, relaxing constraint")
            for cat in self.scene.category_ids:
                cat_lower = cat.lower()
                if cat_lower in excluded or any(ex in cat_lower for ex in ["wall", "floor", "ceiling"]):
                    continue
                objects = self.scene.get_objects_by_category(cat)
                for obj in objects:
                    if self._is_on_same_floor(obj.get_position(), agent_height):
                        dist = self._get_geodesic_distance(agent_pos, obj.get_position())
                        valid_targets.append((cat, dist))
                        break
        
        if not valid_targets:
            logger.warning("No valid target categories found, using 'chair' as default")
            return ("chair", 0.0)
        
        # Randomly select from valid targets
        idx = self.np_random.randint(len(valid_targets))
        return valid_targets[idx]
    
    def _compute_shortest_path_distance(self, 
                                        start_position: np.ndarray,
                                        target_category: str) -> Optional[float]:
        """Compute geodesic distance to nearest target instance on the same floor."""
        all_targets = self.scene.get_objects_by_category(target_category)
        if not all_targets:
            return None
        
        # Filter to only targets on the same floor
        agent_height = start_position[1]
        targets = [
            t for t in all_targets 
            if self._is_on_same_floor(t.get_position(), agent_height)
        ]
        
        if not targets:
            logger.warning(f"No '{target_category}' targets on same floor")
            return None
        
        pathfinder = self.simulator.pathfinder
        if pathfinder is None:
            # Fall back to Euclidean distance
            distances = [
                np.linalg.norm(start_position - t.get_position())
                for t in targets
            ]
            return min(distances)
        
        # Use pathfinder for geodesic distances
        min_dist = float('inf')
        for target in targets:
            try:
                path = pathfinder.find_path(
                    start_position, target.get_position()
                )
                if path.geodesic_distance < min_dist:
                    min_dist = path.geodesic_distance
            except Exception:
                # Fall back to Euclidean
                dist = np.linalg.norm(start_position - target.get_position())
                if dist < min_dist:
                    min_dist = dist
        
        return min_dist if min_dist < float('inf') else None
    
    def _store_frame(self, obs: ProcessedObservation):
        """Store a frame for video, with annotations if enabled."""
        if self._annotate_videos and self.episode_info:
            # Use frame count since step_count isn't updated during teleportation
            frame_num = len(self.rgb_frames) + 1
            task_info = f"Task: Find {self.episode_info.target_category} | Frame: {frame_num}"
            frame = annotate_frame(
                obs.rgb, 
                obs.semantic, 
                self.scene, 
                self.episode_info.target_category,
                task_info
            )
            self.rgb_frames.append(frame)
        else:
            self.rgb_frames.append(obs.rgb.copy())
    
    def _get_geodesic_distance(self, start_pos: np.ndarray, end_pos: np.ndarray) -> float:
        """
        Compute geodesic (trajectory) distance between two positions.
        
        Uses the simulator's pathfinder to account for walls and obstacles.
        Falls back to Euclidean distance if pathfinder unavailable.
        
        Args:
            start_pos: Starting position (x, y, z)
            end_pos: Ending position (x, y, z)
            
        Returns:
            Geodesic distance in meters
        """
        pathfinder = self.simulator.pathfinder
        
        if pathfinder is not None:
            try:
                path = pathfinder.find_path(start_pos, end_pos)
                if path.geodesic_distance < float('inf'):
                    return path.geodesic_distance
            except Exception as e:
                logger.debug(f"Pathfinder failed: {e}")
        
        # Fall back to Euclidean distance
        return float(np.linalg.norm(start_pos - end_pos))
    
    def _find_nearest_target_position(self) -> Optional[np.ndarray]:
        """Find the position of the nearest target object using geodesic distance."""
        if not self.episode_info:
            return None
        
        target = self.episode_info.target_category
        targets = self.scene.get_objects_by_category(target)
        
        if not targets:
            return None
        
        agent_pos, _ = self.simulator.get_agent_state()
        
        nearest_dist = float('inf')
        nearest_pos = None
        
        for target_obj in targets:
            # Use geodesic distance instead of Euclidean
            dist = self._get_geodesic_distance(agent_pos, target_obj.get_position())
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_pos = target_obj.get_position()
        
        return nearest_pos
    
    def _get_angle_to_target(self, target_pos: np.ndarray) -> float:
        """
        Calculate the angle to turn to face a target position.
        
        Args:
            target_pos: Target position (x, y, z)
            
        Returns:
            Angle in radians (positive = turn left, negative = turn right)
        """
        agent_pos, agent_rot = self.simulator.get_agent_state()
        
        # Get current yaw from quaternion
        r = R.from_quat(agent_rot)
        current_yaw = r.as_euler('xyz')[1]
        
        # Calculate direction to target (in X-Z plane, Y is up)
        dx = target_pos[0] - agent_pos[0]
        dz = target_pos[2] - agent_pos[2]
        
        # Target yaw (angle from -Z axis, positive is counterclockwise when viewed from above)
        target_yaw = np.arctan2(-dx, -dz)
        
        # Calculate angle difference
        angle_diff = target_yaw - current_yaw
        
        # Normalize to [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        return angle_diff
    
    def _turn_to_face_target_and_capture(self, obs: ProcessedObservation):
        """
        Turn the agent to face the target object and capture extra frames.
        
        This is called when task success is detected to provide a nice
        ending to the video showing the agent facing the target.
        """
        target_pos = self._find_nearest_target_position()
        
        if target_pos is None:
            logger.warning("Could not find target position to face")
            return
        
        logger.info(f"Turning to face target at {target_pos}")
        
        # Calculate how much we need to turn
        angle_to_target = self._get_angle_to_target(target_pos)
        turn_angle_rad = np.deg2rad(self.simulator.config.turn_angle)
        
        # Determine number of turns needed
        num_turns = int(abs(angle_to_target) / turn_angle_rad)
        num_turns = min(num_turns, 12)  # Cap at 12 turns (full rotation)
        
        # Determine turn direction
        turn_action = Action.TURN_LEFT if angle_to_target > 0 else Action.TURN_RIGHT
        
        # Execute turns and capture frames
        for _ in range(num_turns):
            raw_obs = self.simulator.step(turn_action)
            position, rotation = self.simulator.get_agent_state()
            
            turn_obs = self.obs_processor.process(
                raw_obs, position, rotation, self.scene, self.simulator.pathfinder
            )
            
            # Store annotated frame
            self._store_frame(turn_obs)
        
        # Capture several frames of the target (pause effect)
        raw_obs = self.simulator.get_observations()
        position, rotation = self.simulator.get_agent_state()
        final_obs = self.obs_processor.process(
            raw_obs, position, rotation, self.scene, self.simulator.pathfinder
        )
        
        # Add success celebration frames
        for _ in range(15):  # ~1.5 seconds at 10 fps
            self._store_success_frame(final_obs)
        
        logger.info(f"Captured {num_turns} turning frames + 15 success frames")
    
    def _store_success_frame(self, obs: ProcessedObservation):
        """Store a success celebration frame with special annotation."""
        try:
            import cv2
            
            # Get annotated frame
            task_info = f"SUCCESS! Found {self.episode_info.target_category} in {self.episode_info.step_count} steps"
            frame = annotate_frame(
                obs.rgb, 
                obs.semantic, 
                self.scene, 
                self.episode_info.target_category,
                task_info
            )
            
            h, w = frame.shape[:2]
            
            # Add large "SUCCESS" text in center
            success_text = "SUCCESS!"
            font_scale = 2.0
            thickness = 4
            (text_w, text_h), _ = cv2.getTextSize(success_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Draw text shadow
            x = (w - text_w) // 2
            y = (h + text_h) // 2
            cv2.putText(frame, success_text, (x + 3, y + 3),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
            # Draw text
            cv2.putText(frame, success_text, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            
            self.rgb_frames.append(frame)
            
        except ImportError:
            # Fallback without cv2
            self.rgb_frames.append(obs.rgb.copy())
    
    def _check_success(self, obs: ProcessedObservation) -> bool:
        """
        Check if the navigation task is successful.
        
        Success is True if:
        - Target category is visible in current observations, OR
        - Target was successfully reached earlier (target_reached flag set)
        
        The target_reached flag is set by mark_target_reached() when a goto action
        successfully navigates to the target (called from inference.py).
        """
        # First check if target was already reached during episode
        if self.episode_info.target_reached:
            return True
        
        target = self.episode_info.target_category
        
        # Check if target is visible now (case-insensitive partial match)
        if obs.visible_objects:
            target_lower = target.lower()
            for visible_obj in obs.visible_objects:
                if target_lower in visible_obj.lower():
                    return True
        
        return False
    
    def mark_target_reached(self):
        """
        Mark that the target was successfully reached.
        
        Called when a goto action successfully navigates to the target object.
        This ensures success even if the target isn't visible when stop() is called.
        """
        self.episode_info.target_reached = True
        logger.info(f"Target '{self.episode_info.target_category}' marked as reached")
    
    def _compute_reward(self, 
                       success: bool,
                       step_distance: float,
                       done: bool) -> float:
        """
        Compute reward signal.
        
        Rewards:
        - Success: +10
        - Each step: -0.01 (encourage efficiency)
        - Distance to target reduction: +progress (shaping)
        """
        reward = 0.0
        
        if success:
            reward += 10.0
        elif done:
            reward -= 0.5  # Penalty for failure/timeout
        
        # Small step penalty
        reward -= 0.01
        
        return reward
    
    def _detect_current_room(self, obs: ProcessedObservation) -> str:
        """
        Detect current room based on position or visible objects.
        
        Returns a room identifier like "room-0", "room-1", etc.
        The LLM agent will classify these into semantic names.
        """
        # Simple room ID based on discretized position
        # This creates a unique room ID for each 5m x 5m area
        pos = obs.position
        room_x = int(pos[0] // 5)
        room_z = int(pos[2] // 5)
        
        return f"room-{abs(room_x)}_{abs(room_z)}"
    
    def _compute_frontier_info(self, obs: ProcessedObservation) -> List[str]:
        """
        Compute frontier (unexplored area) information from depth image.
        
        Analyzes depth in different directions to find navigable areas.
        Always returns useful exploration suggestions based on depth and visit history.
        
        Returns:
            List of frontier descriptions like ["ahead (2.5m, unexplored)", "left (1.8m)"]
        """
        frontiers = []
        depth = obs.depth
        
        if depth is None or depth.size == 0:
            return ["explore available directions"]
        
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
        
        h, w = depth.shape
        
        # Analyze depth in 3 horizontal regions: left, center, right
        # Focus on middle rows (avoid floor/ceiling)
        mid_start, mid_end = h // 3, 2 * h // 3
        left_region = depth[mid_start:mid_end, :w//3]
        center_region = depth[mid_start:mid_end, w//3:2*w//3]
        right_region = depth[mid_start:mid_end, 2*w//3:]
        
        # Thresholds
        min_navigable = 1.0  # Need at least 1m to move
        
        # Collect direction info with depths
        direction_info = []
        directions = [
            ("left", left_region),
            ("ahead", center_region),
            ("right", right_region),
        ]
        
        for direction, region in directions:
            valid_depths = region[(region > 0.1) & (region < 10.0)]
            if len(valid_depths) > 0:
                avg_depth = float(np.median(valid_depths))
                max_depth = float(np.max(valid_depths))
                
                # Use max depth to find best exploration option
                explore_depth = max(avg_depth, max_depth * 0.7)
                
                if explore_depth >= min_navigable:
                    is_unvisited = self._is_direction_unvisited(obs.position, obs.yaw, direction)
                    direction_info.append((direction, explore_depth, is_unvisited))
        
        # Sort by unvisited first, then by depth (farther = better exploration)
        direction_info.sort(key=lambda x: (-x[2], -x[1]))
        
        for direction, depth_val, is_unvisited in direction_info:
            status = "unexplored" if is_unvisited else "open"
            frontiers.append(f"{direction} ({depth_val:.1f}m, {status})")
        
        return frontiers if frontiers else ["turn around to find open areas"]
    
    def _is_direction_unvisited(self, current_pos: np.ndarray, yaw: float, direction: str) -> bool:
        """Check if a direction leads to unvisited area."""
        # Calculate target position in that direction
        distance = 3.0  # meters ahead
        
        if direction == "ahead":
            angle = yaw
        elif direction == "left":
            angle = yaw + np.pi / 4
        elif direction == "right":
            angle = yaw - np.pi / 4
        else:
            angle = yaw
        
        # Target position (Habitat uses Y-up, so movement is in X-Z plane)
        target_x = current_pos[0] - distance * np.sin(angle)
        target_z = current_pos[2] - distance * np.cos(angle)
        target_pos = np.array([target_x, current_pos[1], target_z])
        
        # Check if any visited position is close to target
        for visited in self._visited_positions:
            dist = np.linalg.norm(target_pos[[0, 2]] - visited[[0, 2]])
            if dist < self._position_grid_size:
                return False  # Already visited
        
        return True  # Unvisited
    
    def get_task_description(self) -> str:
        """Get natural language task description."""
        if self.episode_info:
            return self.episode_info.target_description
        return "Navigate to find the target object"
    
    def compute_spl(self) -> float:
        """
        Compute Success weighted by Path Length (SPL).
        
        SPL = success * (shortest_path / max(shortest_path, actual_path))
        """
        if not self.episode_info or not self.episode_info.success:
            return 0.0
        
        shortest = self.episode_info.shortest_path_distance
        if shortest is None or shortest <= 0:
            return 0.0
        
        actual = self.episode_info.distance_travelled
        return max(0.0, shortest / max(shortest, actual))
    
    def save_video(self, output_path: str, fps: int = 10):
        """
        Save episode as video.
        
        Args:
            output_path: Output video file path
            fps: Frames per second
        """
        if not self.rgb_frames:
            logger.warning("No frames to save")
            return
        
        try:
            import imageio
            imageio.mimsave(output_path, self.rgb_frames, fps=fps)
            logger.info(f"Saved video with {len(self.rgb_frames)} frames to {output_path}")
        except ImportError:
            logger.warning("imageio not installed. Install with: pip install imageio")
    
    def close(self):
        """Close environment and release resources."""
        if self.simulator:
            self.simulator.close()
        logger.info("Closed ObjectNavEnv")

