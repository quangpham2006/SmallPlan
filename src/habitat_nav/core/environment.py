"""
Object Navigation Environment

High-level environment for object navigation tasks.
Wraps the simulator and provides episode management.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any

import numpy as np

from .simulator import HabitatSimulator, SimulatorConfig
from .observations import ObservationProcessor, SceneWrapper, ProcessedObservation
from ..utils.actions import Action, ActionSpace, HighLevelActionSpace
from ..utils.constants import OCCUPANCY

logger = logging.getLogger(__name__)


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
    step_count: int = 0
    distance_travelled: float = 0.0
    success: bool = False
    done: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "target_category": self.target_category,
            "target_description": self.target_description,
            "step_count": self.step_count,
            "distance_travelled": self.distance_travelled,
            "success": self.success,
            "done": self.done,
            "shortest_path_distance": self.shortest_path_distance,
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
                 seed: int = 42):
        """
        Initialize the ObjectNav environment.
        
        Args:
            scene_id: Scene identifier for Habitat
            config: Optional configuration dictionary
            max_episode_steps: Maximum steps per episode
            success_distance: Distance threshold for success (meters)
            seed: Random seed for reproducibility
        """
        self.scene_id = scene_id
        self.config = config or {}
        self.max_episode_steps = max_episode_steps
        self.success_distance = success_distance
        self.seed = seed
        
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
        self.high_level_action_space = HighLevelActionSpace
        
        # Episode tracking
        self.episode_info: Optional[EpisodeInfo] = None
        self.episode_count = 0
        self._previous_position: Optional[np.ndarray] = None
        
        # Observation history for videos
        self.rgb_frames: List[np.ndarray] = []
        
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
            target_category = self._sample_target_category()
        
        # Get shortest path distance for SPL
        shortest_dist = self._compute_shortest_path_distance(
            position, target_category
        )
        
        # Create episode info
        episode_id = episode_id if episode_id is not None else self.episode_count
        self.episode_info = EpisodeInfo(
            scene_id=self.scene_id,
            episode_id=episode_id,
            target_category=target_category,
            target_description=f"Find a {target_category}",
            initial_position=position.copy(),
            shortest_path_distance=shortest_dist,
        )
        
        self.episode_count += 1
        self._previous_position = position.copy()
        self.rgb_frames = []
        
        # Process observation
        obs = self.obs_processor.process(
            raw_obs, position, rotation, self.scene
        )
        
        # Store frame for video
        self.rgb_frames.append(obs.rgb.copy())
        
        info = {
            **self.episode_info.to_dict(),
            "target": target_category,
            "done": False,
        }
        
        logger.info(f"Reset episode {episode_id}: target='{target_category}'")
        
        return obs, info
    
    def step(self, action: Action | int) -> Tuple[ProcessedObservation, float, Dict]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action to execute (Action enum or int 0-3)
            
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
        
        # Process observation
        obs = self.obs_processor.process(
            raw_obs, position, rotation, self.scene
        )
        
        # Store frame for video
        self.rgb_frames.append(obs.rgb.copy())
        
        # Check success
        success = self._check_success(obs)
        
        # Check termination conditions
        if self.episode_info.step_count >= self.max_episode_steps:
            done = True
            logger.info(f"Episode timed out at {self.max_episode_steps} steps")
        
        if success:
            done = True
            self.episode_info.success = True
            logger.info(f"Episode succeeded! Target '{self.episode_info.target_category}' found")
        
        self.episode_info.done = done
        
        # Compute reward
        reward = self._compute_reward(success, step_distance, done)
        
        info = {
            **self.episode_info.to_dict(),
            "target": self.episode_info.target_category,
            "action_taken": action.name,
        }
        
        return obs, reward, info
    
    def _sample_target_category(self) -> str:
        """Sample a random target category from available objects."""
        # Filter out structural categories
        excluded = {
            "wall", "floor", "ceiling", "void", "unknown", 
            "misc", "door", "window", "outdoor", "room"
        }
        
        valid_categories = [
            cat for cat in self.scene.category_ids
            if cat.lower() not in excluded
            and not any(ex in cat.lower() for ex in ["wall", "floor", "ceiling"])
        ]
        
        if not valid_categories:
            logger.warning("No valid target categories, using 'chair' as default")
            return "chair"
        
        return self.np_random.choice(valid_categories)
    
    def _compute_shortest_path_distance(self, 
                                        start_position: np.ndarray,
                                        target_category: str) -> Optional[float]:
        """Compute geodesic distance to nearest target instance."""
        targets = self.scene.get_objects_by_category(target_category)
        if not targets:
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
    
    def _check_success(self, obs: ProcessedObservation) -> bool:
        """
        Check if the navigation task is successful.
        
        Success requires:
        1. Target category is visible in observations
        2. Agent is within success_distance of target
        """
        target = self.episode_info.target_category
        
        # Check if target is visible
        if obs.visible_objects and target in obs.visible_objects:
            # Check distance to nearest target instance
            targets = self.scene.get_objects_by_category(target)
            agent_pos = obs.position
            
            for target_obj in targets:
                distance = np.linalg.norm(
                    agent_pos - target_obj.get_position()
                )
                if distance <= self.success_distance:
                    return True
        
        return False
    
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

