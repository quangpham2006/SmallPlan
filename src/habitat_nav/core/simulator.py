"""
Habitat Simulator Wrapper

Clean interface for Habitat-Sim following Nav-R1 conventions.
Handles simulator initialization, sensor configuration, and action execution.

Reference: https://github.com/AIGeeksGroup/Nav-R1
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import habitat_sim
    from habitat_sim import Agent, AgentConfiguration, AgentState
    from habitat.utils.geometry_utils import quaternion_to_list, quaternion_from_coeff
    HABITAT_AVAILABLE = True
except ImportError:
    HABITAT_AVAILABLE = False
    logging.warning("habitat_sim not available. Install with: pip install habitat-sim")

from ..utils.actions import Action, ActionSpace
from ..utils.constants import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class SimulatorConfig:
    """
    Configuration for Habitat Simulator.
    
    Follows Nav-R1 default settings for navigation tasks.
    """
    # Image settings
    image_width: int = 256
    image_height: int = 256
    hfov: float = 90.0
    
    # Sensor heights
    sensor_height: float = 1.5
    
    # Agent settings  
    agent_height: float = 1.5
    agent_radius: float = 0.1
    
    # Action settings (Nav-R1 defaults)
    forward_step_size: float = 0.25
    turn_angle: float = 30.0
    
    # Depth settings
    depth_min: float = 0.0
    depth_max: float = 10.0
    
    # Render mode
    enable_physics: bool = True
    
    # Scene dataset
    scene_dataset_config: str = ""
    
    @classmethod
    def from_dict(cls, config: Dict) -> "SimulatorConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


class HabitatSimulator:
    """
    Wrapper around Habitat-Sim simulator.
    
    Provides a clean interface for:
    - Sensor observation retrieval (RGB, Depth, Semantic)
    - Discrete action execution (Nav-R1 style)
    - Agent state management
    
    Example:
        config = SimulatorConfig()
        sim = HabitatSimulator(scene_id="00800-TEEsavR23oF", config=config)
        
        obs = sim.reset()
        obs = sim.step(Action.MOVE_FORWARD)
        
        sim.close()
    """
    
    def __init__(self, scene_id: str, config: Optional[SimulatorConfig] = None):
        """
        Initialize the Habitat simulator.
        
        Args:
            scene_id: Scene identifier (e.g., "00800-TEEsavR23oF" for HM3D)
            config: Simulator configuration (uses defaults if None)
        """
        if not HABITAT_AVAILABLE:
            raise ImportError("habitat_sim is required. Install with: pip install habitat-sim")
        
        self.scene_id = scene_id
        self.config = config or SimulatorConfig()
        self.action_space = ActionSpace(
            forward_step=self.config.forward_step_size,
            turn_angle=self.config.turn_angle
        )
        
        self._sim: Optional[habitat_sim.Simulator] = None
        self._agent_id: int = 0
        
        # Initialize simulator
        self._create_simulator()
        
    def _get_scene_path(self) -> str:
        """Resolve full path to scene file."""
        habitat_data = os.environ.get("HABITAT_DATA_PATH", "data")
        scene_name = self.scene_id.split("-")[-1] if "-" in self.scene_id else self.scene_id
        
        # HM3D scene paths
        hm3d_splits = ["minival", "train", "val", "test", "example"]
        for split in hm3d_splits:
            path = f"{habitat_data}/scene_datasets/hm3d/{split}/{self.scene_id}/{scene_name}.basis.glb"
            if os.path.exists(path):
                return path
        
        # MP3D fallback
        mp3d_path = f"{habitat_data}/scene_datasets/mp3d/{self.scene_id}/{self.scene_id}.glb"
        if os.path.exists(mp3d_path):
            return mp3d_path
        
        # Gibson fallback
        gibson_path = f"{habitat_data}/scene_datasets/gibson/{self.scene_id}.glb"
        if os.path.exists(gibson_path):
            return gibson_path
        
        # Assume full path was provided
        return self.scene_id
    
    def _get_scene_dataset_config(self) -> str:
        """Get scene dataset config path for HM3D."""
        habitat_data = os.environ.get("HABITAT_DATA_PATH", "data")
        config_paths = [
            f"{habitat_data}/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
            f"{habitat_data}/scene_datasets/hm3d/hm3d_basis.scene_dataset_config.json",
        ]
        for path in config_paths:
            if os.path.exists(path):
                return path
        return ""
    
    def _create_simulator(self):
        """Create and configure the Habitat-Sim simulator."""
        # Backend configuration
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self._get_scene_path()
        backend_cfg.enable_physics = self.config.enable_physics
        
        scene_dataset = self._get_scene_dataset_config()
        if scene_dataset:
            backend_cfg.scene_dataset_config_file = scene_dataset
        
        # Agent configuration
        agent_cfg = AgentConfiguration()
        agent_cfg.height = self.config.agent_height
        agent_cfg.radius = self.config.agent_radius
        
        # Sensor specifications
        sensors = self._create_sensor_specs()
        agent_cfg.sensor_specifications = sensors
        
        # Action specifications (Nav-R1 style)
        agent_cfg.action_space = {
            "move_forward": habitat_sim.ActionSpec(
                "move_forward",
                habitat_sim.ActuationSpec(amount=self.config.forward_step_size)
            ),
            "turn_left": habitat_sim.ActionSpec(
                "turn_left", 
                habitat_sim.ActuationSpec(amount=self.config.turn_angle)
            ),
            "turn_right": habitat_sim.ActionSpec(
                "turn_right",
                habitat_sim.ActuationSpec(amount=self.config.turn_angle)
            ),
        }
        
        # Create simulator
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self._sim = habitat_sim.Simulator(cfg)
        
        logger.info(f"Initialized Habitat simulator for scene: {self.scene_id}")
    
    def _create_sensor_specs(self) -> List:
        """Create sensor specifications for RGB, Depth, and Semantic."""
        resolution = [self.config.image_height, self.config.image_width]
        position = [0.0, self.config.sensor_height, 0.0]
        
        # RGB sensor
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "rgb"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = resolution
        rgb_spec.position = position
        rgb_spec.hfov = self.config.hfov
        
        # Depth sensor
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = resolution
        depth_spec.position = position
        depth_spec.hfov = self.config.hfov
        
        # Semantic sensor
        semantic_spec = habitat_sim.CameraSensorSpec()
        semantic_spec.uuid = "semantic"
        semantic_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_spec.resolution = resolution
        semantic_spec.position = position
        semantic_spec.hfov = self.config.hfov
        
        return [rgb_spec, depth_spec, semantic_spec]
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the simulator and return initial observations.
        
        Returns:
            Dictionary with 'rgb', 'depth', 'semantic' observations
        """
        self._sim.reset()
        return self.get_observations()
    
    def step(self, action: Action | int) -> Dict[str, np.ndarray]:
        """
        Execute an action and return observations.
        
        Args:
            action: Action to execute (Action enum or int 0-3)
            
        Returns:
            Dictionary with 'rgb', 'depth', 'semantic' observations
        """
        if isinstance(action, int):
            action = Action(action)
        
        if action == Action.STOP:
            # STOP doesn't change state, just return current observations
            return self.get_observations()
        
        habitat_action = self.action_space.get_habitat_action(action)
        self._sim.step(habitat_action)
        return self.get_observations()
    
    def get_observations(self) -> Dict[str, np.ndarray]:
        """
        Get current sensor observations.
        
        Returns:
            Dictionary containing:
                - 'rgb': RGB image (H, W, 3) uint8
                - 'depth': Depth image (H, W, 1) float32  
                - 'semantic': Semantic segmentation (H, W, 1) int32
        """
        obs = self._sim.get_sensor_observations()
        
        return {
            "rgb": obs.get("rgb", np.zeros(
                (self.config.image_height, self.config.image_width, 3), dtype=np.uint8
            )),
            "depth": obs.get("depth", np.zeros(
                (self.config.image_height, self.config.image_width, 1), dtype=np.float32
            )),
            "semantic": obs.get("semantic", np.zeros(
                (self.config.image_height, self.config.image_width, 1), dtype=np.int32
            )),
        }
    
    def get_agent_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current agent position and orientation.
        
        Returns:
            Tuple of (position, quaternion)
            - position: 3D position (x, y, z) in Habitat's Y-up coordinate system
            - quaternion: Orientation as quaternion (x, y, z, w)
        """
        state = self._sim.get_agent(self._agent_id).get_state()
        position = np.array(state.position)
        quaternion = np.array(quaternion_to_list(state.rotation))
        return position, quaternion
    
    def set_agent_state(self, position: np.ndarray, rotation: np.ndarray):
        """
        Set agent position and orientation.
        
        Args:
            position: 3D position (x, y, z)
            rotation: Quaternion (x, y, z, w) or euler angles (roll, pitch, yaw)
        """
        agent = self._sim.get_agent(self._agent_id)
        state = agent.get_state()
        state.position = position
        
        # Handle both quaternion and euler angle input
        if len(rotation) == 4:
            state.rotation = quaternion_from_coeff(rotation)
        else:
            # Assume euler angles (roll, pitch, yaw)
            r = R.from_euler('xyz', rotation)
            quat = r.as_quat()  # Returns (x, y, z, w)
            state.rotation = quaternion_from_coeff(quat)
        
        agent.set_state(state)
    
    def get_position_2d(self) -> np.ndarray:
        """
        Get agent position in 2D (horizontal plane).
        
        Returns:
            2D position (x, z) - using Habitat's Y-up convention
        """
        pos, _ = self.get_agent_state()
        return np.array([pos[0], pos[2]])
    
    def get_yaw(self) -> float:
        """
        Get agent yaw angle (rotation around Y axis).
        
        Returns:
            Yaw angle in radians
        """
        _, quat = self.get_agent_state()
        r = R.from_quat(quat)
        euler = r.as_euler('xyz')
        return euler[1]  # Y-axis rotation
    
    @property
    def semantic_scene(self):
        """Get semantic scene information."""
        return self._sim.semantic_scene if self._sim else None
    
    @property
    def pathfinder(self):
        """Get pathfinder for navigation queries."""
        return self._sim.pathfinder if self._sim else None
    
    def close(self):
        """Close the simulator and free resources."""
        if self._sim is not None:
            self._sim.close()
            self._sim = None
            logger.info("Closed Habitat simulator")

