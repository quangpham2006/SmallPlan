# Habitat Environment Wrapper for SmallPlan
# Replaces iGibson environment with Habitat-Lab/Habitat-Sim

import logging
import os
from typing import Optional, Dict, Any, List, Tuple

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gymnasium.utils import seeding
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from PIL import Image
from scipy.spatial.transform import Rotation as R

import habitat
from habitat.config.default import get_config
from habitat.core.env import Env
from habitat.core.simulator import Observations
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.utils.geometry_utils import quaternion_to_list, quaternion_from_coeff
from habitat_sim.utils.common import quat_to_magnum, quat_from_magnum

try:
    import habitat_sim
    from habitat_sim import Agent, AgentConfiguration, AgentState
    from habitat_sim.physics import MotionType
    HABITAT_SIM_AVAILABLE = True
except ImportError:
    HABITAT_SIM_AVAILABLE = False
    print("Warning: habitat_sim not available")

from moma_llm.topology.habitat_simpleslam import HabitatSimpleSlam as SimpleSlam
from moma_llm.topology.habitat_topology import HabitatTopologyMapping as TopologyMapping, detect_rooms
from moma_llm.topology.habitat_room_graph import create_room_object_graph, get_closest_node
from moma_llm.topology.graph import plot_graph
from moma_llm.utils.habitat_constants import OCCUPANCY, NODETYPE

log = logging.getLogger(__name__)


class HabitatObjectWrapper:
    """Wrapper to provide iGibson-like object interface for Habitat objects."""
    
    def __init__(self, obj_id: int, semantic_id: int, category: str, 
                 position: np.ndarray, rotation: np.ndarray, 
                 bounding_box: Optional[np.ndarray] = None,
                 name: Optional[str] = None):
        self.obj_id = obj_id
        self.semantic_id = semantic_id
        self.category = category
        self._position = position
        self._rotation = rotation
        self.bounding_box = bounding_box
        self.name = name or f"{category}_{obj_id}"
        self.renderer_instances = [type('obj', (object,), {'id': obj_id})]
        
    def get_position(self) -> np.ndarray:
        return self._position
    
    def set_position(self, pos: np.ndarray):
        self._position = pos
        
    def get_orientation(self) -> np.ndarray:
        return self._rotation
    
    def get_position_orientation(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._position, self._rotation
    
    def get_base_link_position_orientation(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._position, self._rotation
    
    def get_base_aligned_bounding_box(self):
        """Returns bbox center, orientation, extent, and center in frame."""
        if self.bounding_box is not None:
            extent = self.bounding_box
        else:
            extent = np.array([0.5, 0.5, 0.5])
        return self._position, self._rotation, extent, np.zeros(3)
    
    def get_body_ids(self) -> List[int]:
        return [self.obj_id]


class HabitatSceneWrapper:
    """Wrapper to provide iGibson-like scene interface for Habitat scenes."""
    
    def __init__(self, sim: 'habitat_sim.Simulator', scene_id: str):
        self.sim = sim
        self.scene_id = scene_id
        self.scene_dir = ""  # Habitat uses different path structure
        
        # Build object dictionaries
        self.objects_by_name: Dict[str, HabitatObjectWrapper] = {}
        self.objects_by_id: Dict[int, HabitatObjectWrapper] = {}
        self.objects_by_semantic_id: Dict[int, HabitatObjectWrapper] = {}  # For semantic sensor lookup
        self.objects_by_category: Dict[str, List[HabitatObjectWrapper]] = {}
        self.category_ids: set = set()
        
        # Floor information
        self.floor_heights = [0.0]  # Default single floor
        self.floor_map = []
        self.floor_graph = []
        self.trav_map_resolution = 0.05
        self.trav_map_size = 200
        self.trav_map_original_size = 200
        self.trav_map_default_resolution = 0.05
        
        # Room information
        self.room_ins_map = None
        
        self._build_scene_objects()
        
    def _build_scene_objects(self):
        """Build object dictionaries from Habitat scene."""
        if not HABITAT_SIM_AVAILABLE:
            return
            
        scene = self.sim.semantic_scene
        if scene is None:
            log.warning("No semantic scene available")
            return
            
        for obj in scene.objects:
            if obj is None:
                continue
                
            obj_id = obj.id
            category = obj.category.name() if obj.category else "unknown"
            semantic_id = obj.semantic_id if hasattr(obj, 'semantic_id') else obj_id
            
            # Get object center and dimensions
            aabb = obj.aabb
            # Handle both property and method for center
            aabb_center = aabb.center() if callable(aabb.center) else aabb.center
            center = np.array(aabb_center, dtype=np.float32)
            size = np.array(aabb.size(), dtype=np.float32)
            
            # Create unique object name: use category and semantic_id to avoid duplicates
            # Don't use obj_id directly as it may already contain the category name
            obj_name = f"{category}_{semantic_id}"
            
            # Create wrapper
            wrapper = HabitatObjectWrapper(
                obj_id=obj_id,
                semantic_id=semantic_id,
                category=category,
                position=center,
                rotation=np.array([0, 0, 0, 1]),  # Default quaternion
                bounding_box=size,
                name=obj_name
            )
            
            self.objects_by_name[wrapper.name] = wrapper
            self.objects_by_id[obj_id] = wrapper
            # Also store by semantic_id for semantic sensor lookup
            if hasattr(obj, 'semantic_id'):
                self.objects_by_semantic_id[obj.semantic_id] = wrapper
            
            if category not in self.objects_by_category:
                self.objects_by_category[category] = []
            self.objects_by_category[category].append(wrapper)
            self.category_ids.add(category)
            
    def get_random_floor(self) -> int:
        return 0
    
    def get_object_by_id(self, obj_id: int) -> Optional['HabitatObjectWrapper']:
        """Get object wrapper by ID (checks both obj_id and semantic_id)."""
        # First try by obj_id
        obj = self.objects_by_id.get(obj_id, None)
        if obj is not None:
            return obj
        # Fallback to semantic_id (used by semantic sensor)
        return self.objects_by_semantic_id.get(obj_id, None)
    
    def get_object_by_name(self, name: str) -> Optional['HabitatObjectWrapper']:
        """Get object wrapper by name."""
        return self.objects_by_name.get(name, None)
    
    def reset_scene_objects(self):
        """Reset scene objects to initial state."""
        pass
    
    def get_room_instance_by_point(self, point: np.ndarray) -> Optional[int]:
        """Get room instance ID by world point."""
        if self.room_ins_map is None:
            return None
        map_point = self.world_to_map(point[:2])
        if (0 <= map_point[0] < self.room_ins_map.shape[0] and 
            0 <= map_point[1] < self.room_ins_map.shape[1]):
            return self.room_ins_map[map_point[0], map_point[1]]
        return None
    
    def get_room_type_by_point(self, point: np.ndarray) -> Optional[str]:
        """Get room type by world point."""
        room_id = self.get_room_instance_by_point(point)
        return f"room_{room_id}" if room_id is not None else None
    
    def world_to_map(self, point: np.ndarray) -> np.ndarray:
        """Convert world coordinates to map coordinates."""
        return np.flip((np.array(point) / self.trav_map_default_resolution + 
                       self.trav_map_original_size / 2.0)).astype(int)
    
    def map_to_world(self, point: np.ndarray) -> np.ndarray:
        """Convert map coordinates to world coordinates."""
        if len(point.shape) == 1:
            return (np.flip(point) - self.trav_map_original_size / 2.0) * self.trav_map_default_resolution
        else:
            return (np.flip(point, axis=1) - self.trav_map_original_size / 2.0) * self.trav_map_default_resolution
    
    def world_to_seg_map(self, point: np.ndarray) -> np.ndarray:
        """Convert world coordinates to segmentation map coordinates."""
        return self.world_to_map(point)
    
    def build_trav_graph(self, maps_path: str, floor: int, trav_map: np.ndarray):
        """Build traversability graph for navigation."""
        pass


class HabitatRobotWrapper:
    """Wrapper to provide iGibson-like robot interface for Habitat agent."""
    
    def __init__(self, sim: 'habitat_sim.Simulator', agent_id: int = 0):
        self.sim = sim
        self.agent_id = agent_id
        self.base_link = self  # Self-reference for compatibility
        self.eyes = self  # Self-reference for eye position
        
    def get_position(self) -> np.ndarray:
        """Get robot position."""
        agent_state = self.sim.get_agent(self.agent_id).get_state()
        return np.array(agent_state.position)
    
    def get_orientation(self) -> np.ndarray:
        """Get robot orientation as quaternion."""
        agent_state = self.sim.get_agent(self.agent_id).get_state()
        return np.array(quaternion_to_list(agent_state.rotation))
    
    def get_position_orientation(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get both position and orientation."""
        return self.get_position(), self.get_orientation()
    
    def get_rpy(self) -> Tuple[float, float, float]:
        """Get roll, pitch, yaw angles."""
        quat = self.get_orientation()
        r = R.from_quat(quat)
        return r.as_euler('xyz')
    
    def set_position_orientation(self, position: np.ndarray, orientation: np.ndarray):
        """Set robot position and orientation."""
        agent = self.sim.get_agent(self.agent_id)
        agent_state = agent.get_state()
        agent_state.position = position
        agent_state.rotation = quaternion_from_coeff(orientation)
        agent.set_state(agent_state)
        
    def reset(self):
        """Reset robot to initial state."""
        pass
    
    def get_body_ids(self) -> List[int]:
        """Get body IDs for compatibility."""
        return [self.agent_id]


class OurHabitatEnv:
    """
    Habitat environment wrapper that provides iGibson-like interface.
    This allows the high-level planning code to work with minimal changes.
    """
    
    def __init__(self, 
                 config_file: str,
                 scene_id: str,
                 mode: str = "headless",
                 seed: int = 42,
                 **kwargs):
        """
        Initialize Habitat environment.
        
        Args:
            config_file: Path to Habitat config YAML
            scene_id: Scene ID (e.g., from HM3D, MP3D, Gibson)
            mode: Rendering mode ("headless" or "gui")
            seed: Random seed
        """
        self.config_path = config_file
        self.scene_id = scene_id
        self.mode = mode
        
        # Set seed
        self.set_seed(seed)
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize Habitat simulator
        self.sim = self._create_simulator()
        
        # Create wrappers
        self.scene = HabitatSceneWrapper(self.sim, scene_id)
        self.simulator = self  # Self-reference for compatibility
        self.robots = [HabitatRobotWrapper(self.sim)]
        
        # SLAM and topology
        self.slam = SimpleSlam(
            voxel_size=self.config.get("voxel_size", 0.075),
            grid_size=int(np.ceil(self.config.get("grid_size_meter", 30) / 
                                  self.config.get("voxel_size", 0.075))),
            sensor_range=self.config.get("depth_high", 5.0),
            min_points_for_detection=self.config.get("min_points_for_detection", 50),
            verbose=self.config.get("verbose", False)
        )
        
        self.topology_mapping = TopologyMapping(
            size=self.slam.bev_map_semantic.shape,
            voxel_size=self.config.get("voxel_size", 0.075),
            verbose=self.config.get("verbose", False)
        )
        
        # State tracking
        self.opened_doors = []
        self.opened_windows = []
        self.robot_traj = []
        self.rgb_frames = []
        self.episode_info = {}
        self.episode_room_sem_acc = []
        
        # Visualization
        self.f, self.ax = plt.subplots(1, 3, figsize=(17, 5), width_ratios=[7/17, 5/17, 5/17])
        self.made_tight_layout = False
        
        # Task will be set separately
        self.task = None
        
        # Action and observation spaces
        self.action_space = self._create_action_space()
        self.observation_space = self._create_observation_space()
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load and merge configuration."""
        import yaml
        
        default_config = {
            "voxel_size": 0.075,
            "grid_size_meter": 30,
            "depth_high": 5.0,
            "depth_low": 0.0,
            "min_points_for_detection": 50,
            "image_width": 256,
            "image_height": 256,
            "vertical_fov": 90,
            "max_step": 10000000,
            "max_high_level_steps": 50,
            "control_freq": 10.0,
            "navigation_inflation_radius": 0.1,
            "magic_open_cost": 30,
            "consider_open_actions": True,
            "use_viewpoint_assignment": True,
            "topology": {
                "room_sdf_scale": 5,
                "room_sdf_thresh": 0.001
            }
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    default_config.update(file_config)
                    
        return default_config
    
    def _create_simulator(self) -> 'habitat_sim.Simulator':
        """Create Habitat simulator instance."""
        if not HABITAT_SIM_AVAILABLE:
            raise ImportError("habitat_sim is required but not installed")
        
        # Create simulator configuration
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self._get_scene_path()
        backend_cfg.enable_physics = True
        
        # Set scene dataset config for HM3D (required for scene discovery)
        scene_dataset_config = self._get_scene_dataset_config()
        if scene_dataset_config and os.path.exists(scene_dataset_config):
            backend_cfg.scene_dataset_config_file = scene_dataset_config
        
        # Agent configuration
        agent_cfg = AgentConfiguration()
        agent_cfg.height = 1.5
        agent_cfg.radius = 0.1
        
        # Sensor specifications
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "rgb"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [
            self.config.get("image_height", 256),
            self.config.get("image_width", 256)
        ]
        rgb_sensor_spec.position = [0.0, 1.5, 0.0]
        rgb_sensor_spec.hfov = self.config.get("vertical_fov", 90)
        
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [
            self.config.get("image_height", 256),
            self.config.get("image_width", 256)
        ]
        depth_sensor_spec.position = [0.0, 1.5, 0.0]
        depth_sensor_spec.hfov = self.config.get("vertical_fov", 90)
        
        semantic_sensor_spec = habitat_sim.CameraSensorSpec()
        semantic_sensor_spec.uuid = "semantic"
        semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_spec.resolution = [
            self.config.get("image_height", 256),
            self.config.get("image_width", 256)
        ]
        semantic_sensor_spec.position = [0.0, 1.5, 0.0]
        semantic_sensor_spec.hfov = self.config.get("vertical_fov", 90)
        
        agent_cfg.sensor_specifications = [
            rgb_sensor_spec, 
            depth_sensor_spec, 
            semantic_sensor_spec
        ]
        
        # Create simulator
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(cfg)
        
        return sim
    
    def _get_scene_path(self) -> str:
        """Get full path to scene file."""
        # This should be configured based on your Habitat data path
        # Common paths: HM3D, MP3D, Gibson datasets
        habitat_data_path = os.environ.get("HABITAT_DATA_PATH", "data")
        
        # For HM3D scenes, extract the scene name without the number prefix
        # e.g., "00800-TEEsavR23oF" -> "TEEsavR23oF"
        hm3d_scene_name = self.scene_id.split("-")[-1] if "-" in self.scene_id else self.scene_id
        
        # HM3D split folders to check
        hm3d_splits = ["minival", "train", "val", "test", "example"]
        
        # Try different dataset paths
        possible_paths = []
        
        # HM3D paths with split folders (most common structure)
        for split in hm3d_splits:
            possible_paths.append(
                f"{habitat_data_path}/scene_datasets/hm3d/{split}/{self.scene_id}/{hm3d_scene_name}.basis.glb"
            )
        
        # MP3D paths
        possible_paths.append(f"{habitat_data_path}/scene_datasets/mp3d/{self.scene_id}/{self.scene_id}.glb")
        
        # Gibson paths
        possible_paths.append(f"{habitat_data_path}/scene_datasets/gibson/{self.scene_id}.glb")
        
        # Direct path (if scene_id is already a full path)
        possible_paths.append(self.scene_id)
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        # Return the scene_id directly, assuming it's a full path
        return self.scene_id
    
    def _get_scene_dataset_config(self) -> str:
        """Get path to scene dataset configuration file."""
        habitat_data_path = os.environ.get("HABITAT_DATA_PATH", "data")
        
        # Check for HM3D scene dataset configs
        config_paths = [
            f"{habitat_data_path}/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
            f"{habitat_data_path}/scene_datasets/hm3d/hm3d_basis.scene_dataset_config.json",
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                return config_path
        
        return ""
    
    def _create_action_space(self):
        """Create action space compatible with original code."""
        import gymnasium as gym
        # Habitat uses discrete actions, but we'll map to continuous for compatibility
        return gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    
    def _create_observation_space(self):
        """Create observation space."""
        import gymnasium as gym
        return gym.spaces.Dict({
            "rgb": gym.spaces.Box(low=0, high=255, 
                                 shape=(self.config["image_height"], 
                                       self.config["image_width"], 3),
                                 dtype=np.uint8),
            "depth": gym.spaces.Box(low=0, high=10,
                                   shape=(self.config["image_height"],
                                         self.config["image_width"], 1),
                                   dtype=np.float32)
        })
    
    def set_seed(self, seed: int):
        """Set random seed."""
        if seed <= 0:
            seed = None
        self.np_random, _ = seeding.np_random(seed)
        np.random.seed(seed)
        
    def reset(self) -> Dict[str, Any]:
        """Reset environment."""
        self.robot_traj = []
        self.slam.reset()
        self.rgb_frames = []
        
        self.episode_info = {
            "num_low_level_steps": 0,
            "num_high_level_steps": 0,
            "scene_id": self.scene_id,
            "magic_open_actions": 0,
        }
        self.episode_room_sem_acc = []
        
        # Reset simulator
        self.sim.reset()
        
        # Get initial observation
        obs = self._get_observation()
        
        # Do initial SLAM updates by taking observations in different directions
        # This builds up the occupancy map before scene graph computation
        initial_turns = self.config.get("initial_slam_turns", 4)
        for i in range(initial_turns):
            # Update SLAM with current observation
            extrinsic = self._get_camera_extrinsic()
            self.slam.update(obs, extrinsic=extrinsic, scene=self.scene)
            
            # Turn to observe different direction
            self.sim.step("turn_left")
            obs = self._get_observation()
        
        # Final SLAM update
        extrinsic = self._get_camera_extrinsic()
        self.slam.update(obs, extrinsic=extrinsic, scene=self.scene)
        
        if self.task is not None:
            self.episode_info.update(self.task.task_info)
            
        return obs
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation from sensors."""
        obs = self.sim.get_sensor_observations()
        
        # Convert to expected format
        result = {
            "rgb": obs.get("rgb", np.zeros((self.config["image_height"], 
                                           self.config["image_width"], 3), 
                                          dtype=np.uint8)),
            "depth": obs.get("depth", np.zeros((self.config["image_height"],
                                               self.config["image_width"], 1),
                                              dtype=np.float32)),
            "seg": obs.get("semantic", np.zeros((self.config["image_height"],
                                                self.config["image_width"], 1),
                                               dtype=np.int32)),
            "ins_seg": obs.get("semantic", np.zeros((self.config["image_height"],
                                                    self.config["image_width"], 1),
                                                   dtype=np.int32)),
        }
        
        # Compute point cloud from depth
        result["pc"] = self._depth_to_pointcloud(result["depth"])
        
        return result
    
    def _depth_to_pointcloud(self, depth: np.ndarray) -> np.ndarray:
        """Convert depth image to point cloud."""
        h, w = depth.shape[:2]
        
        # Camera intrinsics
        fov = np.deg2rad(self.config.get("vertical_fov", 90))
        fx = fy = w / (2 * np.tan(fov / 2))
        cx, cy = w / 2, h / 2
        
        # Create meshgrid
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Compute 3D points
        z = depth.squeeze()
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        return np.stack([x, y, z], axis=-1)
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """Take a step in the environment."""
        # Map continuous action to Habitat discrete action
        if abs(action[0]) > abs(action[1]):
            if action[0] > 0:
                habitat_action = "move_forward"
            else:
                habitat_action = "move_backward" if hasattr(self, 'move_backward') else "move_forward"
        else:
            if action[1] > 0:
                habitat_action = "turn_left"
            else:
                habitat_action = "turn_right"
        
        # Execute action
        obs = self.sim.step(habitat_action)
        
        # Get observation
        result = self._get_observation()
        
        reward = 0.0
        done = False
        info = {}
        
        return result, reward, done, info
    
    def get_state(self, compute_scene_graph: bool = False) -> Dict[str, Any]:
        """Get current state with optional scene graph computation."""
        state = self._get_observation()
        self.rgb_frames.append(state["rgb"])
        
        robot_pos = self.robots[0].get_position()
        robot_orn = self.robots[0].get_orientation()
        
        # Update SLAM
        extrinsic = self._get_camera_extrinsic()
        self.slam.update(state, extrinsic=extrinsic, scene=self.scene)
        
        # Store robot trajectory
        robot_in_wf = np.eye(4)
        r = R.from_quat(robot_orn)
        robot_in_wf[:3, :3] = r.as_matrix()
        robot_in_wf[:3, 3] = robot_pos
        self.robot_traj.append(robot_in_wf)
        state["robot_traj"] = self.robot_traj
        
        if compute_scene_graph:
            wall_map = self.topology_mapping.update_maps(self.slam)
            voronoi_graph = self.topology_mapping.compute_voronoi_graph(
                self.slam, wall_map=wall_map
            )
            sparse_voronoi_graph = self.topology_mapping.sparsify_topology_graph()
            
            # Detect rooms (simplified for Habitat)
            separated_voronoi_graph = sparse_voronoi_graph.copy()
            door_pos = np.array([[0, 0]])  # Placeholder
            
            # Create room-object graph
            room_graph, room_object_graph = create_room_object_graph(
                scene=self.scene,
                slam=self.slam,
                vor_graph=sparse_voronoi_graph,
                separated_vor_graph=separated_voronoi_graph,
                obj_to_neglect=[],
                opened_doors=self.opened_doors,
                opened_windows=self.opened_windows,
                use_viewpoint_assignment=self.config.get("use_viewpoint_assignment", True),
                verbose=self.config.get("verbose", False)
            )
            
            # Add frontier points to rooms
            # Use X, Z coordinates (horizontal plane) since Habitat uses Y-up
            robot_pos_2d = np.array([robot_pos[0], robot_pos[2]])
            frontier_centers_meter, frontier_img, frontier_classification = self.slam.get_frontiers(
                robot_pos_2d, 
                occupancy_map=self.slam.bev_map_occupancy
            )
            state["frontier_img"] = frontier_img
            
            # Assign frontier points to rooms (this was missing!)
            if len(frontier_centers_meter) > 0 and len(separated_voronoi_graph.nodes) > 0:
                closest_nodes, _ = get_closest_node(
                    query_coords_world=frontier_centers_meter,
                    graph=separated_voronoi_graph,
                    slam=self.slam
                )
                if self.config.get("verbose", False):
                    print(f"DEBUG Frontier Assignment: {len(frontier_centers_meter)} frontiers found")
                for close_node, frontier_point in zip(closest_nodes, frontier_centers_meter):
                    node_data = separated_voronoi_graph.nodes.get(tuple(close_node), {})
                    room_id = node_data.get("room_id", 0)
                    room_name = NODETYPE.roomname(room_id)
                    room_node = room_object_graph.nodes.get(room_name)
                    if room_node is not None:
                        if "frontier_points" not in room_node:
                            room_node["frontier_points"] = set()
                        classification = frontier_classification.get(tuple(frontier_point), "unknown")
                        room_node["frontier_points"].add((tuple(frontier_point), classification))
                
                # Debug: print frontier assignment results
                if self.config.get("verbose", False):
                    for room_name in room_object_graph.successors("root"):
                        fp_count = len(room_object_graph.nodes.get(room_name, {}).get("frontier_points", set()))
                        if fp_count > 0:
                            print(f"DEBUG Frontier Assignment: Room '{room_name}' has {fp_count} frontier points")
            else:
                if self.config.get("verbose", False):
                    print(f"DEBUG Frontier Assignment: No frontiers found or no graph nodes")
            
            # Get current room (use X, Z for horizontal plane)
            if len(separated_voronoi_graph.nodes) > 0:
                closest_nodes, _ = get_closest_node(
                    query_coords_world=np.array([robot_pos_2d]),
                    graph=separated_voronoi_graph,
                    slam=self.slam
                )
                if len(closest_nodes) > 0:
                    room_id = separated_voronoi_graph.nodes.get(
                        tuple(closest_nodes[0]), {}
                    ).get("room_id", 0)
                    state["robot_current_room"] = NODETYPE.roomname(room_id)
                else:
                    state["robot_current_room"] = NODETYPE.roomname(0)
            else:
                state["robot_current_room"] = NODETYPE.roomname(0)
            
            state["room_graph"] = room_graph if room_graph else nx.DiGraph()
            state["voronoi_graph"] = sparse_voronoi_graph
            state["separated_voronoi_graph"] = separated_voronoi_graph
            state["room_object_graph"] = room_object_graph if room_object_graph else nx.DiGraph()
            state["door_pos"] = door_pos
            
        self.episode_info["num_low_level_steps"] += 1
        self.episode_info["num_high_level_steps"] += compute_scene_graph
        
        if self.episode_info["num_high_level_steps"] >= self.config.get("max_high_level_steps", 50):
            self.episode_info["failure_reason"] = "max_high_level_steps timeout"
            
        if len(self.robot_traj) > 1:
            self.episode_info["dist_travelled"] = np.linalg.norm(
                np.diff(np.stack(self.robot_traj)[:, :2, 3], axis=0), axis=-1
            ).sum()
        else:
            self.episode_info["dist_travelled"] = 0.0
            
        return state
    
    def _get_camera_extrinsic(self) -> np.ndarray:
        """Get camera extrinsic matrix."""
        agent_state = self.sim.get_agent(0).get_state()
        
        # Build extrinsic matrix
        pos = np.array(agent_state.position)
        quat = quaternion_to_list(agent_state.rotation)
        
        extrinsic = np.eye(4)
        r = R.from_quat(quat)
        extrinsic[:3, :3] = r.as_matrix()
        extrinsic[:3, 3] = pos
        
        return np.linalg.inv(extrinsic)
    
    def visualize(self, state: Optional[Dict] = None):
        """Visualize current state."""
        if state is None:
            state = self.get_state(compute_scene_graph=True)
            
        if self.task:
            self.f.suptitle(f"{self.scene_id}, {self.task.task_description}")
        else:
            self.f.suptitle(f"{self.scene_id}")
            
        margin = 20
        occupied_indices = np.where(self.slam.bev_map_occupancy == OCCUPANCY.OCCUPIED)
        
        if len(occupied_indices[0]) > 0:
            min_x = max(0, np.min(occupied_indices[0]) - margin)
            max_x = min(np.max(occupied_indices[0]) + margin, self.slam.bev_map_occupancy.shape[0])
            min_y = max(0, np.min(occupied_indices[1]) - margin)
            max_y = min(np.max(occupied_indices[1]) + margin, self.slam.bev_map_occupancy.shape[1])
        else:
            min_x, max_x = 0, self.slam.bev_map_occupancy.shape[0]
            min_y, max_y = 0, self.slam.bev_map_occupancy.shape[1]
            
        bounds = (min_x, max_x, min_y, max_y)
        
        # Plot room object graph
        if "room_object_graph" in state:
            plot_graph(state["room_object_graph"], self.slam.bev_map_occupancy, 
                      ax=self.ax[1], bounds=bounds)
        self.ax[1].set_title("room_object_graph")
        
        # Plot voronoi graph
        if "separated_voronoi_graph" in state and "frontier_img" in state:
            plot_graph(state["separated_voronoi_graph"], 
                      self.slam.bev_map_occupancy + 5 * state["frontier_img"],
                      ax=self.ax[2], bounds=bounds)
        self.ax[2].set_title("separated_voronoi_graph + frontiers")
        
        # Plot robot trajectory
        if len(self.robot_traj) > 0:
            robot_traj_pixel = self.slam.world2voxel(
                np.stack(self.robot_traj)[:, :3, 3]
            )[:, :2]
            
            norm = mpl.colors.Normalize(vmin=0, vmax=2000)
            xy = robot_traj_pixel[:, ::-1].reshape(-1, 1, 2)
            segments = np.hstack([xy[:-1], xy[1:]]) if len(xy) > 1 else xy
            
            for ax in self.ax[1:]:
                coll = LineCollection(segments, cmap=plt.cm.gray)
                coll.set_array(norm(np.arange(xy.shape[0])))
                ax.add_collection(coll)
                
                # Draw robot direction
                yaw = self.robots[0].get_rpy()[2]
                ax.arrow(robot_traj_pixel[-1, 1],
                        robot_traj_pixel[-1, 0],
                        2 * np.sin(yaw),
                        2 * np.cos(yaw),
                        width=1.5,
                        head_width=3.0,
                        fc="w")
                ax.set_axis_off()
                
        if "door_pos" in state:
            self.ax[1].scatter(state["door_pos"][:, 1], state["door_pos"][:, 0],
                              c="r", s=35.0, marker="x")
                              
        if not self.made_tight_layout:
            self.f.tight_layout()
            self.made_tight_layout = True
            
    def close(self):
        """Close environment."""
        if self.sim is not None:
            self.sim.close()
        cv2.destroyAllWindows()
        plt.close(self.f)
        
    def plot_object_position(self, obj_name, color="lime", marker="x"):
        """Plot object position on map."""
        if isinstance(obj_name, str):
            if obj_name in self.scene.objects_by_name:
                pos = self.scene.objects_by_name[obj_name].get_position()
            else:
                return
        else:
            pos = np.array(obj_name)
        pos_world = self.slam.world2voxel(pos[:2])
        self.ax[1].scatter(pos_world[1], pos_world[0], c=color, s=35.0, marker=marker)
        
    def run_simulation(self):
        """Run physics simulation step."""
        self.sim.step_physics(1.0 / 60.0)
        
    def sync(self, force_sync: bool = False):
        """Sync simulator state."""
        pass
    
    @property
    def renderer(self):
        """Get renderer (compatibility property)."""
        return type('obj', (object,), {'V': self._get_camera_extrinsic()})
    
    def save_video(self, output_path: str, fps: int = 30):
        """
        Save collected RGB frames as a video file.
        
        Args:
            output_path: Path to save the video (e.g., 'output.mp4')
            fps: Frames per second for the video
        """
        if not self.rgb_frames:
            print("No RGB frames to save")
            return
            
        try:
            import imageio
            print(f"Saving video with {len(self.rgb_frames)} frames to {output_path}")
            imageio.mimsave(output_path, self.rgb_frames, fps=fps)
            print(f"Video saved to {output_path}")
        except ImportError:
            print("imageio not installed. Install with: pip install imageio imageio-ffmpeg")
            # Fallback: save as GIF using PIL
            try:
                from PIL import Image
                images = [Image.fromarray(frame) for frame in self.rgb_frames]
                gif_path = output_path.replace('.mp4', '.gif')
                images[0].save(gif_path, save_all=True, append_images=images[1:], 
                              duration=1000//fps, loop=0)
                print(f"Saved as GIF to {gif_path}")
            except Exception as e:
                print(f"Failed to save video: {e}")


def create_habitat_env(config_file: str, 
                       scene_id: str,
                       control_freq: float = 10.0,
                       seed: int = 42,
                       mode: str = "headless") -> OurHabitatEnv:
    """
    Factory function to create Habitat environment.
    
    Args:
        config_file: Path to configuration file
        scene_id: Scene identifier
        control_freq: Control frequency (for compatibility)
        seed: Random seed
        mode: Rendering mode ("headless" or "gui")
        
    Returns:
        OurHabitatEnv instance
    """
    env = OurHabitatEnv(
        config_file=config_file,
        scene_id=scene_id,
        mode=mode,
        seed=seed
    )
    return env

