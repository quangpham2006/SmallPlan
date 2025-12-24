"""
Observation Processing for Habitat Navigation

Handles sensor data processing, point cloud generation, and scene understanding.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


@dataclass
class ObjectInfo:
    """Information about a visible object with distance."""
    category: str
    distance: float
    position: np.ndarray
    obj_id: int
    
    def __str__(self):
        return f"{self.category} ({self.distance:.1f}m)"


@dataclass
class ProcessedObservation:
    """
    Processed observation containing all sensor data and derived information.
    """
    # Raw sensor data
    rgb: np.ndarray
    depth: np.ndarray
    semantic: np.ndarray
    
    # Agent state
    position: np.ndarray
    rotation: np.ndarray
    yaw: float
    
    # Derived data
    pointcloud: Optional[np.ndarray] = None
    visible_objects: Optional[Set[str]] = None
    
    # Scene graph info - objects with distances
    visible_object_info: Optional[List[ObjectInfo]] = None  # Objects with distances
    
    def get_objects_with_distances(self) -> Dict[str, float]:
        """Get dict of {object_category: min_distance}."""
        if not self.visible_object_info:
            return {}
        
        result = {}
        for obj in self.visible_object_info:
            if obj.category not in result or obj.distance < result[obj.category]:
                result[obj.category] = obj.distance
        return result
    
    def format_nearby_objects(self, max_objects: int = 15) -> str:
        """Format nearby objects with distances for prompts."""
        if not self.visible_object_info:
            return "none visible"
        
        # Sort by distance and deduplicate by category
        obj_distances = self.get_objects_with_distances()
        sorted_objs = sorted(obj_distances.items(), key=lambda x: x[1])[:max_objects]
        
        if not sorted_objs:
            return "none visible"
        
        return ", ".join(f"{cat} ({dist:.1f}m)" for cat, dist in sorted_objs)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "rgb": self.rgb,
            "depth": self.depth,
            "semantic": self.semantic,
            "position": self.position,
            "rotation": self.rotation,
            "yaw": self.yaw,
            "pointcloud": self.pointcloud,
            "visible_objects": self.visible_objects,
            "visible_object_info": self.visible_object_info,
        }


class ObservationProcessor:
    """
    Processes raw sensor observations into structured data.
    
    Handles:
    - Point cloud generation from depth images
    - Object detection from semantic segmentation
    - Coordinate transformations
    """
    
    def __init__(self,
                 image_width: int = 256,
                 image_height: int = 256,
                 hfov: float = 90.0,
                 depth_min: float = 0.0,
                 depth_max: float = 10.0):
        """
        Initialize observation processor.
        
        Args:
            image_width: Width of sensor images
            image_height: Height of sensor images
            hfov: Horizontal field of view in degrees
            depth_min: Minimum valid depth in meters
            depth_max: Maximum valid depth in meters
        """
        self.width = image_width
        self.height = image_height
        self.hfov = np.deg2rad(hfov)
        self.depth_min = depth_min
        self.depth_max = depth_max
        
        # Precompute camera intrinsics
        self.fx = self.fy = image_width / (2 * np.tan(self.hfov / 2))
        self.cx = image_width / 2
        self.cy = image_height / 2
        
        # Precompute pixel coordinate grid
        self._u, self._v = np.meshgrid(
            np.arange(image_width),
            np.arange(image_height)
        )
    
    def process(self, 
                raw_obs: Dict[str, np.ndarray],
                agent_position: np.ndarray,
                agent_rotation: np.ndarray,
                scene=None,
                pathfinder=None) -> ProcessedObservation:
        """
        Process raw observations into structured format.
        
        Args:
            raw_obs: Dictionary with 'rgb', 'depth', 'semantic' arrays
            agent_position: Agent position (x, y, z)
            agent_rotation: Agent rotation quaternion (x, y, z, w)
            scene: Optional scene wrapper for object lookup
            pathfinder: Optional pathfinder for geodesic distance computation
            
        Returns:
            ProcessedObservation with all data
        """
        rgb = raw_obs["rgb"]
        depth = raw_obs["depth"]
        semantic = raw_obs["semantic"]
        
        # Compute yaw from quaternion
        r = R.from_quat(agent_rotation)
        yaw = r.as_euler('xyz')[1]
        
        # Generate point cloud
        pointcloud = self.depth_to_pointcloud(depth)
        
        # Detect visible objects with distances
        visible_objects = None
        visible_object_info = None
        if scene:
            visible_objects = self.get_visible_objects(semantic, scene)
            visible_object_info = self.get_visible_objects_with_distance(
                semantic, depth, agent_position, scene, pathfinder
            )
        
        return ProcessedObservation(
            rgb=rgb,
            depth=depth,
            semantic=semantic,
            position=agent_position,
            rotation=agent_rotation,
            yaw=yaw,
            pointcloud=pointcloud,
            visible_objects=visible_objects,
            visible_object_info=visible_object_info,
        )
    
    def depth_to_pointcloud(self, depth: np.ndarray) -> np.ndarray:
        """
        Convert depth image to point cloud in camera frame.
        
        Args:
            depth: Depth image (H, W) or (H, W, 1)
            
        Returns:
            Point cloud (H, W, 3) with (x, y, z) coordinates
        """
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
        
        # Compute 3D points
        z = depth
        x = (self._u - self.cx) * z / self.fx
        y = (self._v - self.cy) * z / self.fy
        
        return np.stack([x, y, z], axis=-1)
    
    def pointcloud_to_world(self,
                            pointcloud: np.ndarray,
                            position: np.ndarray,
                            rotation: np.ndarray) -> np.ndarray:
        """
        Transform point cloud from camera frame to world frame.
        
        Args:
            pointcloud: Points in camera frame (N, 3) or (H, W, 3)
            position: Agent position (3,)
            rotation: Agent rotation quaternion (4,)
            
        Returns:
            Points in world frame with same shape as input
        """
        original_shape = pointcloud.shape
        points = pointcloud.reshape(-1, 3)
        
        # Build transformation matrix
        r = R.from_quat(rotation)
        rotation_matrix = r.as_matrix()
        
        # Transform points
        world_points = (rotation_matrix @ points.T).T + position
        
        return world_points.reshape(original_shape)
    
    def get_visible_objects(self, 
                           semantic: np.ndarray,
                           scene) -> Set[str]:
        """
        Get set of visible object categories from semantic segmentation.
        
        Args:
            semantic: Semantic segmentation image (H, W) or (H, W, 1)
            scene: Scene wrapper with object information
            
        Returns:
            Set of visible object category names
        """
        if semantic.ndim == 3:
            semantic = semantic.squeeze(-1)
        
        visible = set()
        unique_ids = np.unique(semantic)
        
        for obj_id in unique_ids:
            if obj_id == 0:  # Skip background
                continue
            
            # Look up object in scene
            if hasattr(scene, 'get_object_by_id'):
                obj = scene.get_object_by_id(int(obj_id))
                if obj and hasattr(obj, 'category'):
                    visible.add(obj.category)
        
        return visible
    
    def _get_geodesic_distance(self, 
                               start_pos: np.ndarray, 
                               end_pos: np.ndarray,
                               pathfinder=None) -> float:
        """
        Compute geodesic distance between two positions.
        
        Args:
            start_pos: Starting position (x, y, z)
            end_pos: Ending position (x, y, z)
            pathfinder: Optional pathfinder for geodesic computation
            
        Returns:
            Geodesic distance (or Euclidean if pathfinder unavailable)
        """
        if pathfinder is not None:
            try:
                path = pathfinder.find_path(start_pos, end_pos)
                if path.geodesic_distance < float('inf'):
                    return float(path.geodesic_distance)
            except Exception as e:
                logger.debug(f"Pathfinder failed for object distance: {e}")
        
        # Fall back to Euclidean distance
        return float(np.linalg.norm(start_pos - end_pos))
    
    def get_visible_objects_with_distance(self,
                                          semantic: np.ndarray,
                                          depth: np.ndarray,
                                          agent_position: np.ndarray,
                                          scene,
                                          pathfinder=None) -> List[ObjectInfo]:
        """
        Get visible objects with their geodesic distances from the agent.
        
        Args:
            semantic: Semantic segmentation image (H, W) or (H, W, 1)
            depth: Depth image (H, W) or (H, W, 1)
            agent_position: Agent position (x, y, z)
            scene: Scene wrapper with object information
            pathfinder: Optional pathfinder for geodesic distance computation
            
        Returns:
            List of ObjectInfo with category, geodesic distance, position
        """
        if semantic.ndim == 3:
            semantic = semantic.squeeze(-1)
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
        
        object_infos = []
        unique_ids = np.unique(semantic)
        
        # Structural objects to filter
        excluded = {"wall", "floor", "ceiling", "void", "unknown", "misc"}
        
        for obj_id in unique_ids:
            if obj_id == 0:
                continue
            
            obj = scene.get_object_by_id(int(obj_id)) if hasattr(scene, 'get_object_by_id') else None
            if not obj or not hasattr(obj, 'category'):
                continue
            
            category = obj.category.lower()
            if category in excluded or any(ex in category for ex in excluded):
                continue
            
            # Check if object is visible in the semantic mask
            obj_mask = semantic == obj_id
            if not np.any(obj_mask):
                continue
            
            # Use geodesic distance to the object position
            obj_pos = obj.get_position()
            distance = self._get_geodesic_distance(agent_position, obj_pos, pathfinder)
            
            object_infos.append(ObjectInfo(
                category=obj.category,
                distance=distance,
                position=obj_pos,
                obj_id=int(obj_id)
            ))
        
        # Sort by distance
        object_infos.sort(key=lambda x: x.distance)
        
        return object_infos
    
    def get_depth_at_center(self, depth: np.ndarray) -> float:
        """Get depth value at image center."""
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
        
        cy = depth.shape[0] // 2
        cx = depth.shape[1] // 2
        
        # Average over small region for robustness
        region = depth[cy-2:cy+2, cx-2:cx+2]
        return float(np.nanmean(region))
    
    def is_obstacle_ahead(self, 
                         depth: np.ndarray,
                         threshold: float = 0.5) -> bool:
        """
        Check if there's an obstacle directly ahead.
        
        Args:
            depth: Depth image
            threshold: Distance threshold in meters
            
        Returns:
            True if obstacle is closer than threshold
        """
        center_depth = self.get_depth_at_center(depth)
        return center_depth < threshold and center_depth > self.depth_min


class SceneObjectWrapper:
    """
    Wrapper for scene objects providing a consistent interface.
    
    Abstracts away differences between semantic scene object representations.
    """
    
    def __init__(self, obj_id: int, category: str, position: np.ndarray, 
                 name: Optional[str] = None, bounding_box: Optional[np.ndarray] = None):
        """
        Initialize object wrapper.
        
        Args:
            obj_id: Unique object ID
            category: Object category name
            position: Object position (x, y, z)
            name: Optional unique name
            bounding_box: Optional bounding box extents
        """
        self.obj_id = obj_id
        self.category = category
        self._position = np.asarray(position)
        self.name = name or f"{category}_{obj_id}"
        self.bounding_box = bounding_box
    
    def get_position(self) -> np.ndarray:
        """Get object position."""
        return self._position
    
    def get_position_2d(self) -> np.ndarray:
        """Get object position in 2D (x, z)."""
        return np.array([self._position[0], self._position[2]])


class SceneWrapper:
    """
    Wrapper around Habitat semantic scene.
    
    Provides object lookup and scene information.
    """
    
    def __init__(self, semantic_scene, scene_id: str):
        """
        Initialize scene wrapper.
        
        Args:
            semantic_scene: Habitat semantic scene object
            scene_id: Scene identifier
        """
        self.semantic_scene = semantic_scene
        self.scene_id = scene_id
        
        # Build object dictionaries
        self.objects_by_id: Dict[int, SceneObjectWrapper] = {}
        self.objects_by_category: Dict[str, List[SceneObjectWrapper]] = {}
        self.category_ids: Set[str] = set()
        
        self._build_objects()
    
    def _build_objects(self):
        """Build object dictionaries from semantic scene."""
        if self.semantic_scene is None:
            logger.warning("No semantic scene available")
            return
        
        objects = list(self.semantic_scene.objects) if self.semantic_scene.objects else []
        
        for obj in objects:
            if obj is None:
                continue
            
            obj_id = obj.id
            category = obj.category.name() if obj.category else "unknown"
            semantic_id = getattr(obj, 'semantic_id', obj_id)
            
            # Get object center
            aabb = obj.aabb
            center = np.array(aabb.center() if callable(aabb.center) else aabb.center)
            size = np.array(aabb.size())
            
            wrapper = SceneObjectWrapper(
                obj_id=obj_id,
                category=category,
                position=center,
                name=f"{category}_{semantic_id}",
                bounding_box=size
            )
            
            self.objects_by_id[obj_id] = wrapper
            self.objects_by_id[semantic_id] = wrapper  # Also index by semantic ID
            
            if category not in self.objects_by_category:
                self.objects_by_category[category] = []
            self.objects_by_category[category].append(wrapper)
            
            self.category_ids.add(category)
        
        logger.info(f"Built scene with {len(self.objects_by_id)} objects, "
                   f"{len(self.category_ids)} categories")
    
    def get_object_by_id(self, obj_id: int) -> Optional[SceneObjectWrapper]:
        """Get object by ID."""
        return self.objects_by_id.get(obj_id)
    
    def get_objects_by_category(self, category: str) -> List[SceneObjectWrapper]:
        """Get all objects of a category."""
        return self.objects_by_category.get(category, [])
    
    def get_all_categories(self) -> Set[str]:
        """Get all object categories in scene."""
        return self.category_ids

