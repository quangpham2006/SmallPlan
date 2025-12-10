# Habitat-compatible SimpleSlam module for SmallPlan
# Provides SLAM functionality compatible with Habitat-Lab/Habitat-Sim sensors

import copy
import matplotlib.pyplot as plt
import cv2
import numpy as np
from numba import njit, prange
import scipy.ndimage
from typing import Dict, Any, Optional, Set, Tuple

from moma_llm.utils.habitat_constants import (
    OCCUPANCY, 
    HABITAT_SEMANTIC_CLASSES,
    HABITAT_CLASS_ID_TO_NAME
)
from moma_llm.navigation.frontier import find_frontiers, classify_frontiers


@njit(parallel=False)
def last_nonzero_numba(arr, value):
    """Find last non-zero value along z-axis for each (x,y) position."""
    for x in prange(arr.shape[0]):
        for y in prange(arr.shape[1]):
            for z in range(arr.shape[2], 0, -1):
                if arr[x, y, z] != 0:
                    value[x, y] = arr[x, y, z]
                    break
    return value


class HabitatSimpleSlam:
    """
    Simple SLAM implementation for Habitat environment.
    Processes RGB-D and semantic observations to build occupancy maps.
    """
    
    def __init__(self, 
                 grid_size: int, 
                 voxel_size: float, 
                 sensor_range: float, 
                 min_points_for_detection: int,
                 verbose: bool = False):
        """
        Initialize SLAM module.
        
        Args:
            grid_size: Size of the voxel grid
            voxel_size: Size of each voxel in meters
            sensor_range: Maximum sensor range in meters
            min_points_for_detection: Minimum points to count as detection
            verbose: Enable debug output
        """
        self.sensor_range = sensor_range
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.midpoint = grid_size // 2
        assert self.grid_size % 2 == 0
        self.min_points_for_detection = min_points_for_detection
        self.verbose = verbose
        
        # Semantic class mappings
        self.semantic_classes = HABITAT_SEMANTIC_CLASSES
        self.class_id_to_name = HABITAT_CLASS_ID_TO_NAME
        
        self.reset()

    @property
    def seen_instances(self) -> Set[int]:
        """Get set of seen instance IDs."""
        return set(self.instance_viewpoints.keys())
    
    def reset(self):
        """Reset SLAM state for new episode."""
        self.voxel_map = np.zeros([self.grid_size] * 3, dtype=np.float32)
        self.bev_map_semantic = np.zeros([self.grid_size] * 2, dtype=np.float32)
        self.bev_map_occupancy = np.zeros_like(self.bev_map_semantic)
        # instance_id: (viewpoint_position, distance-to-instance)
        self.instance_viewpoints: Dict[int, Tuple[np.ndarray, float]] = {}

    def world2voxel(self, world_coords: np.ndarray) -> np.ndarray:
        """
        Convert world coordinates to voxel indices.
        
        Args:
            world_coords: World coordinates (2D or 3D)
            
        Returns:
            Voxel indices
        """
        voxel_idx = np.round(world_coords / self.voxel_size).astype(int) + self.midpoint
        # Clip to valid range instead of asserting
        voxel_idx = np.clip(voxel_idx, 1, self.grid_size - 2)
        return voxel_idx
    
    def voxel2world(self, voxel_coords: np.ndarray) -> np.ndarray:
        """
        Convert voxel indices to world coordinates.
        
        Args:
            voxel_coords: Voxel indices
            
        Returns:
            World coordinates
        """
        if isinstance(voxel_coords, tuple):
            voxel_coords = np.array(voxel_coords)
        world_coords = (voxel_coords - self.midpoint) * self.voxel_size
        return world_coords

    def _update_closest_viewpoints(self, 
                                    scene, 
                                    instance_seg: np.ndarray, 
                                    dist: np.ndarray, 
                                    extrinsic_inv: np.ndarray):
        """
        Update closest viewpoint information for detected instances.
        
        Args:
            scene: Scene wrapper with object information
            instance_seg: Instance segmentation array
            dist: Distance array
            extrinsic_inv: Inverse extrinsic matrix
        """
        viewpoint_pos_world = extrinsic_inv[:3, 3]
        
        # Get unique instances and their counts
        unique_instances, counts = np.unique(instance_seg, return_counts=True)
        
        newly_detected = 0
        for instance_id, count in zip(unique_instances, counts):
            if instance_id == 0:  # Skip background
                continue
                
            if count > self.min_points_for_detection:
                # Get minimum distance for this instance
                mask = instance_seg == instance_id
                min_dist = np.min(dist[mask.reshape(-1)])
                
                # Update if closer than previous viewpoint
                if self.instance_viewpoints.get(instance_id, (None, np.inf))[1] > min_dist:
                    was_new = instance_id not in self.instance_viewpoints
                    self.instance_viewpoints[instance_id] = (viewpoint_pos_world.copy(), min_dist)
                    if was_new:
                        newly_detected += 1
                        if self.verbose:
                            obj = scene.get_object_by_id(instance_id)
                            obj_name = getattr(obj, 'category', 'unknown') if obj else 'unknown'
                            print(f"  New instance detected: {instance_id} ({obj_name}), {count} points, dist={min_dist:.2f}m")
        
        if self.verbose and newly_detected > 0:
            print(f"Total instances tracked: {len(self.instance_viewpoints)} (+{newly_detected} new)")

    @property
    def clipping_range(self) -> Tuple[int, int]:
        """Get z-axis clipping range for 2D projection."""
        rng = int(1 / self.voxel_size)
        return self.midpoint - rng, self.midpoint + rng
    
    def delete_obj_from_voxel_map(self, obj):
        """
        Delete object from voxel map and update BEV maps.
        This is essential for allowing navigation through opened doors.
        
        Args:
            obj: Object wrapper with bounding box information or position
        """
        try:
            # Try to get bounding box, otherwise use position
            if hasattr(obj, 'get_base_aligned_bounding_box'):
                bbox_center, _, bbox_extent, _ = obj.get_base_aligned_bounding_box()
                half_extent = bbox_extent / 2
                min_corner = bbox_center[:2] - half_extent[:2]
                max_corner = bbox_center[:2] + half_extent[:2]
            else:
                # Fallback: use position with default extent
                pos = obj.get_position()
                door_extent = 0.5  # Default door half-width in meters
                # Habitat: Y-up, so horizontal plane is X-Z
                min_corner = np.array([pos[0] - door_extent, pos[2] - door_extent])
                max_corner = np.array([pos[0] + door_extent, pos[2] + door_extent])
            
            # Convert to voxel coordinates
            min_voxel = self.world2voxel(min_corner)
            max_voxel = self.world2voxel(max_corner)
            
            # Expand by 1 voxel to ensure complete clearing
            x_min, x_max = min_voxel[0] - 1, max_voxel[0] + 2
            y_min, y_max = min_voxel[1] - 1, max_voxel[1] + 2
            
            # Clip to valid range
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(self.grid_size, x_max)
            y_max = min(self.grid_size, y_max)
            
            # Clear voxels in 3D map
            self.voxel_map[x_min:x_max, y_min:y_max, :] = 0
            
            # CRITICAL: Update BEV occupancy map to mark as FREE
            # This allows path planning to go through the doorway
            self.bev_map_occupancy[x_min:x_max, y_min:y_max] = OCCUPANCY.FREE
            self.bev_map_semantic[x_min:x_max, y_min:y_max] = 0
            
            if self.verbose:
                print(f"Deleted object from map: voxel range [{x_min}:{x_max}, {y_min}:{y_max}]")
            
        except Exception as e:
            print(f"Warning: Could not delete object from map: {e}")

    def _update_voxel_map(self, 
                          state: Dict[str, Any], 
                          extrinsic: np.ndarray, 
                          scene):
        """
        Update voxel map from sensor observations.
        
        Args:
            state: Observation dictionary with rgb, depth, seg, pc
            extrinsic: Camera extrinsic matrix
            scene: Scene wrapper
        """
        pc = state["pc"].reshape(-1, 3)
        dist = np.linalg.norm(pc, axis=1)
        dist_dense = dist.reshape(state["depth"].shape[:2])
        
        # Filter by sensor range
        within_sensing_range = dist_dense < self.sensor_range
        
        # Get valid points
        points = pc[within_sensing_range.reshape(-1)]
        
        # Debug: check depth stats
        if self.verbose:
            valid_depth = dist_dense[dist_dense > 0]
            if len(valid_depth) > 0:
                print(f"Depth stats: min={valid_depth.min():.2f}, max={valid_depth.max():.2f}, "
                      f"mean={valid_depth.mean():.2f}, within_range={np.sum(within_sensing_range)}")
        
        if len(points) == 0:
            print("No points within sensing range!")
            return
            
        # Get semantic labels
        seg = state.get("seg", np.zeros_like(state["depth"]))
        if seg.ndim == 3:
            seg = seg[:, :, 0]
        # Ensure seg has the same shape as within_sensing_range before indexing
        if seg.shape != within_sensing_range.shape:
            # seg might be 1D already or have different shape
            if seg.size == within_sensing_range.size:
                seg = seg.reshape(within_sensing_range.shape)
            else:
                # Use zeros if shape doesn't match
                seg = np.zeros(within_sensing_range.shape, dtype=seg.dtype)
        seg = seg[within_sensing_range].reshape(-1)
        
        # Get instance segmentation
        ins_seg = state.get("ins_seg", None)
        if ins_seg is None:
            ins_seg = seg.copy()  # Use semantic seg as instance seg
        else:
            if ins_seg.ndim == 3:
                ins_seg = ins_seg[:, :, 0]
            # Ensure ins_seg has the same shape as within_sensing_range
            if ins_seg.shape != within_sensing_range.shape:
                if ins_seg.size == within_sensing_range.size:
                    ins_seg = ins_seg.reshape(within_sensing_range.shape)
                else:
                    ins_seg = np.zeros(within_sensing_range.shape, dtype=ins_seg.dtype)
            ins_seg = ins_seg[within_sensing_range].reshape(-1)
        
        # Add homogeneous coordinate
        points = np.c_[points, np.ones(points.shape[0])]
        
        # Transform to world coordinates
        # extrinsic from habitat_env._get_camera_extrinsic() is world-to-camera
        # We need camera-to-world to transform camera points to world
        camera2world = np.linalg.inv(extrinsic)
        
        # Update closest viewpoints
        self._update_closest_viewpoints(
            scene=scene,
            instance_seg=ins_seg,
            dist=dist[within_sensing_range.reshape(-1)],
            extrinsic_inv=camera2world  # camera2world for viewpoint position
        )
        
        # Get world points (camera2world @ camera_points)
        world_points = camera2world.dot(points.T).T
        voxel_idx = self.world2voxel(world_points[:, :3])
        
        # Update voxel map
        # Use height-based floor detection (z < 0.15m is floor)
        # Note: in Habitat, y is typically up, but after transform z might be vertical
        floor_height_threshold = 0.15
        floor_class_id = self.semantic_classes.get("floor", 2)
        
        # Detect floor by height (points near ground level)
        # Camera is typically at ~1.5m, so floor points have low y (or z) values
        is_floor = world_points[:, 1] < floor_height_threshold  # y is up in Habitat
        
        # Debug: check world point distribution
        if self.verbose:
            print(f"World points Y range: min={world_points[:, 1].min():.2f}, max={world_points[:, 1].max():.2f}")
            print(f"Floor points (y < {floor_height_threshold}): {np.sum(is_floor)} / {len(is_floor)}")
        
        if np.any(is_floor):
            self.voxel_map[
                voxel_idx[is_floor, 0],
                voxel_idx[is_floor, 1],
                voxel_idx[is_floor, 2]
            ] = floor_class_id
        
        # Then write other objects (obstacles)
        non_floor_idx = ~is_floor
        if np.any(non_floor_idx):
            # Use a generic obstacle class (1 = wall/obstacle)
            obstacle_class = 1
            self.voxel_map[
                voxel_idx[non_floor_idx, 0],
                voxel_idx[non_floor_idx, 1],
                voxel_idx[non_floor_idx, 2]
            ] = obstacle_class

    def _update_2d_map(self):
        """Update 2D bird's eye view maps from 3D voxel map."""
        # In Habitat: Y is up (height), X and Z are horizontal
        # Voxel map is [x, y, z] where y is height
        # For BEV, we want to project along Y (height) axis to get X-Z plane
        
        # Clip out ceiling: only keep voxels at reasonable height (y-axis, index 1)
        height_clip_low = self.clipping_range[0]
        height_clip_high = self.clipping_range[1]
        clipped_map = self.voxel_map[:, height_clip_low:height_clip_high, :]
        
        # Project to 2D (X-Z plane) by taking max along Y (height) axis
        # Shape changes from [x, y_clipped, z] to [x, z]
        bev_map = np.max(clipped_map, axis=1)
        self.bev_map_semantic = bev_map
        
        # Create occupancy map
        occupancy_map = np.zeros_like(self.bev_map_semantic)
        
        # Floor ID (from height-based detection)
        floor_id = self.semantic_classes.get("floor", 2)
        
        # Any observed point that's not floor is an obstacle
        occupancy_map[bev_map > 0] = OCCUPANCY.OCCUPIED
        occupancy_map[bev_map == floor_id] = OCCUPANCY.FREE
        
        # Also check if we have ANY observations (floor or obstacle)
        # to mark as explored (free if only floor, occupied if obstacles)
        has_any_observation = np.any(clipped_map > 0, axis=1)
        has_floor_only = np.all((clipped_map == 0) | (clipped_map == floor_id), axis=1) & has_any_observation
        occupancy_map[has_floor_only & (occupancy_map == OCCUPANCY.UNEXPLORED)] = OCCUPANCY.FREE
        
        self.bev_map_occupancy = occupancy_map
        
    def update(self, 
               state: Dict[str, Any], 
               extrinsic: np.ndarray, 
               scene):
        """
        Update SLAM from new observation.
        
        Args:
            state: Observation dictionary
            extrinsic: Camera extrinsic matrix
            scene: Scene wrapper
        """
        self._update_voxel_map(state, extrinsic, scene=scene)
        self._update_2d_map()
        
        # Debug: print SLAM stats
        if self.verbose:
            free_count = np.sum(self.bev_map_occupancy == OCCUPANCY.FREE)
            occupied_count = np.sum(self.bev_map_occupancy == OCCUPANCY.OCCUPIED)
            voxel_count = np.sum(self.voxel_map > 0)
            print(f"SLAM update: voxels={voxel_count}, free={free_count}, occupied={occupied_count}")
    
    def get_frontiers(self, 
                      agent_pos_meter: np.ndarray, 
                      occupancy_map: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Find frontier points for exploration.
        
        Args:
            agent_pos_meter: Agent position in meters
            occupancy_map: Optional occupancy map (uses internal if None)
            
        Returns:
            Tuple of (frontier centers in world coords, frontier image, frontier classification)
        """
        if occupancy_map is None:
            occupancy_map = self.bev_map_occupancy
            
        occupancy_map = occupancy_map.astype(np.uint8)
        
        robot_pos_pixel = self.world2voxel(agent_pos_meter)[:2]
        frontier_centers_pixel, frontier_img = find_frontiers(
            occupancy_map,
            robot_pos_pixel,
            smoothing_kernel_size=3
        )
        frontier_centers_world = self.voxel2world(frontier_centers_pixel)
        
        frontier_classification = classify_frontiers(
            slam=self,
            frontier_img=frontier_img,
            frontier_centers_world=frontier_centers_world
        )
        
        return frontier_centers_world, frontier_img, frontier_classification


# Create alias for compatibility
SimpleSlam = HabitatSimpleSlam

