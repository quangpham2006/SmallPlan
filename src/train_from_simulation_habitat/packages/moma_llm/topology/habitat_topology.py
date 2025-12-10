# Habitat-compatible Topology Mapping module for SmallPlan
# Provides voronoi-based room detection without iGibson dependencies

import copy
import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.ndimage.morphology
import scipy.stats
import skfmm
import skimage
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
from sklearn.neighbors import KernelDensity
from typing import Dict, List, Tuple, Optional, Any

from moma_llm.topology.graph import sparsify_graph, plot_graph
from moma_llm.navigation.habitat_navigation import get_circular_kernel
from moma_llm.topology.habitat_room_graph import get_body_properties
from moma_llm.utils.habitat_constants import HABITAT_SEMANTIC_CLASSES, OCCUPANCY


# =====================================================
# Quaternion utilities (replacing igibson.utils.mesh_util)
# =====================================================

def xyzw2wxyz(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion from (x, y, z, w) to (w, x, y, z) format.
    
    Args:
        quat: Quaternion in (x, y, z, w) format
        
    Returns:
        Quaternion in (w, x, y, z) format
    """
    return np.array([quat[3], quat[0], quat[1], quat[2]])


def wxyz2xyzw(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion from (w, x, y, z) to (x, y, z, w) format.
    
    Args:
        quat: Quaternion in (w, x, y, z) format
        
    Returns:
        Quaternion in (x, y, z, w) format
    """
    return np.array([quat[1], quat[2], quat[3], quat[0]])


def quat2rotmat(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        quat: Quaternion in (w, x, y, z) format
        
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quat
    
    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm > 0:
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Build rotation matrix
    rot = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
    return rot


def rotmat2quat(rot: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion.
    
    Args:
        rot: 3x3 rotation matrix
        
    Returns:
        Quaternion in (w, x, y, z) format
    """
    trace = np.trace(rot)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])


# =====================================================
# Topology Mapping
# =====================================================

def boundary_integral(sdf: np.ndarray, 
                      x1: int, y1: int, 
                      x2: int, y2: int, 
                      normalize: bool) -> float:
    """
    Compute boundary integral along a line segment.
    
    Args:
        sdf: Signed distance field
        x1, y1: Start point
        x2, y2: End point
        normalize: Whether to normalize by path length
        
    Returns:
        Integral value
    """
    N_interp = int(np.linalg.norm(np.array([x1-x2, y1-y2])))
    integral = 0
    if N_interp > 0:
        for j in range(N_interp):
            x = int((x1 + (float(j) / N_interp) * (x2 - x1)))
            y = int((y1 + (float(j) / N_interp) * (y2 - y1)))
            if sdf[x, y] > 0:
                integral += sdf[x, y]
    return integral / (N_interp + 1e8) if normalize else integral


def compute_sdf(boundary_mask: np.ndarray, distance_scale: float = 1) -> np.ndarray:
    """
    Compute signed distance field from boundary mask.
    
    Args:
        boundary_mask: Binary boundary mask
        distance_scale: Scale factor for distances
        
    Returns:
        Signed distance field
    """
    dx = 1
    f = distance_scale / dx
    
    # Check if boundary_mask has any boundaries (transitions between 0 and 1)
    phi = 1 - boundary_mask
    if np.all(phi == 0) or np.all(phi == 1):
        # No valid boundary - return zeros
        return np.zeros_like(boundary_mask, dtype=np.float32)
    
    try:
        sdf = skfmm.distance(phi)
        sdf[sdf > f] = f
        sdf = sdf / f
        sdf = 1 - sdf
        return sdf
    except ValueError as e:
        # Handle case where there's no zero contour
        print(f"Warning: compute_sdf failed: {e}")
        return np.zeros_like(boundary_mask, dtype=np.float32)


class HabitatTopologyMapping:
    """
    Topology mapping for Habitat environments.
    Builds voronoi-based room graphs from occupancy maps.
    """
    
    def __init__(self, size: int, voxel_size: float, verbose: bool = False):
        """
        Initialize topology mapper.
        
        Args:
            size: Grid size
            voxel_size: Size of each voxel in meters
            verbose: Enable debug output
        """
        self.size = size
        self.voxel_size = voxel_size
        self.places = nx.Graph()
        self.obstacle_map = None
        self.sdf_scale = 3.0
        self.verbose = verbose
    
    def _debug_print(self, message: str) -> None:
        """Print debug message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    @staticmethod
    def update_maps(slam) -> np.ndarray:
        """
        Update wall map from SLAM data.
        
        Args:
            slam: SLAM module
            
        Returns:
            Aggregated wall map
        """
        zero_height_pixel = slam.midpoint
        height_cutoff = int(1.75 / slam.voxel_size)
        
        # Get semantic class IDs
        wall_id = HABITAT_SEMANTIC_CLASSES.get("wall", 1)
        door_id = HABITAT_SEMANTIC_CLASSES.get("door", 4)
        window_id = HABITAT_SEMANTIC_CLASSES.get("window", 9)
        ceiling_id = HABITAT_SEMANTIC_CLASSES.get("ceiling", 17)
        
        agg_wall_map = np.isin(
            slam.voxel_map[:, :, zero_height_pixel:zero_height_pixel + height_cutoff],
            [wall_id, door_id, window_id]
        ).any(axis=2)
        
        filled_height_cutoff = int(1.2 / slam.voxel_size)
        agg_wall_map = np.logical_or(
            agg_wall_map,
            (slam.voxel_map[:, :, zero_height_pixel:zero_height_pixel + height_cutoff] > 0).sum(2) > filled_height_cutoff
        ).astype(np.float32)
        
        return agg_wall_map

    def compute_convex_hull(self, slam) -> List:
        """
        Compute convex hull of occupied space.
        
        Args:
            slam: SLAM module
            
        Returns:
            List of convex hull contours
        """
        self.occupied_map = slam.bev_map_semantic / max(OCCUPANCY)
        
        dilatation_size = 3
        dilation_shape = cv2.MORPH_ELLIPSE
        element = cv2.getStructuringElement(
            dilation_shape,
            (2 * dilatation_size + 1, 2 * dilatation_size + 1),
            (dilatation_size, dilatation_size)
        )
        self.occupied_map = cv2.dilate(self.occupied_map, element).astype(np.uint8)

        contours, _ = cv2.findContours(self.occupied_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            self.max_area_hull = []
            return []

        max_area_contours = [sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[0]]
        max_area_hull = [cv2.convexHull(max_area_contours[0], False)]

        self.max_area_hull = max_area_hull
        return max_area_hull

    @staticmethod
    def compute_filled_room_map(slam, dilate_erode_fill: bool) -> np.ndarray:
        """
        Compute filled room map from occupancy.
        
        Args:
            slam: SLAM module
            dilate_erode_fill: Whether to apply morphological operations
            
        Returns:
            Filled room map
        """
        ceiling_id = HABITAT_SEMANTIC_CLASSES.get("ceiling", 17)
        
        occupied_map = slam.bev_map_occupancy > 0
        occupied_map[(slam.voxel_map == ceiling_id).any(axis=2)] = True
        
        if dilate_erode_fill:
            kernel = get_circular_kernel(1)
            occupied_map = cv2.erode(
                cv2.dilate(occupied_map.astype(np.uint8), kernel, iterations=1),
                kernel,
                iterations=1
            )
        
        free_map = scipy.ndimage.binary_fill_holes(occupied_map.astype(bool)).astype(int)
        return free_map

    def compute_voronoi_graph(self, 
                               slam, 
                               wall_map: np.ndarray, 
                               sdf_scale: float = 3.0) -> nx.Graph:
        """
        Compute voronoi graph from wall map.
        
        Args:
            slam: SLAM module
            wall_map: Wall/obstacle map
            sdf_scale: SDF scale factor
            
        Returns:
            Voronoi graph
        """
        self.obstacle_map = wall_map
        self.sdf_scale = sdf_scale
        self.places = nx.Graph()
        
        free_map = (slam.bev_map_occupancy != OCCUPANCY.UNEXPLORED).astype(int)
        occupancy_map = (slam.bev_map_occupancy == OCCUPANCY.OCCUPIED).astype(int)
        bev_map_occupancy = slam.bev_map_occupancy
        
        # Check if we have any explored/occupied space
        self._debug_print(f"DEBUG Topology: free_map sum={np.sum(free_map)}, occupancy_map sum={np.sum(occupancy_map)}")
        if np.sum(free_map) == 0 and np.sum(occupancy_map) == 0:
            print("No explored space yet, returning empty graph")
            return self.places
        
        # Find frontier indices
        selem = skimage.morphology.disk(1)
        neighbor_unexp_idx = (skimage.filters.rank.minimum(free_map.astype(np.uint8), selem) == 0)
        neighbor_occp_idx = (skimage.filters.rank.maximum(free_map.astype(np.uint8), selem) == 1)
        frontier_idx = neighbor_unexp_idx & neighbor_occp_idx
        wall_hull_map = np.maximum(occupancy_map, frontier_idx)
        
        # Check if wall_hull_map has any boundaries
        self._debug_print(f"DEBUG Topology: wall_hull_map sum={np.sum(wall_hull_map)}, frontier_idx sum={np.sum(frontier_idx)}")
        if np.sum(wall_hull_map) == 0:
            print("No walls/boundaries detected yet, returning empty graph")
            return self.places
        
        # Compute SDF
        boundary_sdf = compute_sdf(wall_hull_map, distance_scale=sdf_scale)
        
        # Check if SDF computation succeeded
        self._debug_print(f"DEBUG Topology: boundary_sdf sum={np.sum(boundary_sdf)}")
        if np.sum(boundary_sdf) == 0:
            print("SDF computation returned empty, returning empty graph")
            return self.places
            
        boundary_sdf[slam.bev_map_occupancy == OCCUPANCY.FREE] = 0
        sdf_inner = compute_sdf(wall_hull_map, distance_scale=1.1)
        if np.sum(sdf_inner) > 0:
            boundary_sdf = np.maximum(boundary_sdf, sdf_inner)

        obstacle_points = np.asarray(np.where(boundary_sdf)).T.astype(np.float32)
        
        self._debug_print(f"DEBUG Topology: obstacle_points count={len(obstacle_points)}")
        if len(obstacle_points) < 4:
            print(f"Not enough obstacle points for Voronoi computation: {len(obstacle_points)}")
            return self.places
            
        vor = Voronoi(obstacle_points)
        self._debug_print(f"DEBUG Topology: Voronoi vertices count={len(vor.vertices)}")
        
        # Filter valid vertices
        clipped_vertices = vor.vertices[np.all(vor.vertices >= 0, axis=1)]
        clipped_vertices = clipped_vertices[np.all(
            clipped_vertices <= max(np.array(bev_map_occupancy.shape) - 1), axis=1
        )]
        self._debug_print(f"DEBUG Topology: clipped_vertices count={len(clipped_vertices)}")
        
        def _ceil(vertices, mask, idx):
            return np.ceil(vertices[:, idx]).astype(int)
        
        def _floor(vertices, mask, idx):
            return np.floor(vertices[:, idx]).astype(int)
        
        if len(clipped_vertices) == 0:
            print("DEBUG Topology: No clipped vertices, returning empty graph")
            return self.places
            
        mask = (boundary_sdf + (bev_map_occupancy == 0)).astype(bool)
        idx1 = mask[_floor(clipped_vertices, mask, 0), _floor(clipped_vertices, mask, 1)] == 0
        idx2 = mask[_floor(clipped_vertices, mask, 0), _ceil(clipped_vertices, mask, 1)] == 0
        idx3 = mask[_ceil(clipped_vertices, mask, 0), _floor(clipped_vertices, mask, 1)] == 0
        idx4 = mask[_ceil(clipped_vertices, mask, 0), _ceil(clipped_vertices, mask, 1)] == 0
        valid = ((idx1.astype(int) + idx2.astype(int) + idx3.astype(int) + idx4.astype(int)) > 3)
        valid_vor_nodes = clipped_vertices[valid]
        self._debug_print(f"DEBUG Topology: valid_vor_nodes count={len(valid_vor_nodes)}")

        # Add nodes
        for vertex in valid_vor_nodes:
            self.places.add_node(tuple(np.round(vertex, 3)))
        self._debug_print(f"DEBUG Topology: nodes after adding={len(self.places.nodes)}")
        
        # Add edges
        ridge_vertices_array = np.array(vor.ridge_vertices)
        simplex_mask = np.all(ridge_vertices_array >= 0, axis=1)
        for simplex in ridge_vertices_array[simplex_mask]:
            v1, v2 = vor.vertices[simplex]
            v1, v2 = tuple(np.round(v1, 3)), tuple(np.round(v2, 3))
            if (v1 in self.places.nodes) and (v2 in self.places.nodes):
                self.places.add_edge(tuple(v1), tuple(v2), dist=np.linalg.norm(np.array(v1) - np.array(v2)))
        
        # Remove nodes outside valid space
        if len(self.places.nodes) > 0:
            nodes_stacked = np.stack(list(self.places.nodes()))
            outside_map = np.logical_or(
                nodes_stacked < 0,
                nodes_stacked >= np.array(wall_hull_map.shape)
            ).any(axis=1)
            
            nodes_outside_free_space = []
            for i, node in enumerate(self.places.nodes()):
                if outside_map[i] or wall_hull_map[int(node[0]), int(node[1])]:
                    nodes_outside_free_space.append(node)
            self._debug_print(f"DEBUG Topology: nodes to remove={len(nodes_outside_free_space)}")
            for node in nodes_outside_free_space:
                self.places.remove_node(node)
        
        self._debug_print(f"DEBUG Topology: final nodes={len(self.places.nodes)}")
        if len(self.places.nodes) == 0:
            print("No nodes in graph, returning empty graph")
        else:
            # Keep largest connected component
            comp_places_subgraphs = [
                self.places.subgraph(c).copy()
                for c in sorted(nx.connected_components(self.places), key=len, reverse=True)
            ]
            self.places = comp_places_subgraphs[0]

        return self.places

    def sparsify_topology_graph(self) -> nx.Graph:
        """Sparsify the topology graph."""
        return sparsify_graph(self.places, self.voxel_size, self.obstacle_map, self.sdf_scale)

    def plot_graph(self, map_underlay: np.ndarray):
        """Plot the topology graph."""
        return plot_graph(self.places, map_underlay)


def detect_rooms(scene, 
                 slam, 
                 graph: nx.Graph, 
                 obstacle_map: np.ndarray, 
                 sdf_scale: float, 
                 thresh: float, 
                 voxel_size: float, 
                 obj_to_neglect: List[str], 
                 opened_windows: set,
                 opened_doors: set = None) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Detect rooms by analyzing door positions and graph structure.
    
    Cuts the voronoi graph at CLOSED door locations to separate rooms.
    Open doors are ignored so that connected spaces remain connected.
    
    Args:
        scene: Scene wrapper
        slam: SLAM module
        graph: Topology graph
        obstacle_map: Obstacle map
        sdf_scale: SDF scale
        thresh: Door detection threshold
        voxel_size: Voxel size
        obj_to_neglect: Objects to ignore
        opened_windows: Opened window names
        opened_doors: Set of opened door names (these won't cause graph cuts)
        
    Returns:
        Tuple of (separated graph, door probability map, door positions)
    """
    if opened_doors is None:
        opened_doors = set()
        
    graph = copy.deepcopy(graph)
    boundary_sdf = compute_sdf(obstacle_map, distance_scale=sdf_scale)

    # Manifold for door probability estimate
    xmin, xmax = 0, slam.bev_map_occupancy.shape[0]
    ymin, ymax = 0, slam.bev_map_occupancy.shape[1]
    xx, yy = np.mgrid[xmin:xmax:complex(0, xmax), ymin:ymax:complex(0, ymax)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    door_positions = []
    
    # Find CLOSED doors only - open doors should not separate rooms
    for instance_id in slam.seen_instances:
        body_property = get_body_properties(scene, instance_id, obj_to_neglect, opened_windows)
        if body_property is not None:
            if body_property["semantic_class_name"] == "door":
                door_name = body_property.get("name", "")
                # Only cut graph at CLOSED doors
                if door_name not in opened_doors:
                    # Habitat uses Y-up coordinate system: position is (X, Y, Z)
                    # For 2D map, we need (X, Z) which is the horizontal plane
                    pos = body_property["pos"]
                    pos_2d = np.array([pos[0], pos[2]])  # X, Z
                    door_positions.append(slam.world2voxel(pos_2d))
    
    if len(door_positions) > 0:
        # Kernel density estimation around doors
        door_pos = np.array(door_positions).reshape(-1, 2)
        kde = KernelDensity(kernel='gaussian', bandwidth=2.0)
        kde.fit(door_pos)
        door_prob = np.exp(kde.score_samples(positions.T).reshape(xx.shape))

        edges_tbd = []
        for edge in list(graph.edges()):
            x1, y1 = edge[0]
            x2, y2 = edge[1]
            edge_score = boundary_integral(door_prob, int(x1), int(y1), int(x2), int(y2), normalize=False)
            if edge_score > thresh:
                edges_tbd.append(edge)
        
        graph.remove_edges_from(edges_tbd)
        graph.remove_nodes_from(list(nx.isolates(graph)))

        del_prob = door_prob
    else:
        door_pos = np.empty((0, 2))
        del_prob = boundary_sdf

    # Filter small components
    components = list(nx.connected_components(graph))
    if len(components) > 0 and len(graph.nodes) > 4:
        for c in components:
            if len(c) < 10:
                edges = list(graph.subgraph(c).edges())
                cum_edge_len = sum([graph.get_edge_data(e[0], e[1])["dist"] for e in edges]) * voxel_size
                if cum_edge_len < 0.5:
                    graph.remove_nodes_from(c)

    return graph, del_prob, door_pos


# Aliases for compatibility
TopologyMapping = HabitatTopologyMapping

