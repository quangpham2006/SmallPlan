# Habitat-compatible Room Graph module for SmallPlan
# Provides room-object graph construction without iGibson dependencies

import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Any, Optional, Set

from moma_llm.llm.habitat_llm import object_states
from moma_llm.utils.habitat_constants import HABITAT_SEMANTIC_CLASSES, NODETYPE


def get_body_properties(scene, obj_id: int, obj_to_neglect: List[str], opened_windows: Set[str]) -> Optional[Dict]:
    """
    Get properties of an object in Habitat scene.
    
    Args:
        scene: Habitat scene wrapper
        obj_id: Object instance ID
        obj_to_neglect: List of object names to ignore
        opened_windows: Set of opened window names
        
    Returns:
        Dictionary of object properties or None
    """
    obj = scene.get_object_by_id(obj_id)
    if obj is None:
        print(f"object_id {obj_id} not found in scene")
        return None
    
    # Get object name and category
    obj_name = getattr(obj, 'name', None) or getattr(obj, 'handle', f'object_{obj_id}')
    obj_category = getattr(obj, 'category', None) or getattr(obj, 'semantic_class', 'unknown')
    
    # Skip floor/wall/ceiling objects
    if obj_name in ["floors", "walls", "ceilings", "floor", "wall", "ceiling"]:
        return None
    if obj_name in obj_to_neglect:
        return None
    
    # Helper to get position from object
    def _get_obj_position(o):
        if hasattr(o, 'get_position') and callable(o.get_position):
            return np.asarray(o.get_position(), dtype=np.float32)
        elif hasattr(o, '_position'):
            return np.asarray(o._position, dtype=np.float32)
        elif hasattr(o, 'position'):
            p = o.position
            if callable(p):
                return np.asarray(p(), dtype=np.float32)
            return np.asarray(p, dtype=np.float32)
        return np.zeros(3, dtype=np.float32)
    
    # Get bounding box
    try:
        if hasattr(obj, 'get_base_aligned_bounding_box'):
            pos, orn, bbox_extent, _ = obj.get_base_aligned_bounding_box()
        elif hasattr(obj, 'aabb'):
            aabb = obj.aabb
            # Handle both property and method access for min/max
            aabb_min = aabb.min() if callable(aabb.min) else aabb.min
            aabb_max = aabb.max() if callable(aabb.max) else aabb.max
            pos = (np.array(aabb_min) + np.array(aabb_max)) / 2
            bbox_extent = np.array(aabb_max) - np.array(aabb_min)
            orn = np.array([0, 0, 0, 1])  # Default quaternion
        else:
            # Fallback: use object position
            pos = _get_obj_position(obj)
            bbox_extent = np.ones(3) * 0.5
            orn = np.array([0, 0, 0, 1])
    except Exception as e:
        print(f"Error getting bounding box for {obj_name}: {e}")
        pos = _get_obj_position(obj)
        bbox_extent = np.ones(3) * 0.5
        orn = np.array([0, 0, 0, 1])
    
    # Get object states
    states = {}
    
    # Handle Open state
    if hasattr(obj, 'states'):
        obj_states = obj.states
        if hasattr(obj_states, 'get'):
            open_state = obj_states.get(object_states.Open)
            if open_state is not None:
                if obj_category == "window":
                    states[object_states.Open] = (obj_name in opened_windows)
                else:
                    if hasattr(open_state, 'get_value'):
                        states[object_states.Open] = open_state.get_value()
                    else:
                        states[object_states.Open] = bool(open_state)
    elif hasattr(obj, 'is_open'):
        if obj_category == "window":
            states[object_states.Open] = (obj_name in opened_windows)
        else:
            states[object_states.Open] = obj.is_open
    
    # Ensure pos is a proper array (handle callable pos)
    if callable(pos):
        pos = pos()
    pos_array = np.atleast_1d(np.asarray(pos, dtype=np.float32))
    if pos_array.size < 3:
        pos_array = np.concatenate([pos_array, np.zeros(3 - pos_array.size)])
    
    properties = {
        "bbox": np.array(bbox_extent, dtype=np.float32) if hasattr(bbox_extent, '__len__') else np.ones(3, dtype=np.float32) * bbox_extent,
        "semantic_class_name": obj_category,
        "pos": tuple(pos_array[:3]),
        "orn": np.asarray(orn, dtype=np.float32) if hasattr(orn, '__len__') else np.array([0, 0, 0, 1], dtype=np.float32),
        "name": obj_name,
        "states": states,
    }
    return properties


def get_seen_object_nodes(scene, slam, obj_to_neglect: List[str], opened_windows: Set[str]) -> Dict:
    """
    Get all seen object nodes from SLAM data.
    
    Args:
        scene: Scene wrapper
        slam: SLAM module
        obj_to_neglect: Objects to ignore
        opened_windows: Set of opened windows
        
    Returns:
        Dictionary mapping positions to object properties
    """
    seen_instance_list = list(slam.seen_instances)
    
    body_properties = {}
    
    for instance_id in seen_instance_list:
        if instance_id == 0:  # Skip background
            continue
            
        body_property = get_body_properties(scene, instance_id, obj_to_neglect, opened_windows)
        if body_property is not None:
            body_properties[tuple(body_property["pos"])] = body_property
            body_property["instance_id"] = instance_id
    
    return body_properties


def get_closest_node(query_coords_world: np.ndarray, 
                     graph: nx.Graph, 
                     slam, 
                     dist_thresh: Optional[float] = None) -> Tuple:
    """
    Find closest graph node(s) to query coordinates.
    
    Args:
        query_coords_world: World coordinates to query
        graph: Graph with node positions
        slam: SLAM module for coordinate conversion
        dist_thresh: Optional distance threshold
        
    Returns:
        Tuple of (closest nodes, distances)
    """
    if len(graph.nodes) == 0:
        if dist_thresh is not None:
            return [[] for _ in range(len(query_coords_world))], [[] for _ in range(len(query_coords_world))]
        else:
            return np.array([]), np.array([])
    
    graph_node_pos = np.stack(list(graph.nodes))
    node_coords_world = slam.voxel2world(graph_node_pos)
    
    # Ensure query_coords_world is 2D
    if query_coords_world.ndim == 1:
        query_coords_world = query_coords_world.reshape(1, -1)
    
    dist_matrix = distance_matrix(query_coords_world, node_coords_world)
    
    if dist_thresh is not None:
        closest_nodes, closest_nodes_dists = [], []
        for i in range(len(query_coords_world)):
            idx = dist_matrix[i] < dist_thresh
            closest_nodes.append(graph_node_pos[idx])
            closest_nodes_dists.append(dist_matrix[i][idx])
        return closest_nodes, closest_nodes_dists
    else:
        closest_node_idx = np.argmin(dist_matrix, axis=1)
        closest_nodes_dist = np.min(dist_matrix, axis=1)
        closest_nodes = graph_node_pos[closest_node_idx]
        return closest_nodes, closest_nodes_dist


def map_open_doors_to_components(scene, 
                                  slam, 
                                  separated_vor_graph: nx.Graph, 
                                  opened_doors: Set[str]) -> Dict:
    """
    Map open doors to connected components in voronoi graph.
    
    Args:
        scene: Scene wrapper
        slam: SLAM module
        separated_vor_graph: Separated voronoi graph
        opened_doors: Set of opened door names
        
    Returns:
        Dictionary mapping door names to component IDs
    """
    if len(opened_doors) == 0:
        return {}
    
    open_door_pos = {}
    
    for door in opened_doors:
        obj = scene.get_object_by_name(door)
        if obj is None:
            continue
            
        try:
            if hasattr(obj, 'get_base_aligned_bounding_box'):
                pos, _, _, _ = obj.get_base_aligned_bounding_box()
            elif hasattr(obj, 'position'):
                pos = obj.position
            else:
                continue
            open_door_pos[door] = slam.world2voxel(np.array(pos))[:2]
        except Exception as e:
            print(f"Warning: Could not get position for door {door}: {e}")
            continue
    
    if len(open_door_pos) == 0:
        return {}
        
    open_door_positions = np.stack(list(open_door_pos.values()))
    
    # Compute min distance between each component and the open doors
    c_min_dists = {}
    components = list(nx.connected_components(separated_vor_graph))
    
    for c_id, c_nodes in enumerate(components):
        c_pos = np.array(list(c_nodes))
        c_door_dists = cdist(open_door_positions, c_pos, metric="euclidean")
        c_min_dist = np.min(c_door_dists, axis=1)
        c_min_dists[c_id] = c_min_dist
    
    # Get the two closest components to each open door
    c_min_dists_array = np.array(list(c_min_dists.values()))
    two_closest_c = np.argsort(c_min_dists_array, axis=0)[:2].T
    
    two_closest_doors = {}
    for d, c in zip(opened_doors, two_closest_c):
        two_closest_doors[d] = list(c)
        
    return two_closest_doors


def create_room_object_graph(scene, 
                              slam, 
                              vor_graph: nx.Graph, 
                              separated_vor_graph: nx.Graph, 
                              obj_to_neglect: List[str], 
                              opened_doors: Set[str], 
                              opened_windows: Set[str], 
                              use_viewpoint_assignment: bool,
                              verbose: bool = False) -> Tuple[nx.Graph, nx.DiGraph]:
    """
    Create room-object hierarchical graph.
    
    Args:
        scene: Scene wrapper
        slam: SLAM module
        vor_graph: Voronoi graph
        separated_vor_graph: Separated voronoi graph
        obj_to_neglect: Objects to ignore
        opened_doors: Set of opened doors
        opened_windows: Set of opened windows
        use_viewpoint_assignment: Whether to use viewpoint-based assignment
        
    Returns:
        Tuple of (room graph, room-object graph)
    """
    room_graph = nx.Graph()
    room_object_graph = nx.DiGraph()
    
    room_object_graph.add_node("root", node_type=NODETYPE.ROOT, pos_map=(0, 0, 0))
    components = list(nx.connected_components(separated_vor_graph))
    
    for c_id, c_nodes in enumerate(components):
        # Add room IDs to nodes
        for node in c_nodes:
            separated_vor_graph.nodes[node]['room_id'] = c_id
        
        # Add room nodes
        room_subgraph = separated_vor_graph.subgraph(c_nodes).copy()
        room_center_node = nx.center(room_subgraph)[0] if len(room_subgraph) > 0 else list(c_nodes)[0]
        
        room_graph.add_node(
            NODETYPE.roomname(c_id),
            pos=tuple(slam.voxel2world(np.array(room_center_node))),
            pos_map=tuple(room_center_node),
            room_id=c_id,
            node_type=NODETYPE.ROOM
        )
        room_object_graph.add_node(
            NODETYPE.roomname(c_id),
            pos=tuple(slam.voxel2world(np.array(room_center_node))),
            pos_map=tuple(room_center_node),
            room_id=c_id,
            node_type=NODETYPE.ROOM,
            frontier_points=set(),
            closed_doors=set(),
            open_doors=set()
        )
        room_object_graph.add_edge("root", NODETYPE.roomname(c_id))
    
    # Construct neighborhood-respective room graph
    for c_id, c_node in enumerate(room_graph.nodes):
        for c_id2, c_node2 in enumerate(room_graph.nodes):
            if c_id < c_id2:
                try:
                    path = nx.shortest_path(
                        vor_graph,
                        source=tuple(room_graph.nodes[c_node]["pos_map"]),
                        target=tuple(room_graph.nodes[c_node2]["pos_map"])
                    )
                    rooms_traveled = set()
                    for node in path:
                        if node in separated_vor_graph.nodes:
                            rooms_traveled.add(separated_vor_graph.nodes[node]["room_id"])
                    if len(rooms_traveled) < 3:
                        room_graph.add_edge(NODETYPE.roomname(c_id), NODETYPE.roomname(c_id2))
                except nx.NetworkXNoPath:
                    continue
    
    object_room_assignment(
        scene=scene,
        slam=slam,
        vor_graph=vor_graph,
        separated_vor_graph=separated_vor_graph,
        room_object_graph=room_object_graph,
        obj_to_neglect=obj_to_neglect,
        opened_doors=opened_doors,
        opened_windows=opened_windows,
        use_viewpoint_assignment=use_viewpoint_assignment,
        verbose=verbose
    )
    
    return room_graph, room_object_graph


def object_room_assignment(scene, 
                           slam, 
                           vor_graph: nx.Graph, 
                           separated_vor_graph: nx.Graph, 
                           room_object_graph: nx.DiGraph, 
                           obj_to_neglect: List[str], 
                           opened_doors: Set[str], 
                           opened_windows: Set[str], 
                           use_viewpoint_assignment: bool,
                           verbose: bool = False):
    """
    Assign objects to rooms in the graph.
    
    Args:
        scene: Scene wrapper
        slam: SLAM module
        vor_graph: Voronoi graph
        separated_vor_graph: Separated voronoi graph
        room_object_graph: Room-object graph to update
        obj_to_neglect: Objects to ignore
        opened_doors: Opened door names
        opened_windows: Opened window names
        use_viewpoint_assignment: Use viewpoint-based assignment
        verbose: Enable debug output
    """
    # Get all seen objects
    object_nodes = get_seen_object_nodes(scene, slam, obj_to_neglect, opened_windows)
    if len(object_nodes) == 0:
        return
    
    door2comp = map_open_doors_to_components(scene, slam, separated_vor_graph, opened_doors)
    
    object_closeness_thresh = 10.0  # Increased from 2.5 to debug coordinate issues
    vp_closeness_thresh = 15.0  # Increased from 5.0
    
    # Get viewpoint coordinates
    # In Habitat: Y is UP, so we use X and Z (indices 0 and 2) for horizontal plane
    viewpoint_coords_list = []
    for on in object_nodes.values():
        instance_id = on["instance_id"]
        if instance_id in slam.instance_viewpoints:
            vp = slam.instance_viewpoints[instance_id][0]
            viewpoint_coords_list.append(np.array([vp[0], vp[2]]))  # X, Z
        else:
            pos = on["pos"]
            viewpoint_coords_list.append(np.array([pos[0], pos[2]]))  # X, Z
    
    if len(viewpoint_coords_list) == 0:
        return
        
    viewpoint_coords_world = np.stack(viewpoint_coords_list)
    vp_closer_than_thresh, vp_closest_nodes_dists = get_closest_node(
        viewpoint_coords_world, graph=separated_vor_graph, slam=slam, dist_thresh=vp_closeness_thresh
    )
    
    # Use X and Z (indices 0 and 2) for horizontal plane in Habitat
    object_pos_list = list(object_nodes.keys())
    object_coords_world = np.array([[p[0], p[2]] for p in object_pos_list])
    closest_nodes, closest_nodes_dists = get_closest_node(
        object_coords_world, graph=separated_vor_graph, slam=slam, dist_thresh=object_closeness_thresh
    )
    
    # Debug: print coordinate ranges and minimum distances
    if verbose:
        from scipy.spatial import distance_matrix as dist_mat
        graph_node_pos = np.stack(list(separated_vor_graph.nodes))
        node_coords_world = slam.voxel2world(graph_node_pos)
        print(f"DEBUG Room Assignment: Graph nodes voxel range: [{graph_node_pos.min(axis=0)}, {graph_node_pos.max(axis=0)}]")
        print(f"DEBUG Room Assignment: Graph nodes world range: X=[{node_coords_world[:, 0].min():.2f}, {node_coords_world[:, 0].max():.2f}], Z=[{node_coords_world[:, 1].min():.2f}, {node_coords_world[:, 1].max():.2f}]")
        print(f"DEBUG Room Assignment: Object coords world range: X=[{object_coords_world[:, 0].min():.2f}, {object_coords_world[:, 0].max():.2f}], Z=[{object_coords_world[:, 1].min():.2f}, {object_coords_world[:, 1].max():.2f}]")
        print(f"DEBUG Room Assignment: Sample graph node (world): {node_coords_world[0]}")
        print(f"DEBUG Room Assignment: Sample object coord (world): {object_coords_world[0]}")
        print(f"DEBUG Room Assignment: voxel_size={slam.voxel_size}, midpoint={slam.midpoint}")
        # Check minimum distances
        dm = dist_mat(object_coords_world, node_coords_world)
        min_dists = dm.min(axis=1)
        print(f"DEBUG Room Assignment: Min distances to graph: min={min_dists.min():.2f}, max={min_dists.max():.2f}, mean={min_dists.mean():.2f}")
        print(f"DEBUG Room Assignment: Objects within threshold: {np.sum(min_dists < object_closeness_thresh)} / {len(min_dists)}")
    
    all_path_sep_voronoi = dict(nx.all_pairs_dijkstra_path_length(vor_graph, weight="dist"))
    
    # Add all objects as children of their room
    for i, node_prop in enumerate(object_nodes.values()):
        nodes_closer_than_thresh = closest_nodes[i]
        nodes_closer_than_thresh_dists = closest_nodes_dists[i]
        
        if len(nodes_closer_than_thresh) == 0:
            print(f"Object {node_prop['name']} too far from room graph - not assigned")
            continue
        
        if use_viewpoint_assignment:
            dists_to_vp = []
            for n, nd in zip(nodes_closer_than_thresh, nodes_closer_than_thresh_dists):
                min_dist = np.inf
                for closest_vp, closest_vp_d in zip(vp_closer_than_thresh[i], vp_closest_nodes_dists[i]):
                    path_dist = all_path_sep_voronoi.get(tuple(closest_vp), {}).get(tuple(n), np.inf)
                    total_dist = path_dist + (nd / slam.voxel_size)**1.3 + (closest_vp_d / slam.voxel_size)
                    min_dist = min(min_dist, total_dist)
                dists_to_vp.append(min_dist)
            
            if np.min(dists_to_vp) == np.inf:
                print(f"Object {node_prop['name']} not reachable from any viewpoint")
                continue
            node_closest_to_viewpoint = nodes_closer_than_thresh[np.argmin(dists_to_vp)]
        else:
            node_closest_to_viewpoint = nodes_closer_than_thresh[np.argmin(nodes_closer_than_thresh_dists)]
        
        node_prop["room_id"] = separated_vor_graph.nodes.get(tuple(node_closest_to_viewpoint), {}).get("room_id", 0)
        node_prop["pos_map"] = tuple(slam.world2voxel(np.array(node_prop["pos"])))
        node_prop["closest_vor_node"] = tuple(node_closest_to_viewpoint)
        
        if node_prop["semantic_class_name"] == "door":
            if node_prop["name"] not in opened_doors:
                room_node = room_object_graph.nodes.get(NODETYPE.roomname(node_prop["room_id"]))
                if room_node:
                    room_node["closed_doors"].add(node_prop["name"])
            else:
                comps = door2comp.get(node_prop["name"], [])
                if len(comps) == 1:
                    room_object_graph.nodes[NODETYPE.roomname(comps[0])]["open_doors"].add((node_prop["name"], None))
                elif len(comps) == 2:
                    room_object_graph.nodes[NODETYPE.roomname(comps[0])]["open_doors"].add(
                        (node_prop["name"], NODETYPE.roomname(comps[1]))
                    )
                    room_object_graph.nodes[NODETYPE.roomname(comps[1])]["open_doors"].add(
                        (node_prop["name"], NODETYPE.roomname(comps[0]))
                    )
                continue
        
        room_object_graph.add_node(node_prop["name"], **node_prop, node_type=NODETYPE.OBJECT)
        room_object_graph.add_edge(NODETYPE.roomname(node_prop["room_id"]), node_prop["name"])

