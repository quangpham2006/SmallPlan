# Topology module exports for SmallPlan
# Provides imports for Habitat

# Import Habitat topology modules
try:
    from .habitat_simpleslam import HabitatSimpleSlam
    from .habitat_room_graph import (
        get_body_properties,
        get_seen_object_nodes,
        get_closest_node,
        map_open_doors_to_components,
        create_room_object_graph,
        object_room_assignment
    )
    from .habitat_topology import (
        HabitatTopologyMapping,
        detect_rooms,
        boundary_integral,
        compute_sdf,
        quat2rotmat,
        xyzw2wxyz,
        wxyz2xyzw,
        rotmat2quat
    )
    HABITAT_TOPOLOGY_AVAILABLE = True
    SimpleSlam = HabitatSimpleSlam
    TopologyMapping = HabitatTopologyMapping
except ImportError:
    HABITAT_TOPOLOGY_AVAILABLE = False
    HabitatSimpleSlam = None
    HabitatTopologyMapping = None
    detect_rooms = None
    get_body_properties = None
    get_seen_object_nodes = None
    get_closest_node = None
    map_open_doors_to_components = None
    create_room_object_graph = None
    object_room_assignment = None
    boundary_integral = None
    compute_sdf = None
    quat2rotmat = None
    xyzw2wxyz = None
    wxyz2xyzw = None
    rotmat2quat = None
    SimpleSlam = None
    TopologyMapping = None

# Always import graph utilities (no simulator dependency)
from .graph import sparsify_graph, plot_graph

__all__ = [
    # Availability flag
    "HABITAT_TOPOLOGY_AVAILABLE",
    # Default exports
    "SimpleSlam",
    "TopologyMapping",
    "detect_rooms",
    "get_body_properties",
    "get_seen_object_nodes",
    "get_closest_node",
    "map_open_doors_to_components",
    "create_room_object_graph",
    "object_room_assignment",
    "boundary_integral",
    "compute_sdf",
    "quat2rotmat",
    "xyzw2wxyz",
    "wxyz2xyzw",
    "rotmat2quat",
    # Graph utilities
    "sparsify_graph",
    "plot_graph",
]

if HABITAT_TOPOLOGY_AVAILABLE:
    __all__.extend([
        "HabitatSimpleSlam",
        "HabitatTopologyMapping",
        "habitat_detect_rooms",
    ])
