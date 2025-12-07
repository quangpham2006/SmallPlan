# Topology module exports for SmallPlan
# Provides conditional imports for iGibson and Habitat

# Try to import iGibson topology modules
try:
    from .simpleslam import SimpleSlam as IGibsonSimpleSlam
    from .room_graph import (
        get_body_properties as igibson_get_body_properties,
        get_seen_object_nodes as igibson_get_seen_object_nodes,
        get_closest_node as igibson_get_closest_node,
        map_open_doors_to_components as igibson_map_open_doors_to_components,
        create_room_object_graph as igibson_create_room_object_graph,
        object_room_assigment as igibson_object_room_assignment
    )
    from .topology import (
        TopologyMapping as IGibsonTopologyMapping,
        detect_rooms as igibson_detect_rooms,
        boundary_integral,
        compute_sdf
    )
    IGIBSON_TOPOLOGY_AVAILABLE = True
except ImportError:
    IGibsonSimpleSlam = None
    IGibsonTopologyMapping = None
    igibson_detect_rooms = None
    IGIBSON_TOPOLOGY_AVAILABLE = False

# Try to import Habitat topology modules
try:
    from .habitat_simpleslam import HabitatSimpleSlam
    from .habitat_room_graph import (
        get_body_properties as habitat_get_body_properties,
        get_seen_object_nodes as habitat_get_seen_object_nodes,
        get_closest_node as habitat_get_closest_node,
        map_open_doors_to_components as habitat_map_open_doors_to_components,
        create_room_object_graph as habitat_create_room_object_graph,
        object_room_assignment as habitat_object_room_assignment
    )
    from .habitat_topology import (
        HabitatTopologyMapping,
        detect_rooms as habitat_detect_rooms,
        boundary_integral,
        compute_sdf,
        quat2rotmat,
        xyzw2wxyz,
        wxyz2xyzw,
        rotmat2quat
    )
    HABITAT_TOPOLOGY_AVAILABLE = True
except ImportError:
    HabitatSimpleSlam = None
    HabitatTopologyMapping = None
    habitat_detect_rooms = None
    HABITAT_TOPOLOGY_AVAILABLE = False

# Always import graph utilities (no simulator dependency)
from .graph import sparsify_graph, plot_graph

# Provide default exports based on availability
if HABITAT_TOPOLOGY_AVAILABLE:
    SimpleSlam = HabitatSimpleSlam
    TopologyMapping = HabitatTopologyMapping
    detect_rooms = habitat_detect_rooms
    get_body_properties = habitat_get_body_properties
    get_seen_object_nodes = habitat_get_seen_object_nodes
    get_closest_node = habitat_get_closest_node
    map_open_doors_to_components = habitat_map_open_doors_to_components
    create_room_object_graph = habitat_create_room_object_graph
    object_room_assignment = habitat_object_room_assignment
elif IGIBSON_TOPOLOGY_AVAILABLE:
    SimpleSlam = IGibsonSimpleSlam
    TopologyMapping = IGibsonTopologyMapping
    detect_rooms = igibson_detect_rooms
    get_body_properties = igibson_get_body_properties
    get_seen_object_nodes = igibson_get_seen_object_nodes
    get_closest_node = igibson_get_closest_node
    map_open_doors_to_components = igibson_map_open_doors_to_components
    create_room_object_graph = igibson_create_room_object_graph
    object_room_assignment = igibson_object_room_assignment
else:
    SimpleSlam = None
    TopologyMapping = None
    detect_rooms = None
    get_body_properties = None
    get_seen_object_nodes = None
    get_closest_node = None
    map_open_doors_to_components = None
    create_room_object_graph = None
    object_room_assignment = None

__all__ = [
    # Availability flags
    "IGIBSON_TOPOLOGY_AVAILABLE",
    "HABITAT_TOPOLOGY_AVAILABLE",
    # Default exports (based on availability)
    "SimpleSlam",
    "TopologyMapping",
    "detect_rooms",
    "get_body_properties",
    "get_seen_object_nodes",
    "get_closest_node",
    "map_open_doors_to_components",
    "create_room_object_graph",
    "object_room_assignment",
    # Graph utilities
    "sparsify_graph",
    "plot_graph",
]

# Add simulator-specific exports to __all__
if IGIBSON_TOPOLOGY_AVAILABLE:
    __all__.extend([
        "IGibsonSimpleSlam",
        "IGibsonTopologyMapping",
        "igibson_detect_rooms",
    ])

if HABITAT_TOPOLOGY_AVAILABLE:
    __all__.extend([
        "HabitatSimpleSlam",
        "HabitatTopologyMapping",
        "habitat_detect_rooms",
        "quat2rotmat",
        "xyzw2wxyz",
        "wxyz2xyzw",
        "rotmat2quat",
    ])
