# SmallPlan Tasks Module
# Provides task definitions for both iGibson and Habitat

# Try to import iGibson-compatible tasks
try:
    from .object_search_task import ObjectSearchTask as IGibsonObjectSearchTask
    from .patched_scene import (
        MonkeyPatchedInteractiveIndoorScene,
        CLASS_NAME_TO_CLASS_ID as IGIBSON_CLASS_NAME_TO_CLASS_ID,
        REPLACED_CATEGORIES
    )
    from .object_sampling import add_objects_from_our_distribution as igibson_add_objects
    IGIBSON_TASKS_AVAILABLE = True
except ImportError:
    IGIBSON_TASKS_AVAILABLE = False
    IGibsonObjectSearchTask = None
    MonkeyPatchedInteractiveIndoorScene = None
    IGIBSON_CLASS_NAME_TO_CLASS_ID = {}
    igibson_add_objects = None

# Try to import Habitat-compatible tasks
try:
    from .habitat_object_search_task import (
        HabitatObjectSearchTask,
        ObjectSearchTask as HabitatObjectSearchTaskAlias
    )
    from .habitat_patched_scene import (
        CLASS_NAME_TO_CLASS_ID as HABITAT_CLASS_NAME_TO_CLASS_ID,
        CLASS_ID_TO_CLASS_NAME as HABITAT_CLASS_ID_TO_CLASS_NAME,
        get_class_name_to_class_id,
        HabitatScenePatcher,
        REPLACED_CATEGORIES as HABITAT_REPLACED_CATEGORIES
    )
    from .habitat_object_sampling import (
        add_objects_from_our_distribution as habitat_add_objects,
        OBJECT_DISTRIBUTION_PRIOR,
        is_receptacle,
        get_objects_by_category,
        get_objects_in_room,
        get_receptacle_objects
    )
    HABITAT_TASKS_AVAILABLE = True
except ImportError:
    HABITAT_TASKS_AVAILABLE = False
    HabitatObjectSearchTask = None
    HABITAT_CLASS_NAME_TO_CLASS_ID = {}
    HABITAT_CLASS_ID_TO_CLASS_NAME = {}
    habitat_add_objects = None

__all__ = [
    "IGIBSON_TASKS_AVAILABLE",
    "HABITAT_TASKS_AVAILABLE",
]

if IGIBSON_TASKS_AVAILABLE:
    __all__.extend([
        "IGibsonObjectSearchTask",
        "MonkeyPatchedInteractiveIndoorScene",
        "IGIBSON_CLASS_NAME_TO_CLASS_ID",
        "igibson_add_objects",
        "REPLACED_CATEGORIES",
    ])

if HABITAT_TASKS_AVAILABLE:
    __all__.extend([
        "HabitatObjectSearchTask",
        "HABITAT_CLASS_NAME_TO_CLASS_ID",
        "HABITAT_CLASS_ID_TO_CLASS_NAME",
        "get_class_name_to_class_id",
        "HabitatScenePatcher",
        "habitat_add_objects",
        "OBJECT_DISTRIBUTION_PRIOR",
        "is_receptacle",
        "get_objects_by_category",
        "get_objects_in_room",
        "get_receptacle_objects",
    ])

# Provide a default ObjectSearchTask based on what's available
if HABITAT_TASKS_AVAILABLE:
    ObjectSearchTask = HabitatObjectSearchTask
    CLASS_NAME_TO_CLASS_ID = HABITAT_CLASS_NAME_TO_CLASS_ID
elif IGIBSON_TASKS_AVAILABLE:
    ObjectSearchTask = IGibsonObjectSearchTask
    CLASS_NAME_TO_CLASS_ID = IGIBSON_CLASS_NAME_TO_CLASS_ID
else:
    ObjectSearchTask = None
    CLASS_NAME_TO_CLASS_ID = {}

