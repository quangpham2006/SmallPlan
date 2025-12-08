# SmallPlan Tasks Module
# Provides task definitions for Habitat

# Import Habitat-compatible tasks
try:
    from .habitat_object_search_task import HabitatObjectSearchTask
    from .habitat_patched_scene import (
        CLASS_NAME_TO_CLASS_ID,
        CLASS_ID_TO_CLASS_NAME,
        get_class_name_to_class_id,
        HabitatScenePatcher,
        REPLACED_CATEGORIES
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
    ObjectSearchTask = HabitatObjectSearchTask
except ImportError:
    HABITAT_TASKS_AVAILABLE = False
    HabitatObjectSearchTask = None
    CLASS_NAME_TO_CLASS_ID = {}
    CLASS_ID_TO_CLASS_NAME = {}
    get_class_name_to_class_id = None
    HabitatScenePatcher = None
    REPLACED_CATEGORIES = {}
    habitat_add_objects = None
    OBJECT_DISTRIBUTION_PRIOR = None
    is_receptacle = None
    get_objects_by_category = None
    get_objects_in_room = None
    get_receptacle_objects = None
    ObjectSearchTask = None

__all__ = [
    "HABITAT_TASKS_AVAILABLE",
]

if HABITAT_TASKS_AVAILABLE:
    __all__.extend([
        "HabitatObjectSearchTask",
        "ObjectSearchTask",
        "CLASS_NAME_TO_CLASS_ID",
        "CLASS_ID_TO_CLASS_NAME",
        "get_class_name_to_class_id",
        "HabitatScenePatcher",
        "habitat_add_objects",
        "OBJECT_DISTRIBUTION_PRIOR",
        "is_receptacle",
        "get_objects_by_category",
        "get_objects_in_room",
        "get_receptacle_objects",
        "REPLACED_CATEGORIES",
    ])

