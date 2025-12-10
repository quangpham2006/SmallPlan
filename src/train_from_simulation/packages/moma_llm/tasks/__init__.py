# SmallPlan Tasks Module
# iGibson-only version

from .object_search_task import ObjectSearchTask
from .patched_scene import (
    MonkeyPatchedInteractiveIndoorScene,
    CLASS_NAME_TO_CLASS_ID,
    REPLACED_CATEGORIES
)
from .object_sampling import add_objects_from_our_distribution

__all__ = [
    "ObjectSearchTask",
    "MonkeyPatchedInteractiveIndoorScene",
    "CLASS_NAME_TO_CLASS_ID",
    "REPLACED_CATEGORIES",
    "add_objects_from_our_distribution",
]
