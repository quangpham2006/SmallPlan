# Habitat Patched Scene - Semantic Class Mappings
# Provides semantic class mappings compatible with Habitat-Lab/Habitat-Sim
# Replaces iGibson-specific patched_scene.py

import logging
from collections import OrderedDict
from typing import Dict, Any, Optional, List

log = logging.getLogger(__name__)


# ============================================================================
# Category Replacements (for consistency with iGibson naming)
# ============================================================================

REPLACED_CATEGORIES = {
    "bottom_cabinet_no_top": "bottom_cabinet",
    "countertop": {"bedroom": "nightstand", "kitchen": "kitchen_counter", "childs_room": "nightstand"},
    "chest": {"bedroom": "dresser", "childs_room": "dresser"},
    "breakfast_table": "table",
}


# ============================================================================
# Habitat Semantic Classes
# Mapping from class names to class IDs for various Habitat datasets
# ============================================================================

# HM3D semantic classes (Habitat-Matterport 3D)
HM3D_CLASS_NAME_TO_CLASS_ID = OrderedDict([
    ("void", 0),
    ("wall", 1),
    ("floor", 2),
    ("chair", 3),
    ("door", 4),
    ("table", 5),
    ("picture", 6),
    ("cabinet", 7),
    ("cushion", 8),
    ("window", 9),
    ("sofa", 10),
    ("bed", 11),
    ("curtain", 12),
    ("chest_of_drawers", 13),
    ("plant", 14),
    ("sink", 15),
    ("stairs", 16),
    ("ceiling", 17),
    ("toilet", 18),
    ("stool", 19),
    ("towel", 20),
    ("mirror", 21),
    ("tv_monitor", 22),
    ("shower", 23),
    ("column", 24),
    ("bathtub", 25),
    ("counter", 26),
    ("fireplace", 27),
    ("lighting", 28),
    ("beam", 29),
    ("railing", 30),
    ("shelving", 31),
    ("blinds", 32),
    ("gym_equipment", 33),
    ("seating", 34),
    ("board_panel", 35),
    ("furniture", 36),
    ("appliances", 37),
    ("clothes", 38),
    ("objects", 39),
    ("misc", 40),
])

# MP3D semantic classes (Matterport3D)
MP3D_CLASS_NAME_TO_CLASS_ID = OrderedDict([
    ("void", 0),
    ("wall", 1),
    ("floor", 2),
    ("chair", 3),
    ("door", 4),
    ("table", 5),
    ("picture", 6),
    ("cabinet", 7),
    ("cushion", 8),
    ("window", 9),
    ("sofa", 10),
    ("bed", 11),
    ("curtain", 12),
    ("chest_of_drawers", 13),
    ("plant", 14),
    ("sink", 15),
    ("stairs", 16),
    ("ceiling", 17),
    ("toilet", 18),
    ("stool", 19),
    ("towel", 20),
    ("mirror", 21),
    ("tv_monitor", 22),
    ("shower", 23),
    ("column", 24),
    ("bathtub", 25),
    ("counter", 26),
    ("fireplace", 27),
    ("lighting", 28),
    ("beam", 29),
    ("railing", 30),
    ("shelving", 31),
    ("blinds", 32),
    ("gym_equipment", 33),
    ("seating", 34),
    ("board_panel", 35),
    ("furniture", 36),
    ("appliances", 37),
    ("clothes", 38),
    ("objects", 39),
    ("misc", 40),
])

# Gibson semantic classes (simplified)
GIBSON_CLASS_NAME_TO_CLASS_ID = OrderedDict([
    ("void", 0),
    ("wall", 1),
    ("floor", 2),
    ("ceiling", 3),
    ("door", 4),
    ("window", 5),
    ("chair", 6),
    ("table", 7),
    ("sofa", 8),
    ("bed", 9),
    ("cabinet", 10),
    ("sink", 11),
    ("toilet", 12),
    ("bathtub", 13),
    ("stairs", 14),
    ("furniture", 15),
    ("appliances", 16),
    ("misc", 17),
])


def get_class_name_to_class_id(dataset: str = "hm3d") -> OrderedDict:
    """
    Get mapping from semantic class name to class ID.
    
    Args:
        dataset: Dataset name ("hm3d", "mp3d", "gibson")
        
    Returns:
        OrderedDict mapping class names to class IDs
    """
    dataset = dataset.lower()
    
    if dataset == "hm3d":
        base_mapping = HM3D_CLASS_NAME_TO_CLASS_ID.copy()
    elif dataset == "mp3d":
        base_mapping = MP3D_CLASS_NAME_TO_CLASS_ID.copy()
    elif dataset == "gibson":
        base_mapping = GIBSON_CLASS_NAME_TO_CLASS_ID.copy()
    else:
        log.warning(f"Unknown dataset: {dataset}, using HM3D mapping")
        base_mapping = HM3D_CLASS_NAME_TO_CLASS_ID.copy()
    
    # Add common iGibson-style aliases for compatibility
    starting_class_id = max(base_mapping.values()) + 1
    
    # Add floor alternatives
    if "floors" not in base_mapping:
        base_mapping["floors"] = base_mapping.get("floor", starting_class_id)
        
    if "walls" not in base_mapping:
        base_mapping["walls"] = base_mapping.get("wall", starting_class_id + 1)
        
    if "ceilings" not in base_mapping:
        base_mapping["ceilings"] = base_mapping.get("ceiling", starting_class_id + 2)
        
    if "carpet" not in base_mapping:
        base_mapping["carpet"] = base_mapping.get("floor", starting_class_id + 3)
    
    # Add replacement categories
    current_id = max(base_mapping.values()) + 1
    for old_category, new_category_map in REPLACED_CATEGORIES.items():
        if isinstance(new_category_map, str):
            if new_category_map not in base_mapping:
                base_mapping[new_category_map] = current_id
                current_id += 1
        elif isinstance(new_category_map, dict):
            for room_type, new_category in new_category_map.items():
                if new_category not in base_mapping:
                    base_mapping[new_category] = current_id
                    current_id += 1
    
    return base_mapping


# Default class mapping (using HM3D)
CLASS_NAME_TO_CLASS_ID = get_class_name_to_class_id("hm3d")

# Reverse mapping
CLASS_ID_TO_CLASS_NAME = {v: k for k, v in CLASS_NAME_TO_CLASS_ID.items()}


def get_class_id_to_class_name(dataset: str = "hm3d") -> Dict[int, str]:
    """
    Get mapping from class ID to class name.
    
    Args:
        dataset: Dataset name
        
    Returns:
        Dict mapping class IDs to class names
    """
    name_to_id = get_class_name_to_class_id(dataset)
    return {v: k for k, v in name_to_id.items()}


# ============================================================================
# Scene Patching Utilities (stubs for Habitat compatibility)
# ============================================================================

class HabitatScenePatcher:
    """
    Provides scene patching functionality for Habitat scenes.
    This is a compatibility layer - Habitat scenes typically don't need patching.
    """
    
    @staticmethod
    def apply_category_replacements(category: str, room_type: Optional[str] = None) -> str:
        """
        Apply category replacements for consistency.
        
        Args:
            category: Original category name
            room_type: Optional room type for context-dependent replacement
            
        Returns:
            Replaced category name
        """
        if category not in REPLACED_CATEGORIES:
            return category
            
        replacement = REPLACED_CATEGORIES[category]
        
        if isinstance(replacement, str):
            return replacement
        elif isinstance(replacement, dict) and room_type:
            return replacement.get(room_type, category)
        
        return category
    
    @staticmethod
    def get_semantic_class_id(category: str, dataset: str = "hm3d") -> int:
        """
        Get semantic class ID for a category.
        
        Args:
            category: Category name
            dataset: Dataset name
            
        Returns:
            Class ID (or -1 if not found)
        """
        mapping = get_class_name_to_class_id(dataset)
        return mapping.get(category, mapping.get("misc", -1))


# ============================================================================
# Compatibility with original iGibson-based code
# ============================================================================

# These are stubs to maintain API compatibility with the original code
# that expected iGibson's InteractiveIndoorScene patching

class MonkeyPatchedInteractiveIndoorScene:
    """
    Stub class for compatibility with iGibson-based code.
    In Habitat, scenes don't need monkey-patching.
    """
    
    @staticmethod
    def _add_object(obj):
        """Stub for object addition."""
        return obj
    
    @staticmethod
    def _orig_add_object(obj):
        """Stub for original object addition."""
        return obj

