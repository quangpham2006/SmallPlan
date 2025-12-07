# Habitat Object Sampling - Stub Module
# Provides stubs for object sampling compatibility with Habitat-Lab/Habitat-Sim
# 
# Note: Habitat scenes are typically static and don't support dynamic object spawning
# like iGibson does. This module provides compatibility stubs.

import logging
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional, Set
from enum import Enum

import numpy as np

from moma_llm.llm.habitat_llm import object_states

log = logging.getLogger(__name__)


# ============================================================================
# Object Distribution Prior (for reference/future use)
# ============================================================================

class RelationType(Enum):
    """Object relation types."""
    ON_TOP = "ON_TOP"
    INSIDE = "INSIDE"
    UNDER = "UNDER"


# Default object placement prior (simplified)
# Maps: room_type -> furniture -> relation -> object -> probability
DEFAULT_OBJECT_DISTRIBUTION = {
    "kitchen": {
        "counter": {
            object_states.OnTop: {
                "apple": 0.3,
                "cup": 0.4,
                "plate": 0.5,
                "bowl": 0.4,
                "knife": 0.3,
            }
        },
        "cabinet": {
            object_states.Inside: {
                "cup": 0.3,
                "plate": 0.4,
                "bowl": 0.3,
            }
        },
        "refrigerator": {
            object_states.Inside: {
                "apple": 0.4,
                "bottle": 0.5,
            }
        }
    },
    "bedroom": {
        "bed": {
            object_states.OnTop: {
                "pillow": 0.8,
                "book": 0.2,
            }
        },
        "nightstand": {
            object_states.OnTop: {
                "lamp": 0.6,
                "book": 0.3,
                "phone": 0.4,
            }
        },
        "dresser": {
            object_states.OnTop: {
                "picture": 0.3,
                "lamp": 0.2,
            },
            object_states.Inside: {
                "clothes": 0.7,
            }
        }
    },
    "bathroom": {
        "sink": {
            object_states.OnTop: {
                "soap": 0.5,
                "toothbrush": 0.4,
            }
        },
        "bathtub": {
            object_states.Inside: {
                "bottle": 0.3,
            }
        }
    },
    "living_room": {
        "sofa": {
            object_states.OnTop: {
                "cushion": 0.6,
                "remote": 0.3,
            }
        },
        "table": {
            object_states.OnTop: {
                "book": 0.4,
                "cup": 0.3,
                "remote": 0.3,
            }
        }
    }
}


def load_object_distribution_prior() -> Dict:
    """
    Load object placement distribution prior.
    
    Returns:
        Nested dictionary of object placement probabilities
    """
    return DEFAULT_OBJECT_DISTRIBUTION


OBJECT_DISTRIBUTION_PRIOR = load_object_distribution_prior()


# ============================================================================
# Stub Functions for Habitat Compatibility
# ============================================================================

def create_new_object(env, scene, category: str, obj_number: int,
                      in_rooms: Optional[str] = None,
                      rendering_params: Optional[Dict] = None,
                      scale: Optional[float] = None,
                      bounding_box: Optional[np.ndarray] = None) -> Optional[Any]:
    """
    Create a new object in the scene.
    
    Note: This is a stub for Habitat compatibility.
    Habitat scenes are typically static and don't support dynamic object creation.
    
    Args:
        env: Environment instance
        scene: Scene instance
        category: Object category
        obj_number: Object number for naming
        in_rooms: Room(s) the object is in
        rendering_params: Rendering parameters
        scale: Object scale
        bounding_box: Object bounding box
        
    Returns:
        None (dynamic object creation not supported in Habitat)
    """
    log.warning(f"Dynamic object creation not supported in Habitat. "
                f"Requested: {category}_{obj_number}")
    return None


def match_furniture(given_furniture: List[str], 
                    spawnable_object_distribution: Dict) -> Dict:
    """
    Match furniture objects from prior to scene furniture.
    
    Note: Stub for Habitat compatibility.
    
    Args:
        given_furniture: List of furniture categories
        spawnable_object_distribution: Current distribution
        
    Returns:
        Updated distribution (unchanged in stub)
    """
    return spawnable_object_distribution


def match_rooms(given_room_types: List[str],
                spawnable_object_distribution: Dict) -> Dict:
    """
    Match room types from prior to scene rooms.
    
    Note: Stub for Habitat compatibility.
    
    Args:
        given_room_types: List of room types
        spawnable_object_distribution: Current distribution
        
    Returns:
        Updated distribution (unchanged in stub)
    """
    return spawnable_object_distribution


def match_distribution_against_objs() -> Dict:
    """
    Match object distribution against available objects.
    
    Note: Stub for Habitat compatibility.
    
    Returns:
        Empty distribution (dynamic spawning not supported)
    """
    return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))


def relation_to_str(relation) -> str:
    """
    Convert relation to string representation.
    
    Args:
        relation: Relation object/class
        
    Returns:
        String representation
    """
    return str(relation).split('.')[-1].split("'")[0]


def get_sbert_matching(categories: List[str],
                       possible_categories: List[str],
                       score_thresh: float = 0.0,
                       closest_k: int = 0) -> Dict[str, List[str]]:
    """
    Get semantic similarity matching between categories.
    
    Note: This is a simplified stub. For full functionality,
    use the SentenceBERT-based matching from the sbert module.
    
    Args:
        categories: Source categories
        possible_categories: Target categories to match against
        score_thresh: Similarity threshold
        closest_k: Number of closest matches to return
        
    Returns:
        Dictionary mapping source categories to matched categories
    """
    # Simple exact match fallback
    mapping = {c: [] for c in categories}
    
    for cat in categories:
        for possible in possible_categories:
            if cat.lower() == possible.lower():
                mapping[cat].append(possible)
            elif cat.lower() in possible.lower() or possible.lower() in cat.lower():
                mapping[cat].append(possible)
                
    return mapping


def is_receptacle(obj_category: str) -> bool:
    """
    Check if object category is a receptacle (can contain other objects).
    
    Args:
        obj_category: Object category name
        
    Returns:
        Whether the object is a receptacle
    """
    receptacles = {
        "cabinet", "drawer", "shelf", "shelving", "shelving_unit",
        "refrigerator", "fridge", "microwave", "oven", "dishwasher",
        "washer", "dryer", "closet", "wardrobe", "dresser",
        "chest_of_drawers", "nightstand", "box", "basket", "bin",
        "container", "bag", "backpack", "suitcase"
    }
    
    if obj_category.lower() in ["door", "window"]:
        return False
        
    return obj_category.lower() in receptacles


def add_objects_from_our_distribution(env) -> Tuple[List, Dict]:
    """
    Add objects to scene based on distribution prior.
    
    Note: This is a stub for Habitat compatibility.
    Habitat scenes are static and don't support dynamic object spawning.
    
    Args:
        env: Environment instance
        
    Returns:
        Tuple of (empty list, empty dict)
    """
    log.info("Dynamic object spawning not supported in Habitat. "
             "Using existing scene objects only.")
    return [], {}


# ============================================================================
# Object Query Utilities (work with existing scene objects)
# ============================================================================

def get_objects_by_category(scene, category: str) -> List[Any]:
    """
    Get all objects of a given category from the scene.
    
    Args:
        scene: Scene instance
        category: Object category
        
    Returns:
        List of objects
    """
    if hasattr(scene, 'objects_by_category'):
        return scene.objects_by_category.get(category, [])
    return []


def get_objects_in_room(scene, room_type: str) -> List[Any]:
    """
    Get all objects in a given room type.
    
    Args:
        scene: Scene instance
        room_type: Room type
        
    Returns:
        List of objects
    """
    objects = []
    
    if hasattr(scene, 'objects_by_name'):
        for obj in scene.objects_by_name.values():
            if hasattr(obj, 'in_rooms') and obj.in_rooms:
                for room in obj.in_rooms:
                    if room_type.lower() in room.lower():
                        objects.append(obj)
                        break
                        
    return objects


def get_receptacle_objects(scene) -> List[Any]:
    """
    Get all receptacle objects from the scene.
    
    Args:
        scene: Scene instance
        
    Returns:
        List of receptacle objects
    """
    receptacles = []
    
    if hasattr(scene, 'objects_by_category'):
        for category, objects in scene.objects_by_category.items():
            if is_receptacle(category):
                receptacles.extend(objects)
                
    return receptacles


def get_object_relations(scene, obj) -> Dict[str, List[Any]]:
    """
    Get relations for an object (what's on top, inside, etc.).
    
    Note: This is a simplified version. Full spatial relation
    computation would require physics simulation.
    
    Args:
        scene: Scene instance
        obj: Object to get relations for
        
    Returns:
        Dictionary of relation type to related objects
    """
    # Simplified stub - actual implementation would use spatial queries
    return {
        "on_top": [],
        "inside": [],
        "under": [],
        "next_to": []
    }

