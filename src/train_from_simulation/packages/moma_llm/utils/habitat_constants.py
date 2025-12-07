# Habitat-specific constants for SmallPlan
# Replaces iGibson-specific constants for scene IDs, etc.

from enum import IntEnum, Enum
from pathlib import Path
from typing import Dict, List, Set

# Project paths
PROJECT_DIR = Path(__file__).parent.parent.parent
PACKAGE_DIR = Path(__file__).parent.parent

# ============================================================================
# HM3D Scene IDs
# These are example scene IDs from Habitat-Matterport 3D dataset
# Adjust based on the scenes you have downloaded
# ============================================================================

# HM3D Minival scenes (downloaded to data/scene_datasets/hm3d/minival/)
HM3D_TRAINING_SCENES = [
    "00800-TEEsavR23oF",
    "00801-HaxA7YrQdEC", 
    "00802-wcojb4TFT35",
    "00803-k1cupFYWXJ6",
    "00804-BHXhpBwSMLh",
    "00805-SUHsP6z2gcJ",
    "00806-tQ5s4ShP627",
    "00807-rsggHU7g7dh",
]

HM3D_TEST_SCENES = [
    "00808-y9hTuugGdiq",
    "00809-Qpor2mEya8F",
]

# HM3D data path configuration
HM3D_DATA_PATH = "/media/khointn/SmallPlan/data"
HM3D_SPLIT = "minival"  # minival, train, or val

# ============================================================================
# MP3D Scene IDs (Matterport3D)
# Example scene IDs - adjust based on your dataset
# ============================================================================

MP3D_TRAINING_SCENES = [
    "17DRP5sb8fy",
    "1LXtFkjw3qL",
    "1pXnuDYAj8r",
    "29hnd4uzFmX",
    "2azQ1b91cZZ",
    "2n8kARJN3HM",
    "2t7WUuJeko7",
    "5LpN3gDmAk7",
]

MP3D_TEST_SCENES = [
    "5q7pvUzZiYa",
    "5ZKStnWn8Zo",
    "759xd9YjKW5",
    "7y3sRwLe3Va",
    "8194nk5LbLH",
]

# ============================================================================
# Gibson Scene IDs
# These map approximately to the original iGibson scenes used in SmallPlan
# Note: Some iGibson scenes may not have direct Gibson equivalents
# ============================================================================

GIBSON_TRAINING_SCENES = [
    "Beechwood",
    "Benevolence", 
    "Ihlen",
    "Merom",
    "Pomaria",
    "Rs",
    "Wainscott",
]

GIBSON_TEST_SCENES = [
    "Collierville",
    "Corozal",
    "Darden",
    "Markleeville",
    "Wiconisco",
]

# ============================================================================
# iGibson Scene IDs (from /media/datasets/ig_dataset/scenes)
# These are the original iGibson scenes that SmallPlan was designed for
# ============================================================================

IGIBSON_DATA_PATH = "/media/datasets/ig_dataset"

IGIBSON_TRAINING_SCENES = [
    "Merom_0_int",
    "Benevolence_0_int",
    "Pomaria_0_int",
    "Wainscott_1_int",
    "Rs_int",
    "Ihlen_0_int",
    "Beechwood_1_int",
    "Ihlen_1_int",
]

IGIBSON_TEST_SCENES = [
    "Benevolence_1_int",
    "Wainscott_0_int",
    "Pomaria_2_int",
    "Benevolence_2_int",
    "Beechwood_0_int",
    "Pomaria_1_int",
    "Merom_1_int",
]

# Default scene lists (using HM3D minival scenes)
TRAINING_SCENES = HM3D_TRAINING_SCENES
TEST_SCENES = HM3D_TEST_SCENES

# Default dataset type
DEFAULT_DATASET = "hm3d"  # Options: "igibson", "hm3d", "mp3d", "gibson"

# Maximum turn angle per step (radians)
MAX_TURN_ANGLE = 0.35

# Possible room categories for classification
POSSIBLE_ROOMS = [
    "kitchen", 
    "living room", 
    "combined kitchen and living room", 
    "bedroom", 
    "bathroom", 
    "hallway", 
    "office", 
    "dining room",
    "laundry room",
    "garage",
    "closet",
    "other room", 
    "unknown room"
]

# ============================================================================
# Semantic Class Mappings for Habitat
# These are common semantic classes across Habitat datasets
# ============================================================================

HABITAT_SEMANTIC_CLASSES = {
    "void": 0,
    "wall": 1,
    "floor": 2,
    "ceiling": 3,
    "door": 4,
    "window": 5,
    "chair": 6,
    "table": 7,
    "sofa": 8,
    "bed": 9,
    "cabinet": 10,
    "counter": 11,
    "sink": 12,
    "toilet": 13,
    "bathtub": 14,
    "shower": 15,
    "mirror": 16,
    "tv": 17,
    "plant": 18,
    "lamp": 19,
    "book": 20,
    "picture": 21,
    "refrigerator": 22,
    "stove": 23,
    "oven": 24,
    "microwave": 25,
    "dishwasher": 26,
    "washer": 27,
    "dryer": 28,
    "stairs": 29,
    "railing": 30,
    "fireplace": 31,
    "other": 255,
}

# Create reverse mapping
HABITAT_CLASS_ID_TO_NAME = {v: k for k, v in HABITAT_SEMANTIC_CLASSES.items()}

# ============================================================================
# Occupancy and Node Types (shared with original codebase)
# ============================================================================

class OCCUPANCY(IntEnum):
    """Occupancy states for map cells."""
    UNEXPLORED = 0
    FREE = 1
    OCCUPIED = 2

    @staticmethod
    def cmap():
        return {
            OCCUPANCY.UNEXPLORED: (0, 0, 0),
            OCCUPANCY.FREE: (0, 1, 0),
            OCCUPANCY.OCCUPIED: (0, 0, 1)
        }
        
    @staticmethod
    def to_rgb(arr):
        import numpy as np
        rgb = np.zeros((arr.shape[0], arr.shape[1], 3))
        for k, v in OCCUPANCY.cmap().items():
            rgb[arr == k] = v
        return rgb


class NODETYPE(IntEnum):
    """Node types for scene graph."""
    ROOT = 0
    FLOOR = 1
    ROOM = 2
    OBJECT = 3
    
    @staticmethod
    def roomname(room_id: int) -> str:
        return f"room-{room_id}"


class EDGETYPE(Enum):
    """Edge types for scene graph."""
    onTop = "onTop"
    inside = "inside"
    under = "under"
    inHand = "inHand"
    inRoom = "inRoom"
    roomConnected = "roomConnected"


class FRONTIER_CLASSIFICATION(IntEnum):
    """Frontier point classification."""
    WITHIN = 0
    LEADING_OUT = 1


# ============================================================================
# Helper functions for dataset selection
# ============================================================================

def get_scenes_for_dataset(dataset: str, split: str = "train") -> List[str]:
    """
    Get scene IDs for a given dataset and split.
    
    Args:
        dataset: Dataset name ("igibson", "hm3d", "mp3d", "gibson")
        split: Data split ("train" or "test")
        
    Returns:
        List of scene IDs
    """
    dataset = dataset.lower()
    split = split.lower()
    
    if dataset == "igibson":
        return IGIBSON_TRAINING_SCENES if split == "train" else IGIBSON_TEST_SCENES
    elif dataset == "hm3d":
        return HM3D_TRAINING_SCENES if split == "train" else HM3D_TEST_SCENES
    elif dataset == "mp3d":
        return MP3D_TRAINING_SCENES if split == "train" else MP3D_TEST_SCENES
    elif dataset == "gibson":
        return GIBSON_TRAINING_SCENES if split == "train" else GIBSON_TEST_SCENES
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_scene_path(scene_id: str, dataset: str, data_path: str = None) -> str:
    """
    Get full path to scene file.
    
    Args:
        scene_id: Scene identifier
        dataset: Dataset name
        data_path: Base data path (optional, uses defaults per dataset)
        
    Returns:
        Full path to scene file
    """
    import os
    
    dataset = dataset.lower()
    
    if dataset == "igibson":
        # iGibson scenes use OBJ mesh files
        base_path = data_path or IGIBSON_DATA_PATH
        # Return path to the mesh directory - the floor visual mesh
        mesh_path = os.path.join(base_path, "scenes", scene_id, "shape", "visual", "floor_0_vm.obj")
        if os.path.exists(mesh_path):
            return mesh_path
        # Fallback to scene directory
        return os.path.join(base_path, "scenes", scene_id)
    elif dataset == "hm3d":
        base_path = data_path or HM3D_DATA_PATH
        # Extract the scene name from scene_id (e.g., "00800-TEEsavR23oF" -> "TEEsavR23oF")
        scene_name = scene_id.split("-")[-1] if "-" in scene_id else scene_id
        
        # Try minival first, then train, then val
        for split in [HM3D_SPLIT, "minival", "train", "val"]:
            glb_path = os.path.join(base_path, "scene_datasets", "hm3d", split, scene_id, f"{scene_name}.basis.glb")
            if os.path.exists(glb_path):
                return glb_path
        
        # Fallback to old path format
        return os.path.join(base_path, "scene_datasets", "hm3d", scene_id, f"{scene_name}.basis.glb")
    elif dataset == "mp3d":
        base_path = data_path or "data"
        return os.path.join(base_path, "scene_datasets", "mp3d", scene_id, f"{scene_id}.glb")
    elif dataset == "gibson":
        base_path = data_path or "data"
        return os.path.join(base_path, "scene_datasets", "gibson", f"{scene_id}.glb")
    else:
        return scene_id


def get_igibson_scene_mesh_path(scene_id: str, mesh_type: str = "floor") -> str:
    """
    Get path to iGibson scene mesh file.
    
    Args:
        scene_id: Scene identifier (e.g., "Rs_int")
        mesh_type: Type of mesh ("floor", "wall", "ceiling")
        
    Returns:
        Path to OBJ mesh file
    """
    import os
    
    base_path = IGIBSON_DATA_PATH
    mesh_dir = os.path.join(base_path, "scenes", scene_id, "shape", "visual")
    
    if mesh_type == "floor":
        return os.path.join(mesh_dir, "floor_0_vm.obj")
    elif mesh_type == "wall":
        return os.path.join(mesh_dir, "wall_vm.obj")
    elif mesh_type == "ceiling":
        return os.path.join(mesh_dir, "ceiling_vm.obj")
    else:
        return mesh_dir


# Mapping of common iGibson class names to Habitat equivalents
IGIBSON_TO_HABITAT_CLASS = {
    "walls": "wall",
    "floors": "floor",
    "ceilings": "ceiling",
    "carpet": "floor",
    "window": "window",
    "door": "door",
    "chair": "chair",
    "table": "table",
    "sofa": "sofa",
    "bed": "bed",
    "bottom_cabinet": "cabinet",
    "top_cabinet": "cabinet",
    "sink": "sink",
    "toilet": "toilet",
    "bathtub": "bathtub",
    "shower": "shower",
    "mirror": "mirror",
    "tv_screen": "tv",
    "plant": "plant",
    "lamp": "lamp",
    "book": "book",
    "picture": "picture",
    "fridge": "refrigerator",
    "stove": "stove",
    "oven": "oven",
    "microwave": "microwave",
}


def igibson_class_to_habitat(igibson_class: str) -> str:
    """
    Convert iGibson class name to Habitat equivalent.
    
    Args:
        igibson_class: iGibson semantic class name
        
    Returns:
        Habitat semantic class name
    """
    return IGIBSON_TO_HABITAT_CLASS.get(igibson_class, igibson_class)

