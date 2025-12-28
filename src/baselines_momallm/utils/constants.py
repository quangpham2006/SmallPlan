"""
Constants and Configuration for Habitat Navigation

Contains scene lists, node types, and other constants used throughout the module.
"""

from enum import IntEnum
from typing import List, Dict


class OCCUPANCY(IntEnum):
    """Occupancy map values."""
    FREE = 0
    OCCUPIED = 1
    UNEXPLORED = 2


class FRONTIER_CLASSIFICATION(IntEnum):
    """Frontier point classification."""
    WITHIN = 0       # Frontier within current room
    LEADING_OUT = 1  # Frontier leading to unexplored area


class NODETYPE:
    """Node type identifiers for scene graphs."""
    ROOM = 0
    OBJECT = 1
    FRONTIER = 2
    
    @staticmethod
    def roomname(room_id: int) -> str:
        """Generate room name from ID."""
        return f"room-{room_id}"
    
    @staticmethod
    def parse_room_id(name: str) -> int:
        """Parse room ID from room name."""
        if name.startswith("room-"):
            try:
                return int(name.split("-")[1])
            except (ValueError, IndexError):
                return -1
        return -1


# HM3D Scene Lists
# These are scene IDs from the HM3D dataset
HM3D_TRAINING_SCENES = [
    # "00800-TEEsavR23oF",  # Has semantic annotations
    # "00802-wcojb4TFT35",  # Has semantic annotations
    "00803-k1cupFYWXJ6",  # Has semantic annotations
    # "00808-y9hTuugGdiq",  # Has semantic annotations
    "00810-CrMo8WxCyVb",  # Has semantic annotations
    # "00813-svBbv1Pavdk",  # Has semantic annotations
    # "00814-p53SfW6mjZe",  # Has semantic annotations
    # "00815-h1zeeAwLh9Z",  # Has semantic annotations
    # "00820-mL8ThkuaVTM",  # Has semantic annotations
    "00821-eF36g7L6Z9M",  # Has semantic annotations
    # "00823-7MXmsvcQjpJ",  # Has semantic annotations
    # "00824-Dd4bFSTQ8gi",  # Has semantic annotations
    # "00827-BAbdmeyTvMZ",  # Has semantic annotations
    # "00829-QaLdnwvtxbs",  # Has semantic annotations
    # "00831-yr17PDCnDDW",  # Has semantic annotations
    # "00832-qyAac8rV8Zk",  # Has semantic annotations
    # "00835-q3zU7Yy5E5s",  # Has semantic annotations
    # "00839-zt1RVoi7PcG",  # Has semantic annotations
    # "00843-DYehNKdT76V",  # Has semantic annotations
    # "00844-q5QZSEeHe5g",  # Has semantic annotations
    # "00847-bCPU9suPUw9",  # Has semantic annotations
    # "00848-ziup5kvtCCR",  # Has semantic annotations
    # "00849-a8BtkwhxdRV",  # Has semantic annotations
    # "00853-5cdEh9F2hJL",  # Has semantic annotations
    # "00861-GLAQ4DNUx5U",  # Has semantic annotations
    # "00862-LT9Jq6dN3Ea",  # Has semantic annotations
    # "00869-MHPLjHsuG27",  # Has semantic annotations
    # "00871-VBzV5z6i1WS",  # Has semantic annotations
    # "00873-bxsVRursffK",  # Has semantic annotations
]

HM3D_TEST_SCENES = [
    "00876-mv2HUxq3B53",  # Has semantic annotations
    "00877-4ok3usBNeis",  # Has semantic annotations
    "00878-XB4GS9ShBRE",  # Has semantic annotations
    "00880-Nfvxx8J5NCo",  # Has semantic annotations
    "00890-6s7QHgap2fW",  # Has semantic annotations
    "00891-cvZr5TUy5C5",  # Has semantic annotations
    "00894-HY1NcmCgn3n",  # Has semantic annotations
]

# MP3D Scene Lists (for compatibility)
MP3D_TRAINING_SCENES = [
    "17DRP5sb8fy",
    "1LXtFkjw3qL",
    "1pXnuDYAj8r",
    "29hnd4uzFmX",
    "2azQ1b91cZZ",
]

MP3D_TEST_SCENES = [
    "5LpN3gDmAk7",
    "5q7pvUzZiYa",
    "759xd9YjKW5",
]

# Default scene lists (using HM3D)
TRAINING_SCENES = HM3D_TRAINING_SCENES
TEST_SCENES = HM3D_TEST_SCENES

# OVON Dataset Scenes (from Habitat-OVON dataset)
# Located in data/datasets/ovon/hm3d/{val_seen,val_unseen}/content/
OVON_VAL_SEEN_SCENES = [
    "4ok3usBNeis", "5cdEh9F2hJL", "6s7QHgap2fW", "7MXmsvcQjpJ",
    "a8BtkwhxdRV", "BAbdmeyTvMZ", "bCPU9suPUw9", "bxsVRursffK",
    "CrMo8WxCyVb", "cvZr5TUy5C5", "Dd4bFSTQ8gi", "DYehNKdT76V",
    "eF36g7L6Z9M", "GLAQ4DNUx5U", "h1zeeAwLh9Z", "HY1NcmCgn3n",
    "k1cupFYWXJ6", "LT9Jq6dN3Ea", "MHPLjHsuG27", "mL8ThkuaVTM",
    "mv2HUxq3B53", "Nfvxx8J5NCo", "p53SfW6mjZe", "q3zU7Yy5E5s",
    "q5QZSEeHe5g", "QaLdnwvtxbs", "qyAac8rV8Zk", "svBbv1Pavdk",
    "TEEsavR23oF", "VBzV5z6i1WS", "wcojb4TFT35", "XB4GS9ShBRE",
    "y9hTuugGdiq", "yr17PDCnDDW", "ziup5kvtCCR", "zt1RVoi7PcG",
]

OVON_VAL_UNSEEN_SCENES = [
    # These scenes are from the val_unseen split (val_unseen_hard.json.gz)
    # Same scene IDs as val_seen but with different (unseen) object categories
    "4ok3usBNeis", "5cdEh9F2hJL", "6s7QHgap2fW", "7MXmsvcQjpJ",
    "a8BtkwhxdRV", "BAbdmeyTvMZ", "bCPU9suPUw9", "bxsVRursffK",
    "CrMo8WxCyVb", "cvZr5TUy5C5", "Dd4bFSTQ8gi", "DYehNKdT76V",
    "eF36g7L6Z9M", "GLAQ4DNUx5U", "h1zeeAwLh9Z", "HY1NcmCgn3n",
    "k1cupFYWXJ6", "LT9Jq6dN3Ea", "MHPLjHsuG27", "mL8ThkuaVTM",
    "mv2HUxq3B53", "Nfvxx8J5NCo", "p53SfW6mjZe", "q3zU7Yy5E5s",
    "q5QZSEeHe5g", "QaLdnwvtxbs", "qyAac8rV8Zk", "svBbv1Pavdk",
    "TEEsavR23oF", "VBzV5z6i1WS", "wcojb4TFT35", "XB4GS9ShBRE",
    "y9hTuugGdiq", "yr17PDCnDDW", "ziup5kvtCCR", "zt1RVoi7PcG",
]


def get_scenes_for_dataset(dataset: str = "hm3d", split: str = "train") -> List[str]:
    """
    Get scene IDs for a specific dataset and split.
    
    Args:
        dataset: Dataset name ("hm3d", "mp3d", "ovon")
        split: Split name ("train", "test", "val", "val_seen", "val_unseen")
        
    Returns:
        List of scene IDs
    """
    dataset = dataset.lower()
    split = split.lower()
    
    scene_map = {
        "hm3d": {
            "train": HM3D_TRAINING_SCENES,
            "test": HM3D_TEST_SCENES,
            "val": HM3D_TEST_SCENES,  # Use test as val
        },
        "mp3d": {
            "train": MP3D_TRAINING_SCENES,
            "test": MP3D_TEST_SCENES,
            "val": MP3D_TEST_SCENES,
        },
        "ovon": {
            "val_seen": OVON_VAL_SEEN_SCENES,
            "val_unseen": OVON_VAL_UNSEEN_SCENES,
            "train": [],  # OVON doesn't use custom train scenes
            "val": OVON_VAL_SEEN_SCENES,  # Default val to val_seen
        }
    }
    
    if dataset not in scene_map:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(scene_map.keys())}")
    
    if split not in scene_map[dataset]:
        raise ValueError(f"Unknown split: {split}. Available: {list(scene_map[dataset].keys())}")
    
    return scene_map[dataset][split]


def get_ovon_episodes_path(split: str = "val_seen") -> str:
    """
    Get the path to OVON episodes directory.
    
    Args:
        split: Split name ("val_seen", "val_unseen")
        
    Returns:
        Path to the episodes content directory
    """
    return f"data/datasets/ovon/hm3d/{split}/content"


# Room type categories (for classification)
POSSIBLE_ROOMS = [
    "bathroom", "bedroom", "closet", "corridor", "dining room",
    "entryway", "garage", "hallway", "kitchen", "laundry room",
    "living room", "office", "other room", "outdoor", "stairs"
]


# Default configuration values
DEFAULT_CONFIG = {
    # Simulator settings
    "image_width": 256,
    "image_height": 256,
    "hfov": 90,
    "sensor_height": 1.5,
    
    # Agent settings
    "agent_height": 1.5,
    "agent_radius": 0.1,
    "forward_step_size": 0.25,  # Nav-R1 default
    "turn_angle": 30.0,         # Nav-R1 default (degrees)
    
    # Navigation settings
    "navigation_inflation_radius": 0.4,
    "success_distance_threshold": 1.5,
    
    # Episode settings
    "max_episode_steps": 500,
    "max_high_level_steps": 200,
    
    # SLAM settings
    "voxel_size": 0.075,
    "grid_size_meter": 30,
    "depth_min": 0.0,
    "depth_max": 5.0,
}


# Object categories to exclude from navigation targets
EXCLUDED_OBJECT_CATEGORIES = {
    # Structural
    "wall", "walls", "floor", "floors", "ceiling", "ceilings",
    "window", "windows", "door", "doors",
    
    # Abstract
    "void", "unknown", "misc", "unlabeled", "object", "objects",
    
    # Large fixtures
    "fireplace", "column", "pillar", "beam", "stairs",
    
    # Outdoor
    "roof", "ground", "terrain", "sky", "outdoor",
    
    # Agent
    "agent", "robot",
}

