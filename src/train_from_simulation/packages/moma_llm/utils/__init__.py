# Try to import constants (should work even without iGibson due to fallbacks)
try:
    from .constants import TEST_SCENES, TRAINING_SCENES
    IGIBSON_CONSTANTS_AVAILABLE = True
except ImportError:
    TEST_SCENES = []
    TRAINING_SCENES = []
    IGIBSON_CONSTANTS_AVAILABLE = False

# Try to import iGibson-specific utils (requires iGibson)
try:
    from .utils import get_config
    IGIBSON_UTILS_AVAILABLE = True
except ImportError:
    get_config = None
    IGIBSON_UTILS_AVAILABLE = False

# Try to import Habitat constants
try:
    from .habitat_constants import (
        HM3D_TRAINING_SCENES,
        HM3D_TEST_SCENES,
        MP3D_TRAINING_SCENES,
        MP3D_TEST_SCENES,
        GIBSON_TRAINING_SCENES,
        GIBSON_TEST_SCENES,
        get_scenes_for_dataset,
        get_scene_path,
        HABITAT_SEMANTIC_CLASSES,
        TRAINING_SCENES as HABITAT_TRAINING_SCENES,
        TEST_SCENES as HABITAT_TEST_SCENES,
    )
    HABITAT_CONSTANTS_AVAILABLE = True
    
    # Override default scenes with Habitat scenes if iGibson constants not available
    if not IGIBSON_CONSTANTS_AVAILABLE:
        TRAINING_SCENES = HABITAT_TRAINING_SCENES
        TEST_SCENES = HABITAT_TEST_SCENES
    
    # Provide a simple get_config for Habitat if iGibson utils not available
    if not IGIBSON_UTILS_AVAILABLE:
        from pathlib import Path
        def get_config(config_file: str) -> Path:
            """Get config file path for Habitat."""
            project_dir = Path(__file__).parent.parent.parent
            config_path = project_dir / "configs" / config_file
            if config_path.exists():
                return config_path
            return Path(config_file)
            
except ImportError:
    HABITAT_CONSTANTS_AVAILABLE = False

__all__ = [
    "TEST_SCENES",
    "TRAINING_SCENES",
    "get_config",
    "IGIBSON_CONSTANTS_AVAILABLE",
    "IGIBSON_UTILS_AVAILABLE",
    "HABITAT_CONSTANTS_AVAILABLE",
]

if HABITAT_CONSTANTS_AVAILABLE:
    __all__.extend([
        "HM3D_TRAINING_SCENES",
        "HM3D_TEST_SCENES", 
        "MP3D_TRAINING_SCENES",
        "MP3D_TEST_SCENES",
        "GIBSON_TRAINING_SCENES",
        "GIBSON_TEST_SCENES",
        "get_scenes_for_dataset",
        "get_scene_path",
        "HABITAT_SEMANTIC_CLASSES",
    ])
