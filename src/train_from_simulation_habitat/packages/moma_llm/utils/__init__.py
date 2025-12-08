# Import Habitat constants
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
        TRAINING_SCENES,
        TEST_SCENES,
    )
    HABITAT_CONSTANTS_AVAILABLE = True
except ImportError:
    HABITAT_CONSTANTS_AVAILABLE = False
    TRAINING_SCENES = []
    TEST_SCENES = []
    HM3D_TRAINING_SCENES = []
    HM3D_TEST_SCENES = []
    MP3D_TRAINING_SCENES = []
    MP3D_TEST_SCENES = []
    GIBSON_TRAINING_SCENES = []
    GIBSON_TEST_SCENES = []
    get_scenes_for_dataset = None
    get_scene_path = None
    HABITAT_SEMANTIC_CLASSES = {}

# Import utils
try:
    from .utils import get_config
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    get_config = None

__all__ = [
    "TEST_SCENES",
    "TRAINING_SCENES",
    "get_config",
    "HABITAT_CONSTANTS_AVAILABLE",
    "UTILS_AVAILABLE",
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
