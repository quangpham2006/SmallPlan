# Try to import iGibson environment and components (may not be available)
try:
    from .env import OurIGibsonEnv, create_igibson_env
    from .llm_env import JsonLLMEnv, LLMEnv
    from .high_level_env import HighLevelEnv
    from .baselines import GreedyBaseline as IGibsonGreedyBaseline
    from .baselines import RandomBaseline as IGibsonRandomBaseline
    IGIBSON_AVAILABLE = True
except ImportError:
    OurIGibsonEnv = None
    create_igibson_env = None
    JsonLLMEnv = None
    LLMEnv = None
    HighLevelEnv = None
    IGibsonGreedyBaseline = None
    IGibsonRandomBaseline = None
    IGIBSON_AVAILABLE = False

# Try to import Habitat environment and components
try:
    from .habitat_env import OurHabitatEnv, create_habitat_env
    from .habitat_llm_env import HabitatHighLevelEnv, HabitatLLMEnv
    from .habitat_baselines import HabitatGreedyBaseline, HabitatRandomBaseline
    HABITAT_AVAILABLE = True
except ImportError:
    OurHabitatEnv = None
    create_habitat_env = None
    HabitatHighLevelEnv = None
    HabitatLLMEnv = None
    HabitatGreedyBaseline = None
    HabitatRandomBaseline = None
    HABITAT_AVAILABLE = False

# Provide default GreedyBaseline and RandomBaseline based on availability
if HABITAT_AVAILABLE:
    GreedyBaseline = HabitatGreedyBaseline
    RandomBaseline = HabitatRandomBaseline
elif IGIBSON_AVAILABLE:
    GreedyBaseline = IGibsonGreedyBaseline
    RandomBaseline = IGibsonRandomBaseline
else:
    GreedyBaseline = None
    RandomBaseline = None

__all__ = [
    "GreedyBaseline",
    "RandomBaseline",
    "IGIBSON_AVAILABLE",
    "HABITAT_AVAILABLE",
]

if IGIBSON_AVAILABLE:
    __all__.extend([
        "OurIGibsonEnv", 
        "create_igibson_env", 
        "JsonLLMEnv", 
        "LLMEnv", 
        "HighLevelEnv",
        "IGibsonGreedyBaseline",
        "IGibsonRandomBaseline"
    ])
    
if HABITAT_AVAILABLE:
    __all__.extend([
        "OurHabitatEnv", 
        "create_habitat_env", 
        "HabitatHighLevelEnv", 
        "HabitatLLMEnv",
        "HabitatGreedyBaseline",
        "HabitatRandomBaseline"
    ])
