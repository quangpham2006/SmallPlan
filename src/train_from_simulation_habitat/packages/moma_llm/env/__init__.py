# Import Habitat environment and components
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

# Provide default GreedyBaseline and RandomBaseline
if HABITAT_AVAILABLE:
    GreedyBaseline = HabitatGreedyBaseline
    RandomBaseline = HabitatRandomBaseline
else:
    GreedyBaseline = None
    RandomBaseline = None

__all__ = [
    "GreedyBaseline",
    "RandomBaseline",
    "HABITAT_AVAILABLE",
]

if HABITAT_AVAILABLE:
    __all__.extend([
        "OurHabitatEnv", 
        "create_habitat_env", 
        "HabitatHighLevelEnv", 
        "HabitatLLMEnv",
        "HabitatGreedyBaseline",
        "HabitatRandomBaseline"
    ])
