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

# Import Multi-LLM environment
try:
    from .habitat_multi_llm_env import HabitatMultiLLMEnv
    MULTI_LLM_AVAILABLE = True
except ImportError:
    HabitatMultiLLMEnv = None
    MULTI_LLM_AVAILABLE = False

# Import prompt modules
from . import prompts_v3
from . import prompts_v4

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
    "MULTI_LLM_AVAILABLE",
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

if MULTI_LLM_AVAILABLE:
    __all__.extend([
        "HabitatMultiLLMEnv"
    ])

# Always export prompt modules
__all__.extend([
    "prompts_v3",
    "prompts_v4"
])
