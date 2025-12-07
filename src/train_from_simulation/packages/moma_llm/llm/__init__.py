# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.

# Try to import iGibson-compatible LLM (may not be available)
try:
    from .llm import LLM, LLM_hugging, Conversation, inflect_engine
    IGIBSON_LLM_AVAILABLE = True
except ImportError:
    IGIBSON_LLM_AVAILABLE = False

# Try to import Habitat-compatible LLM
try:
    from .habitat_llm import (
        LLM as HabitatLLM,
        LLM_hugging as HabitatLLM_hugging,
        Conversation as HabitatConversation,
        object_states as habitat_object_states,
        inflect_engine,
    )
    HABITAT_LLM_AVAILABLE = True
except ImportError:
    HABITAT_LLM_AVAILABLE = False

__all__ = [
    "IGIBSON_LLM_AVAILABLE",
    "HABITAT_LLM_AVAILABLE",
]

if IGIBSON_LLM_AVAILABLE:
    __all__.extend(["LLM", "LLM_hugging", "Conversation", "inflect_engine"])
    
if HABITAT_LLM_AVAILABLE:
    __all__.extend([
        "HabitatLLM", 
        "HabitatLLM_hugging", 
        "HabitatConversation",
        "habitat_object_states"
    ])
