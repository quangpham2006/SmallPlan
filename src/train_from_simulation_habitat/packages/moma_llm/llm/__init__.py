# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.

# Import Habitat-compatible LLM
try:
    from .habitat_llm import (
        LLM,
        LLM_hugging,
        Conversation,
        object_states,
        inflect_engine,
    )
    HABITAT_LLM_AVAILABLE = True
except ImportError:
    HABITAT_LLM_AVAILABLE = False
    LLM = None
    LLM_hugging = None
    Conversation = None
    object_states = None
    inflect_engine = None

# Import Storyteller LLM for multi-LLM system
try:
    from .storyteller_llm import StorytellerLLM, ActionRecord, StorytellerContext
    STORYTELLER_AVAILABLE = True
except ImportError:
    STORYTELLER_AVAILABLE = False
    StorytellerLLM = None
    ActionRecord = None
    StorytellerContext = None

__all__ = [
    "HABITAT_LLM_AVAILABLE",
    "STORYTELLER_AVAILABLE",
]

if HABITAT_LLM_AVAILABLE:
    __all__.extend([
        "LLM", 
        "LLM_hugging", 
        "Conversation",
        "object_states",
        "inflect_engine"
    ])

if STORYTELLER_AVAILABLE:
    __all__.extend([
        "StorytellerLLM",
        "ActionRecord",
        "StorytellerContext"
    ])
