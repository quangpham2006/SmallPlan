"""
Two-Planner Agent Module

Contains the dual-LLM agent system with:
1. Main Planner - Makes navigation decisions
2. Narrator - Generates narrative summaries of exploration

The narrator provides context to the main planner through storytelling.
"""

from .narrator import NarratorAgent, NarratorState
from .two_planner_agent import (
    TwoPlannerAgent,
    APIType,
    create_two_planner_openai_agent,
    create_two_planner_custom_agent,
)
from .prompts import (
    # Main planner prompts
    MAIN_PLANNER_SYSTEM_PROMPT,
    MAIN_PLANNER_USER_PROMPT,
    MAIN_PLANNER_USER_PROMPT_NO_STORY,
    
    # Narrator prompts
    NARRATOR_SYSTEM_PROMPT,
    NARRATOR_USER_PROMPT,
    NARRATOR_UPDATE_PROMPT,
    
    # Story formatting
    STORY_SECTION_HEADER,
    STORY_SECTION_TEMPLATE,
    STORY_EMPTY,
    
    # Helper functions
    format_action_history,
    get_decision_guidance,
    format_story_section,
    build_narrator_prompt,
    build_narrator_update_prompt,
    build_main_planner_prompt,
    generate_fallback_story,
)

__all__ = [
    # Agents
    "TwoPlannerAgent",
    "NarratorAgent",
    "NarratorState",
    "APIType",
    
    # Factory functions
    "create_two_planner_openai_agent",
    "create_two_planner_custom_agent",
    
    # Main planner prompts
    "MAIN_PLANNER_SYSTEM_PROMPT",
    "MAIN_PLANNER_USER_PROMPT",
    "MAIN_PLANNER_USER_PROMPT_NO_STORY",
    
    # Narrator prompts
    "NARRATOR_SYSTEM_PROMPT",
    "NARRATOR_USER_PROMPT",
    "NARRATOR_UPDATE_PROMPT",
    
    # Story formatting
    "STORY_SECTION_HEADER",
    "STORY_SECTION_TEMPLATE",
    "STORY_EMPTY",
    
    # Helper functions
    "format_action_history",
    "get_decision_guidance",
    "format_story_section",
    "build_narrator_prompt",
    "build_narrator_update_prompt",
    "build_main_planner_prompt",
    "generate_fallback_story",
]

