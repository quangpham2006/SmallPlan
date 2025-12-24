"""
Image-Enhanced Planner Agent Module

Contains an LLM agent that uses visual context from images
to make navigation decisions. Uses GPT-4o Vision API to
understand the visual scene.

Features:
- Sends recent images to the LLM for visual understanding
- Collects images from different viewing angles
- Falls back to text-only if image sending fails
"""

from .image_planner_agent import (
    ImagePlannerAgent,
    APIType,
    ImageContext,
    create_image_planner_agent,
)
from .prompts import (
    IMAGE_PLANNER_SYSTEM_PROMPT,
    IMAGE_PLANNER_USER_PROMPT,
    format_action_history,
    get_decision_guidance,
    count_recent_failures,
    check_target_in_objects,
    format_discovered_rooms,
)

__all__ = [
    # Agent
    "ImagePlannerAgent",
    "APIType",
    "ImageContext",
    "create_image_planner_agent",
    
    # Prompts
    "IMAGE_PLANNER_SYSTEM_PROMPT",
    "IMAGE_PLANNER_USER_PROMPT",
    
    # Helper functions
    "format_action_history",
    "get_decision_guidance",
    "count_recent_failures",
    "check_target_in_objects",
    "format_discovered_rooms",
]

