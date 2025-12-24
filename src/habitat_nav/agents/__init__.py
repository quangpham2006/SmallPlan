"""
Navigation agents for Habitat ObjectNav.

This module provides three agent architectures:

1. Single Planner (default):
   - One LLM makes all navigation decisions
   - Simpler and lower cost
   - Import from: agents.single_planner or agents (default)

2. Two Planner:
   - Main Planner LLM + Narrator LLM
   - Narrator provides storytelling context to help main planner
   - Better for complex navigation tasks
   - Import from: agents.two_planner

3. Image Planner:
   - Single LLM with visual context from images
   - Sends recent images to GPT-4o Vision for understanding
   - Better visual scene understanding
   - Import from: agents.planner_with_image

Usage:
    # Single planner (default)
    from src.habitat_nav.agents import LLMAgent
    agent = LLMAgent(api_type="openai", model_name="gpt-4o")
    
    # Two planner
    from src.habitat_nav.agents import TwoPlannerAgent
    agent = TwoPlannerAgent(api_type="openai", main_model="gpt-4o", narrator_model="gpt-4o-mini")
    
    # Image planner
    from src.habitat_nav.agents import ImagePlannerAgent
    agent = ImagePlannerAgent(model_name="gpt-4o", max_images=3, action_level="high")
"""

# Base classes (shared)
from .base import BaseAgent, AgentState

# Random agent (shared)
from .random_agent import RandomAgent

# Single planner imports (default/backward compatible)
from .single_planner import (
    LLMAgent,
    APIType,
    create_openai_agent,
    create_custom_api_agent,
)

# Two planner imports (explicit import for dual-LLM system)
from .two_planner import (
    TwoPlannerAgent,
    NarratorAgent,
    create_two_planner_openai_agent,
    create_two_planner_custom_agent,
)

# Image planner imports (vision-based navigation)
from .planner_with_image import (
    ImagePlannerAgent,
    create_image_planner_agent,
)

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentState",
    
    # Random agent
    "RandomAgent",
    
    # Single planner (default)
    "LLMAgent",
    "APIType",
    "create_openai_agent",
    "create_custom_api_agent",
    
    # Two planner
    "TwoPlannerAgent",
    "NarratorAgent",
    "create_two_planner_openai_agent",
    "create_two_planner_custom_agent",
    
    # Image planner
    "ImagePlannerAgent",
    "create_image_planner_agent",
]
