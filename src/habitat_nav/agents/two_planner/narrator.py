"""
Narrator/Storyteller Agent for Two-Planner System

This module implements the Narrator LLM that generates narrative summaries
of the robot's exploration journey to help the main planner make better decisions.

The narrator:
1. Tracks the robot's journey through the environment
2. Summarizes actions, discoveries, and failures
3. Provides context to the main planner through story-like narratives
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .prompts import (
    NARRATOR_SYSTEM_PROMPT,
    build_narrator_prompt,
    build_narrator_update_prompt,
    generate_fallback_story,
    format_discovered_rooms_for_narrator,
)

logger = logging.getLogger(__name__)


@dataclass
class NarratorState:
    """State tracking for the narrator."""
    # Current story summary
    current_story: str = ""
    
    # Exploration tracking
    starting_room: str = ""
    current_room: str = ""
    
    # Action records with rich info for narration
    # Each record: (action, argument, success, feedback, room_before, room_after, discoveries)
    action_records: List[Tuple] = field(default_factory=list)
    
    # Discovered rooms -> objects
    discovered_rooms: Dict[str, List[str]] = field(default_factory=dict)
    
    # Story update counter (for controlling update frequency)
    story_update_counter: int = 0
    
    # Token tracking
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_queries: int = 0
    
    def reset(self):
        """Reset narrator state for new episode."""
        self.current_story = ""
        self.starting_room = ""
        self.current_room = ""
        self.action_records = []
        self.discovered_rooms = {}
        self.story_update_counter = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_queries = 0
    
    def add_action_record(
        self,
        action: str,
        argument: str,
        success: bool,
        feedback: str = "",
        room_before: str = "",
        room_after: str = "",
        discoveries: List[str] = None
    ):
        """Add an action record for narration."""
        self.action_records.append((
            action,
            argument,
            success,
            feedback,
            room_before,
            room_after or room_before,
            discoveries or []
        ))
        self.current_room = room_after or room_before or self.current_room
        self.story_update_counter += 1


class NarratorAgent:
    """
    Narrator/Storyteller LLM Agent.
    
    Generates narrative summaries of the robot's exploration journey
    to provide context for the main planning LLM.
    
    Features:
    - Initial story generation from full history
    - Incremental story updates after each action
    - Fallback story generation if LLM fails
    - Configurable update frequency
    
    Example:
        narrator = NarratorAgent(api_type="openai", model_name="gpt-4o-mini")
        narrator.reset(starting_room="living room")
        
        # After an action
        narrator.record_action("explore", "kitchen", True, "Found fridge and table")
        story = narrator.generate_story(task_description="find the TV")
    """
    
    def __init__(
        self,
        api_type: str = "openai",
        api_url: str = "http://localhost:8000/generate",
        model_name: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        update_frequency: int = 3,  # Generate new story every N actions
        name: str = "narrator"
    ):
        """
        Initialize Narrator agent.
        
        Args:
            api_type: "openai" or "custom"
            api_url: Custom API URL (if api_type="custom")
            model_name: Model name (e.g., "gpt-4o-mini" for cost-effective narration)
            openai_api_key: OpenAI API key (uses env var if None)
            max_tokens: Max tokens for story generation
            temperature: Sampling temperature
            update_frequency: Generate new story every N actions
            name: Agent name for logging
        """
        self.api_type = api_type.lower()
        self.api_url = api_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.update_frequency = update_frequency
        self.name = name
        
        # State
        self.state = NarratorState()
        
        # OpenAI client
        self._openai_client = None
        if self.api_type == "openai":
            self._init_openai(openai_api_key)
        
        # Last interaction tracking
        self.last_system_prompt: str = ""
        self.last_user_prompt: str = ""
        self.last_response: str = ""
        
        logger.info(f"Initialized NarratorAgent: api={api_type}, model={model_name}")
    
    def _init_openai(self, api_key: Optional[str] = None):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                    "or pass openai_api_key parameter."
                )
            
            self._openai_client = OpenAI(api_key=key)
            logger.info("OpenAI client initialized for narrator")
            
        except ImportError:
            raise ImportError(
                "OpenAI package required. Install with: pip install openai"
            )
    
    def reset(self, starting_room: str = ""):
        """Reset narrator for new episode."""
        self.state.reset()
        self.state.starting_room = starting_room
        self.state.current_room = starting_room
        
        self.last_system_prompt = ""
        self.last_user_prompt = ""
        self.last_response = ""
        
        logger.debug(f"Narrator reset, starting room: {starting_room}")
    
    def record_action(
        self,
        action: str,
        argument: str,
        success: bool,
        feedback: str = "",
        room_before: str = "",
        room_after: str = "",
        discoveries: List[str] = None
    ):
        """
        Record an action for narration.
        
        Args:
            action: Action name (e.g., "explore", "goto", "open")
            argument: Action argument (e.g., room name, object name)
            success: Whether action succeeded
            feedback: Feedback message
            room_before: Room before action
            room_after: Room after action
            discoveries: List of newly discovered objects/rooms
        """
        self.state.add_action_record(
            action=action,
            argument=argument,
            success=success,
            feedback=feedback,
            room_before=room_before,
            room_after=room_after,
            discoveries=discoveries
        )
    
    def update_discovered_rooms(self, room_dict: Dict[str, List[str]]):
        """Update the discovered rooms dictionary."""
        self.state.discovered_rooms = room_dict.copy()
    
    def generate_story(
        self,
        task_description: str,
        nearby_objects: List[str] = None,
        unexplored_areas: List[str] = None,
        force_update: bool = False
    ) -> str:
        """
        Generate or update the exploration story.
        
        Args:
            task_description: Current task description
            nearby_objects: Currently visible objects
            unexplored_areas: Areas that haven't been explored
            force_update: Force story regeneration even if not due
            
        Returns:
            Story narrative string
        """
        # Check if we should generate a new story
        should_update = (
            force_update or
            not self.state.current_story or
            self.state.story_update_counter >= self.update_frequency
        )
        
        if not should_update:
            return self.state.current_story
        
        # Generate story
        try:
            if self.state.current_story and self.state.action_records:
                # Incremental update
                story = self._generate_incremental_story(task_description)
            else:
                # Full story generation
                story = self._generate_full_story(
                    task_description,
                    nearby_objects or [],
                    unexplored_areas or []
                )
            
            self.state.current_story = story
            self.state.story_update_counter = 0
            return story
            
        except Exception as e:
            logger.warning(f"Story generation failed: {e}. Using fallback.")
            fallback = generate_fallback_story(
                self.state.starting_room,
                self.state.current_room,
                self.state.action_records
            )
            self.state.current_story = fallback
            return fallback
    
    def _generate_full_story(
        self,
        task_description: str,
        nearby_objects: List[str],
        unexplored_areas: List[str]
    ) -> str:
        """Generate story from full history."""
        system_prompt, user_prompt = build_narrator_prompt(
            task_description=task_description,
            starting_room=self.state.starting_room,
            current_room=self.state.current_room,
            discovered_rooms=self.state.discovered_rooms,
            action_history=self.state.action_records,
            nearby_objects=nearby_objects,
            unexplored_areas=unexplored_areas
        )
        
        return self._query_llm(system_prompt, user_prompt)
    
    def _generate_incremental_story(self, task_description: str) -> str:
        """Generate incremental story update."""
        if not self.state.action_records:
            return self.state.current_story
        
        # Get the last action
        last_action = self.state.action_records[-1]
        action, argument, success, feedback, room_before, room_after, discoveries = last_action
        
        new_action = f"{action}({argument})"
        action_result = "succeeded" if success else f"failed ({feedback})"
        new_obs = ", ".join(discoveries) if discoveries else "no new discoveries"
        
        system_prompt, user_prompt = build_narrator_update_prompt(
            previous_summary=self.state.current_story,
            new_action=new_action,
            action_result=action_result,
            new_observations=new_obs
        )
        
        return self._query_llm(system_prompt, user_prompt)
    
    def _query_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Query the LLM for story generation."""
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.state.total_queries += 1
        
        if self.api_type == "openai":
            return self._query_openai(system_prompt, user_prompt)
        else:
            return self._query_custom_api(system_prompt, user_prompt)
    
    def _query_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Query OpenAI API."""
        if self._openai_client is None:
            logger.error("OpenAI client not initialized for narrator")
            return generate_fallback_story(
                self.state.starting_room,
                self.state.current_room,
                self.state.action_records
            )
        
        try:
            response = self._openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            if response.usage:
                self.state.total_input_tokens += response.usage.prompt_tokens
                self.state.total_output_tokens += response.usage.completion_tokens
            
            result = response.choices[0].message.content
            self.last_response = result or ""
            
            logger.debug(f"Narrator story: {result[:100]}...")
            return result or generate_fallback_story(
                self.state.starting_room,
                self.state.current_room,
                self.state.action_records
            )
            
        except Exception as e:
            logger.error(f"Narrator OpenAI query failed: {e}")
            return generate_fallback_story(
                self.state.starting_room,
                self.state.current_room,
                self.state.action_records
            )
    
    def _query_custom_api(self, system_prompt: str, user_prompt: str) -> str:
        """Query custom API."""
        import requests
        
        try:
            prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"
            
            response = requests.post(
                self.api_url,
                json={
                    "prompt": prompt,
                    "model": self.model_name,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            text = result.get("response") or result.get("text") or result.get("content", "")
            self.last_response = text
            return text or generate_fallback_story(
                self.state.starting_room,
                self.state.current_room,
                self.state.action_records
            )
            
        except Exception as e:
            logger.error(f"Narrator custom API query failed: {e}")
            return generate_fallback_story(
                self.state.starting_room,
                self.state.current_room,
                self.state.action_records
            )
    
    def get_current_story(self) -> str:
        """Get current story without regenerating."""
        return self.state.current_story
    
    def get_last_interaction(self) -> Dict[str, Any]:
        """Get last LLM interaction for logging."""
        return {
            "system_prompt": self.last_system_prompt,
            "user_prompt": self.last_user_prompt,
            "response": self.last_response,
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get narrator metrics."""
        return {
            "name": self.name,
            "total_queries": self.state.total_queries,
            "total_input_tokens": self.state.total_input_tokens,
            "total_output_tokens": self.state.total_output_tokens,
            "total_tokens": self.state.total_input_tokens + self.state.total_output_tokens,
            "action_records_count": len(self.state.action_records),
            "current_story_length": len(self.state.current_story),
        }

