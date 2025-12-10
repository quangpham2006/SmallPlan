"""
Storyteller LLM Module for SmallPlan Multi-LLM System.

This module provides a secondary LLM that reads the main planning LLM's context
and generates a narrative summary of the robot's exploration journey. The story
provides better temporal context and reasoning for the main LLM's next decision.

Example output:
"You started in what appeared to be a hallway. You explored it and found a door
leading to the kitchen. Upon entering the kitchen, you noticed a refrigerator
and some cabinets. You opened the refrigerator but didn't find the target item.
You then noticed another door that might lead to more rooms..."
"""

import os
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from openai import OpenAI, OpenAIError

from dotenv import load_dotenv

# Import prompts from both v3 and v4
from moma_llm.env import prompts_v3
from moma_llm.env import prompts_v4

# Default to v3 for backwards compatibility
STORYTELLER_SYSTEM_PROMPT = prompts_v3.STORYTELLER_SYSTEM_PROMPT
STORYTELLER_USER_PROMPT = prompts_v3.STORYTELLER_USER_PROMPT
STORYTELLER_UPDATE_PROMPT = prompts_v3.STORYTELLER_UPDATE_PROMPT
STORY_EMPTY = prompts_v3.STORY_EMPTY
format_discovered_rooms_for_storyteller = prompts_v3.format_discovered_rooms_for_storyteller
format_action_history_for_storyteller = prompts_v3.format_action_history_for_storyteller
format_story_section = prompts_v3.format_story_section
generate_fallback_story = prompts_v3.generate_fallback_story
build_storyteller_prompt = prompts_v3.build_storyteller_prompt
build_storyteller_update_prompt = prompts_v3.build_storyteller_update_prompt

load_dotenv()


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class StorytellerContext:
    """Context information for the storyteller LLM."""
    task_description: str = ""
    starting_room: str = ""
    current_room: str = ""
    discovered_rooms: Dict[str, List[str]] = field(default_factory=dict)  # room -> objects
    action_history: List[Dict[str, Any]] = field(default_factory=list)  # List of action records
    nearby_objects: List[str] = field(default_factory=list)
    unexplored_areas: List[str] = field(default_factory=list)
    previous_summary: str = ""


@dataclass  
class ActionRecord:
    """Record of an action for storytelling purposes."""
    action: str
    argument: str
    success: bool
    feedback: str = ""
    room_before: str = ""
    room_after: str = ""
    new_discoveries: List[str] = field(default_factory=list)  # New rooms/objects found
    
    def to_narrative(self) -> str:
        """Convert action record to narrative format."""
        status = "successfully" if self.success else "unsuccessfully"
        narrative = f"- {self.action}({self.argument}): {status}"
        
        if self.feedback and not self.success:
            narrative += f" ({self.feedback})"
        
        if self.room_before and self.room_after and self.room_before != self.room_after:
            narrative += f" [moved from {self.room_before} to {self.room_after}]"
            
        if self.new_discoveries:
            narrative += f" [discovered: {', '.join(self.new_discoveries)}]"
            
        return narrative


# ============================================================================
# Storyteller LLM Class
# ============================================================================

class StorytellerLLM:
    """
    Secondary LLM that generates narrative summaries of the robot's exploration.
    
    This LLM reads the context from the main planning LLM and generates a
    story-like summary that helps provide temporal context and reasoning.
    """
    
    def __init__(self,
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.3,
                 debug: bool = False,
                 prompt_version: int = 2) -> None:
        """
        Initialize the Storyteller LLM.
        
        Args:
            model: OpenAI model to use for storytelling
            temperature: Temperature for generation (slightly higher for creativity)
            debug: Whether to print debug information
            prompt_version: Prompt version to use (2 for v3 prompts, 4 for v4 prompts)
        """
        self.model = model
        self.temperature = temperature
        self.debug = debug
        self.prompt_version = prompt_version
        self.client = OpenAI()
        
        # Select prompts based on version
        if prompt_version == 4:
            self._prompts = prompts_v4
        else:
            self._prompts = prompts_v3
        
        # Track the current story
        self.current_summary: str = ""
        self.context: StorytellerContext = StorytellerContext()
        self.action_records: List[ActionRecord] = []
        
        # Token tracking
        self._episode_input_tokens = 0
        self._episode_output_tokens = 0
        self._episode_queries = 0
    
    def reset(self):
        """Reset storyteller state for a new episode."""
        self.current_summary = ""
        self.context = StorytellerContext()
        self.action_records = []
        self._episode_input_tokens = 0
        self._episode_output_tokens = 0
        self._episode_queries = 0
    
    def get_episode_metrics(self) -> Dict[str, Any]:
        """Get token usage metrics for storyteller in current episode."""
        return {
            'storyteller_input_tokens': self._episode_input_tokens,
            'storyteller_output_tokens': self._episode_output_tokens,
            'storyteller_total_tokens': self._episode_input_tokens + self._episode_output_tokens,
            'storyteller_queries': self._episode_queries,
        }
    
    def update_context(self,
                       task_description: str,
                       current_room: str,
                       discovered_rooms: Dict[str, List[str]],
                       nearby_objects: List[str],
                       unexplored_areas: List[str]):
        """
        Update the storyteller's context with current state.
        
        Args:
            task_description: The current task description
            current_room: Current room the robot is in
            discovered_rooms: Dict mapping room names to object lists
            nearby_objects: List of nearby object names
            unexplored_areas: List of rooms with unexplored areas
        """
        # Track starting room (first room we see)
        if not self.context.starting_room:
            self.context.starting_room = current_room
            
        self.context.task_description = task_description
        self.context.current_room = current_room
        self.context.discovered_rooms = discovered_rooms
        self.context.nearby_objects = nearby_objects
        self.context.unexplored_areas = unexplored_areas
    
    def record_action(self,
                      action: str,
                      argument: str,
                      success: bool,
                      feedback: str = "",
                      room_before: str = "",
                      room_after: str = "",
                      new_discoveries: List[str] = None):
        """
        Record an action taken by the robot.
        
        Args:
            action: Action name (goto, explore, open, stop)
            argument: Action argument
            success: Whether the action succeeded
            feedback: Feedback message if action failed
            room_before: Room before the action
            room_after: Room after the action
            new_discoveries: New rooms/objects discovered
        """
        record = ActionRecord(
            action=action,
            argument=argument,
            success=success,
            feedback=feedback,
            room_before=room_before,
            room_after=room_after,
            new_discoveries=new_discoveries or []
        )
        self.action_records.append(record)
    
    def _format_action_history(self) -> str:
        """Format action history for the prompt."""
        if not self.action_records:
            return "No actions taken yet."
        
        # Convert ActionRecord objects to dicts for the formatter
        records_as_dicts = []
        for record in self.action_records:
            records_as_dicts.append({
                'action': record.action,
                'argument': record.argument,
                'success': record.success,
                'feedback': record.feedback,
                'room_before': record.room_before,
                'room_after': record.room_after,
                'new_discoveries': record.new_discoveries,
            })
        
        return self._prompts.format_action_history_for_storyteller(records_as_dicts)
    
    def _format_discovered_rooms(self) -> str:
        """Format discovered rooms for the prompt."""
        return self._prompts.format_discovered_rooms_for_storyteller(self.context.discovered_rooms)
    
    def generate_summary(self, force_new: bool = False) -> str:
        """
        Generate a narrative summary of the exploration so far.
        
        Args:
            force_new: If True, generate a completely new summary instead of updating
            
        Returns:
            Narrative summary string
        """
        # If we have no actions, return empty or minimal summary
        if not self.action_records:
            self.current_summary = self._prompts.STORY_EMPTY
            return self.current_summary
        
        # Decide whether to generate new or update existing
        if not self.current_summary or force_new or len(self.action_records) <= 2:
            # Generate full summary
            return self._generate_full_summary()
        else:
            # Update existing summary with latest action
            return self._update_summary()
    
    def _generate_full_summary(self) -> str:
        """Generate a complete narrative summary."""
        # Build prompts using the helper from selected prompts version
        system_prompt, user_prompt = self._prompts.build_storyteller_prompt(
            task_description=self.context.task_description,
            starting_room=self.context.starting_room,
            current_room=self.context.current_room,
            discovered_rooms=self.context.discovered_rooms,
            action_history=self.action_records,
            nearby_objects=self.context.nearby_objects,
            unexplored_areas=self.context.unexplored_areas
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=300  # Keep summaries concise
            )
            
            self.current_summary = response.choices[0].message.content.strip()
            
            # Update token tracking
            self._episode_queries += 1
            self._episode_input_tokens += response.usage.prompt_tokens
            self._episode_output_tokens += response.usage.completion_tokens
            
            if self.debug:
                print(f"\n[Storyteller] Generated summary:\n{self.current_summary}\n")
                
        except OpenAIError as e:
            print(f"Storyteller error: {e}")
            self.current_summary = self._generate_fallback_summary()
        
        return self.current_summary
    
    def _update_summary(self) -> str:
        """Update the existing summary with the latest action."""
        if not self.action_records:
            return self.current_summary
        
        latest_action = self.action_records[-1]
        
        # Build update prompts using the helper from selected prompts version
        system_prompt, user_prompt = self._prompts.build_storyteller_update_prompt(
            previous_summary=self.current_summary,
            new_action=f"{latest_action.action}({latest_action.argument})",
            action_result="succeeded" if latest_action.success else f"failed ({latest_action.feedback})",
            new_observations=", ".join(latest_action.new_discoveries) if latest_action.new_discoveries else "no new observations"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=300
            )
            
            self.current_summary = response.choices[0].message.content.strip()
            
            # Update token tracking
            self._episode_queries += 1
            self._episode_input_tokens += response.usage.prompt_tokens
            self._episode_output_tokens += response.usage.completion_tokens
            
            if self.debug:
                print(f"\n[Storyteller] Updated summary:\n{self.current_summary}\n")
                
        except OpenAIError as e:
            print(f"Storyteller update error: {e}")
            # Keep existing summary on error
        
        return self.current_summary
    
    def _generate_fallback_summary(self) -> str:
        """Generate a simple fallback summary without LLM."""
        return self._prompts.generate_fallback_story(
            starting_room=self.context.starting_room,
            current_room=self.context.current_room,
            action_records=self.action_records
        )
    
    def get_summary_for_prompt(self) -> str:
        """
        Get the current summary formatted for inclusion in the main LLM's prompt.
        
        Returns:
            Formatted summary string ready to be inserted into the main prompt
        """
        if not self.current_summary:
            return ""
        
        return self._prompts.format_story_section(self.current_summary)

