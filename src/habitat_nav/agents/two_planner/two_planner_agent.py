"""
Two-Planner Agent for Habitat Navigation

This module implements a dual-LLM agent system with:
1. Main Planner LLM - Decides navigation actions
2. Narrator LLM - Generates narrative summaries of exploration

The narrator provides context to the main planner through storytelling,
helping it understand the journey so far and avoid repeating mistakes.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from ..base import BaseAgent, AgentState
from .narrator import NarratorAgent
from .prompts import (
    MAIN_PLANNER_SYSTEM_PROMPT,
    MAIN_PLANNER_USER_PROMPT,
    ROOM_CLASSIFICATION_SYSTEM_PROMPT,
    ROOM_CLASSIFICATION_USER_PROMPT,
    format_action_history,
    get_decision_guidance,
    format_retry_prompt,
    count_recent_failures,
    check_target_in_objects,
    format_discovered_rooms,
    format_story_section,
    build_main_planner_prompt,
)
from ...utils.actions import Action, ActionSpace, HighLevelAction, HighLevelActionSpace
from ...core.observations import ProcessedObservation

logger = logging.getLogger(__name__)


class APIType(Enum):
    """Supported API types for LLM queries."""
    CUSTOM = "custom"
    OPENAI = "openai"


@dataclass
class Conversation:
    """Conversation history with the LLM."""
    messages: List[Dict[str, str]] = field(default_factory=list)
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
    
    def get_prompt(self) -> str:
        parts = []
        for msg in self.messages:
            role = msg["role"].upper()
            parts.append(f"[{role}]\n{msg['content']}")
        return "\n\n".join(parts)
    
    def get_openai_messages(self) -> List[Dict[str, str]]:
        return self.messages.copy()
    
    def clear(self):
        self.messages = []


class TwoPlannerAgent(BaseAgent):
    """
    Two-Planner LLM Agent with Main Planner + Narrator.
    
    Uses two LLMs:
    1. Main Planner: Makes navigation decisions based on current state and story
    2. Narrator: Generates narrative summaries of the exploration journey
    
    The narrator's story is included in the main planner's prompt to provide
    context about what has been tried, discovered, and learned.
    
    Example:
        agent = TwoPlannerAgent(
            api_type="openai",
            main_model="gpt-4o",
            narrator_model="gpt-4o-mini",
            action_level="high"
        )
    """
    
    def __init__(
        self,
        api_type: str = "openai",
        api_url: str = "http://localhost:8000/generate",
        main_model: str = "gpt-4o",
        narrator_model: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        narrator_temperature: float = 0.7,
        max_retries: int = 3,
        action_level: str = "high",
        narrator_update_frequency: int = 3,
        include_story: bool = True,
        name: str = "two_planner_agent"
    ):
        """
        Initialize Two-Planner agent.
        
        Args:
            api_type: Type of API - "custom" or "openai"
            api_url: URL for custom LLM API
            main_model: Model for main planner (e.g., "gpt-4o")
            narrator_model: Model for narrator (e.g., "gpt-4o-mini" for cost efficiency)
            openai_api_key: OpenAI API key
            max_tokens: Maximum tokens for main planner response
            temperature: Main planner sampling temperature
            narrator_temperature: Narrator sampling temperature
            max_retries: Maximum retries on parse failure
            action_level: "low" or "high" level actions
            narrator_update_frequency: Generate new story every N actions
            include_story: Whether to include narrator's story in prompts
            name: Agent name
        """
        super().__init__(name=name)
        
        # API configuration
        self.api_type = APIType(api_type.lower())
        self.api_url = api_url
        self.main_model = main_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Action level
        self.action_level = action_level.lower()
        if self.action_level not in ("low", "high"):
            raise ValueError(f"action_level must be 'low' or 'high', got '{action_level}'")
        
        # Story integration
        self.include_story = include_story
        
        # OpenAI client
        self._openai_client = None
        if self.api_type == APIType.OPENAI:
            self._init_openai(openai_api_key)
        
        # Initialize Narrator
        self.narrator = NarratorAgent(
            api_type=api_type,
            api_url=api_url,
            model_name=narrator_model,
            openai_api_key=openai_api_key,
            max_tokens=256,
            temperature=narrator_temperature,
            update_frequency=narrator_update_frequency,
            name="narrator"
        )
        
        # Conversation tracking
        self.conversation = Conversation()
        
        # Scene understanding (same as single planner)
        self.room_objects: Dict[str, set] = {}
        self.object_id_to_name: Dict[int, str] = {}
        self.room_classification: Dict[str, str] = {}
        self.room_display_names: Dict[str, str] = {}
        self.visited_rooms: set = set()
        self.room_exploration_status: Dict[str, bool] = {}
        self.room_visited_positions: Dict[str, set] = {}
        self.frontier_info: List[str] = []
        
        # Room classification settings
        self._pending_room_classification: bool = False
        self._last_classified_room_count: int = 0
        self._classification_name_counts: Dict[str, int] = {}
        
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_queries = 0
        
        # Last interaction tracking
        self.last_system_prompt: str = ""
        self.last_user_prompt: str = ""
        self.last_response: str = ""
        self.last_action: Any = None
        self.last_action_feedback: str = ""
        self.last_story: str = ""
        
        # Action history: (action_name, argument, success, feedback)
        self.action_history: List[Tuple[str, str, bool, str]] = []
        
        # Track starting room
        self.starting_room: str = ""
        
        logger.info(f"Initialized TwoPlannerAgent: api={self.api_type.value}, "
                   f"main_model={main_model}, narrator_model={narrator_model}, "
                   f"action_level={action_level}")
    
    def _init_openai(self, api_key: Optional[str] = None):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY environment variable."
                )
            
            self._openai_client = OpenAI(api_key=key)
            logger.info("OpenAI client initialized for main planner")
            
        except ImportError:
            raise ImportError("OpenAI package required. Install with: pip install openai")
    
    def act(
        self,
        observation: ProcessedObservation,
        task_description: str,
        info: Dict[str, Any]
    ):
        """
        Query both LLMs and return action.
        
        Flow:
        1. Update scene understanding
        2. Generate/update narrator story
        3. Build main planner prompt with story
        4. Query main planner for action
        5. Parse and return action
        """
        # Update scene understanding
        self._update_scene_understanding(observation, info)
        
        # Generate story from narrator
        story = ""
        if self.include_story:
            story = self._generate_story(task_description, observation)
            self.last_story = story
        
        # Build prompt with story
        system_prompt, user_prompt = self._build_prompts(
            task_description, info, observation, story
        )
        
        # Query main planner
        response = self._query_llm(system_prompt, user_prompt)
        
        # Parse action
        action = self._parse_action(response)
        
        # Log interaction
        self._log_interaction(system_prompt, user_prompt, response, action)
        
        # Record action
        if action:
            action_name = action.name if hasattr(action, 'name') else str(action)
            action_arg = action.argument if hasattr(action, 'argument') else ""
            self.state.add_action(action_name, action_arg, True)
        
        # Return default if parsing failed
        if action is None:
            if self.action_level == "low":
                return Action.STOP
            else:
                return HighLevelAction("stop", "", "Parse failed")
        
        return action
    
    def reset(self, target_category: str = ""):
        """Reset agent for new episode."""
        self.state.reset(target_category)
        self.conversation.clear()
        
        # Reset scene understanding
        self.room_objects = {}
        self.object_id_to_name = {}
        self.room_classification = {}
        self.room_display_names = {}
        self.visited_rooms = set()
        self.room_exploration_status = {}
        self.room_visited_positions = {}
        self.frontier_info = []
        self._pending_room_classification = False
        self._last_classified_room_count = 0
        self._classification_name_counts = {}
        
        # Reset action history
        self.action_history = []
        
        # Reset tracking
        self.last_system_prompt = ""
        self.last_user_prompt = ""
        self.last_response = ""
        self.last_action = None
        self.last_action_feedback = ""
        self.last_story = ""
        self.starting_room = ""
        
        # Reset narrator
        self.narrator.reset(starting_room="")
        
        logger.debug(f"TwoPlannerAgent reset for target: {target_category}")
    
    def update(
        self,
        action,
        observation,
        info: Dict[str, Any]
    ):
        """Update agent after taking an action."""
        self.state.add_observation(observation)
        
        # Determine action success
        action_success = True
        feedback_reason = ""
        
        if "action_result" in info:
            result = info["action_result"]
            if hasattr(result, 'success'):
                action_success = result.success
                if not action_success:
                    feedback_reason = getattr(result, 'feedback', '') or "action failed"
        
        # Get current room for narrator
        current_room = info.get("current_room", "")
        room_before = self.narrator.state.current_room
        
        # Store feedback
        self.last_action_feedback = feedback_reason if not action_success else ""
        
        # Record in action history
        if self.last_action is not None:
            action_name = self.last_action.name if hasattr(self.last_action, 'name') else str(self.last_action)
            action_arg = getattr(self.last_action, 'argument', '') if hasattr(self.last_action, 'argument') else ""
            
            self.action_history.append((
                action_name,
                action_arg,
                action_success,
                feedback_reason if not action_success else ""
            ))
            
            # Record in narrator with rich info
            discoveries = []
            if observation.visible_objects:
                # Find newly discovered objects
                for obj in observation.visible_objects:
                    if obj not in self.state.seen_objects:
                        discoveries.append(obj)
            
            self.narrator.record_action(
                action=action_name,
                argument=action_arg,
                success=action_success,
                feedback=feedback_reason,
                room_before=room_before,
                room_after=current_room,
                discoveries=discoveries
            )
            
            logger.info(f"ACTION RESULT: {action_name}({action_arg}) -> "
                       f"{'SUCCESS' if action_success else 'FAILED: ' + feedback_reason}")
    
    def _generate_story(
        self,
        task_description: str,
        observation: ProcessedObservation
    ) -> str:
        """Generate story from narrator."""
        # Get visible objects and unexplored areas
        nearby_objects = list(observation.visible_objects) if observation.visible_objects else []
        unexplored_areas = self.frontier_info[:5] if self.frontier_info else []
        
        # Update narrator with discovered rooms
        rooms_for_narrator = {}
        for room_id, obj_ids in self.room_objects.items():
            display_name = self._get_room_display_name(room_id)
            obj_names = list(set(self._get_room_objects_as_names(room_id)))
            rooms_for_narrator[display_name] = obj_names
        
        self.narrator.update_discovered_rooms(rooms_for_narrator)
        
        # Generate story
        story = self.narrator.generate_story(
            task_description=task_description,
            nearby_objects=nearby_objects,
            unexplored_areas=unexplored_areas
        )
        
        return story
    
    def _update_scene_understanding(
        self,
        observation: ProcessedObservation,
        info: Dict[str, Any]
    ):
        """Update internal scene model from observation."""
        current_room = info.get("current_room", "unknown")
        
        # Track starting room
        if not self.starting_room:
            self.starting_room = current_room
            self.narrator.state.starting_room = current_room
            self.narrator.state.current_room = current_room
        
        self.visited_rooms.add(current_room)
        
        # Track position for exploration coverage
        if current_room not in self.room_visited_positions:
            self.room_visited_positions[current_room] = set()
        
        pos = observation.position
        pos_key = (round(pos[0]), round(pos[2]))
        self.room_visited_positions[current_room].add(pos_key)
        
        # Collect visible objects
        new_objects_found = False
        if observation.visible_object_info:
            for obj_info in observation.visible_object_info:
                obj_id = obj_info.obj_id
                
                if obj_id in self.object_id_to_name:
                    continue
                
                self.object_id_to_name[obj_id] = obj_info.category
                new_objects_found = True
                
                if current_room not in self.room_objects:
                    self.room_objects[current_room] = set()
                    self._pending_room_classification = True
                    self.room_display_names[current_room] = current_room
                
                self.room_objects[current_room].add(obj_id)
        
        self.state.add_observation(observation)
        
        # Update frontier info
        if "frontier_info" in info:
            self.frontier_info = info["frontier_info"]
            
            if current_room in self.room_objects:
                positions_visited = len(self.room_visited_positions.get(current_room, set()))
                no_frontiers = not self.frontier_info or len(self.frontier_info) == 0
                
                if no_frontiers and positions_visited >= 3:
                    if not self.room_exploration_status.get(current_room, False):
                        self.room_exploration_status[current_room] = True
                else:
                    self.room_exploration_status[current_room] = False
        
        # Room classification
        if new_objects_found or self._pending_room_classification:
            self._maybe_classify_rooms()
    
    def _maybe_classify_rooms(self):
        """Classify rooms if new rooms discovered."""
        current_room_count = len(self.room_objects)
        
        if not self._pending_room_classification and current_room_count == self._last_classified_room_count:
            return
        
        rooms_to_classify = {}
        for room_id, obj_ids in self.room_objects.items():
            if obj_ids:
                obj_names = [self.object_id_to_name.get(oid, "unknown") for oid in obj_ids]
                rooms_to_classify[room_id] = obj_names
        
        if not rooms_to_classify:
            return
        
        try:
            raw_classification = self.classify_rooms(rooms_to_classify, open_set=True)
            
            name_counts = {}
            for room_id in self.room_objects:
                if room_id in self.room_classification:
                    base_name = self.room_classification[room_id]
                    name_counts[base_name] = name_counts.get(base_name, 0) + 1
            
            for room_id, room_type in raw_classification.items():
                if room_id not in self.room_classification or room_type != "other room":
                    self.room_classification[room_id] = room_type
                    
                    base_name = room_type
                    count = name_counts.get(base_name, 0) + 1
                    name_counts[base_name] = count
                    
                    if count == 1:
                        display_name = base_name
                    else:
                        display_name = f"{base_name} {count}"
                    
                    self.room_display_names[room_id] = display_name
            
            self._update_display_names()
            self._pending_room_classification = False
            self._last_classified_room_count = current_room_count
            
        except Exception as e:
            logger.warning(f"Room classification failed: {e}")
    
    def _update_display_names(self):
        """Update display names for unique numbering."""
        rooms_by_type: Dict[str, List[str]] = {}
        for room_id, room_type in self.room_classification.items():
            if room_type not in rooms_by_type:
                rooms_by_type[room_type] = []
            rooms_by_type[room_type].append(room_id)
        
        for room_type, room_ids in rooms_by_type.items():
            if len(room_ids) == 1:
                self.room_display_names[room_ids[0]] = room_type
            else:
                for i, room_id in enumerate(sorted(room_ids)):
                    if i == 0:
                        self.room_display_names[room_id] = room_type
                    else:
                        self.room_display_names[room_id] = f"{room_type} {i + 1}"
    
    def _get_room_display_name(self, room_id: str) -> str:
        """Get human-readable display name for a room."""
        return self.room_display_names.get(room_id, room_id)
    
    def _get_room_objects_as_names(self, room_id: str) -> List[str]:
        """Get list of object category names for a room."""
        obj_ids = self.room_objects.get(room_id, set())
        return [self.object_id_to_name.get(oid, "unknown") for oid in obj_ids]
    
    def _build_prompts(
        self,
        task_description: str,
        info: Dict[str, Any],
        observation: Optional[ProcessedObservation] = None,
        story: str = ""
    ) -> Tuple[str, str]:
        """Build system and user prompts with story integration."""
        # Build room info
        rooms_for_prompt = {}
        room_exploration_for_prompt = {}
        
        for room_id, obj_ids in self.room_objects.items():
            if not obj_ids:
                continue
            
            display_name = self._get_room_display_name(room_id)
            obj_names = list(set(self._get_room_objects_as_names(room_id)))
            rooms_for_prompt[display_name] = obj_names
            room_exploration_for_prompt[display_name] = self.room_exploration_status.get(room_id, False)
        
        discovered_rooms = format_discovered_rooms(rooms_for_prompt, room_exploration_for_prompt)
        
        # Format nearby objects
        if observation and hasattr(observation, 'format_nearby_objects') and observation.visible_object_info:
            nearby = observation.format_nearby_objects(max_objects=15)
        elif self.state.seen_objects:
            nearby = ", ".join(sorted(self.state.seen_objects)[:15])
        else:
            nearby = "none visible"
        
        # Format action history
        action_history_formatted = format_action_history(self.action_history, max_items=5)
        
        # Unexplored info
        if self.frontier_info:
            unexplored = ", ".join(self.frontier_info[:5])
        else:
            unexplored = "none - all nearby directions explored"
        
        # Build exploration status
        exploration_status_parts = []
        rooms_with_unexplored = set()
        fully_explored_set = set()
        
        for room_id, obj_ids in self.room_objects.items():
            if not obj_ids:
                continue
            
            display_name = self._get_room_display_name(room_id)
            is_explored = self.room_exploration_status.get(room_id, False)
            
            if is_explored:
                fully_explored_set.add(display_name)
            else:
                rooms_with_unexplored.add(display_name)
        
        if rooms_with_unexplored:
            exploration_status_parts.append(f"Rooms with unexplored areas: [{', '.join(sorted(rooms_with_unexplored))}]")
        if fully_explored_set:
            exploration_status_parts.append(f"Fully explored rooms: [{', '.join(sorted(fully_explored_set))}]")
        
        if not self.frontier_info:
            unexplored_room_suggestions = set()
            for room_id, obj_ids in self.room_objects.items():
                if obj_ids and not self.room_exploration_status.get(room_id, False):
                    display_name = self._get_room_display_name(room_id)
                    unexplored_room_suggestions.add(display_name)
            
            if unexplored_room_suggestions:
                suggestions = ", ".join(sorted(unexplored_room_suggestions)[:3])
                exploration_status_parts.append(f"⚠️ Current area fully explored. Consider exploring: {suggestions}")
            elif self.room_objects:
                exploration_status_parts.append("⚠️ All discovered rooms explored. Try opening doors to discover new rooms.")
            else:
                exploration_status_parts.append("⚠️ No rooms discovered yet. Use explore() to discover rooms.")
        
        exploration_status = "\n".join(exploration_status_parts) if exploration_status_parts else ""
        
        # Get current room
        raw_current_room = info.get("current_room", "unknown")
        current_room = self._get_room_display_name(raw_current_room)
        
        # Generate decision guidance
        visible_objects = list(observation.visible_objects) if observation and observation.visible_objects else []
        target_found = check_target_in_objects(self.state.target_category, visible_objects)
        recent_failures = count_recent_failures(self.action_history)
        has_unexplored = bool(self.frontier_info) or any(not v for v in self.room_exploration_status.values())
        all_explored = len(self.room_exploration_status) > 0 and all(self.room_exploration_status.values())
        
        decision_guidance = get_decision_guidance(
            target_found=target_found,
            target_name=self.state.target_category,
            recent_failure_count=recent_failures,
            has_unexplored_areas=has_unexplored,
            all_rooms_explored=all_explored
        )
        
        # Get action descriptions
        if self.action_level == "high":
            action_descriptions = HighLevelActionSpace.get_action_descriptions()
        else:
            action_descriptions = ActionSpace.get_action_descriptions()
        
        # Build prompts using the builder function
        last_feedback = self.last_action_feedback if self.last_action_feedback else "None"
        
        system_prompt, user_prompt = build_main_planner_prompt(
            task_description=task_description,
            action_descriptions=action_descriptions,
            current_room=current_room,
            nearby_objects=nearby,
            story_content=story,
            discovered_rooms=discovered_rooms,
            action_history=action_history_formatted,
            unexplored_info=unexplored,
            exploration_status=exploration_status,
            decision_guidance=decision_guidance,
            target_object=self.state.target_category,
            last_feedback=last_feedback,
            include_story=self.include_story
        )
        
        self.conversation.add_message("system", system_prompt)
        self.conversation.add_message("user", user_prompt)
        
        return system_prompt, user_prompt
    
    def _query_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Query the main planner LLM."""
        self.total_queries += 1
        
        if self.api_type == APIType.OPENAI:
            return self._query_openai(system_prompt, user_prompt)
        else:
            return self._query_custom_api(system_prompt, user_prompt)
    
    def _get_fallback_response(self) -> str:
        """Get fallback response based on action level."""
        if self.action_level == "low":
            return "Action: TURN_LEFT"
        else:
            return "Command: explore()"
    
    def _query_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Query OpenAI API."""
        if self._openai_client is None:
            logger.error("OpenAI client not initialized")
            return self._get_fallback_response()
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self._openai_client.chat.completions.create(
                model=self.main_model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            if response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
            
            result = response.choices[0].message.content
            logger.debug(f"Main planner response: {result[:100]}...")
            return result or self._get_fallback_response()
            
        except Exception as e:
            logger.error(f"Main planner OpenAI query failed: {e}")
            return self._get_fallback_response()
    
    def _query_custom_api(self, system_prompt: str, user_prompt: str) -> str:
        """Query custom LLM API."""
        try:
            prompt = self.conversation.get_prompt()
            
            response = requests.post(
                self.api_url,
                json={
                    "prompt": prompt,
                    "model": self.main_model,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            text = result.get("response") or result.get("text") or result.get("content", "")
            return text or self._get_fallback_response()
            
        except Exception as e:
            logger.error(f"Main planner custom API query failed: {e}")
            return self._get_fallback_response()
    
    def _parse_action(self, response: str):
        """Parse action from LLM response."""
        if self.action_level == "low":
            return self._parse_low_level_action(response)
        else:
            return self._parse_high_level_action(response)
    
    def _parse_low_level_action(self, response: str) -> Optional[Action]:
        """Parse low-level action from response."""
        action_str = None
        for line in response.split("\n"):
            line_clean = line.strip()
            if line_clean.lower().startswith("action:"):
                action_str = line.split(":", 1)[1].strip()
                break
        
        if not action_str:
            action_names = ["MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]
            for name in action_names:
                if name in response.upper():
                    action_str = name
                    break
        
        if action_str:
            action = ActionSpace.parse_action(action_str)
            if action is not None:
                self.conversation.add_message("assistant", response)
                return action
        
        logger.warning(f"Failed to parse low-level action from: {response[:100]}...")
        return None
    
    def _parse_high_level_action(self, response: str) -> Optional[HighLevelAction]:
        """Parse high-level action from response."""
        command_match = None
        for line in response.split("\n"):
            line_clean = line.strip().lower()
            if line_clean.startswith("command:"):
                command_text = line.split(":", 1)[1].strip()
                command_match = command_text
                break
        
        if not command_match:
            match = re.search(r'(\w+)\(([^)]*)\)', response)
            if match:
                command_match = match.group(0)
        
        if command_match:
            action = HighLevelActionSpace.parse(command_match)
            if action:
                self.conversation.add_message("assistant", response)
                return action
        
        logger.warning(f"Failed to parse high-level action from: {response[:100]}...")
        return None
    
    def _log_interaction(
        self,
        system_prompt: str,
        user_prompt: str,
        response: str,
        action: Optional[Action]
    ):
        """Store and log LLM interaction."""
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.last_response = response
        self.last_action = action
        
        logger.info("=" * 60)
        logger.info("TWO-PLANNER INTERACTION")
        logger.info("=" * 60)
        
        # Log story
        if self.last_story:
            logger.info("--- NARRATOR STORY ---")
            logger.info(self.last_story)
        
        logger.info("--- SYSTEM PROMPT ---")
        system_lines = system_prompt.strip().split('\n')
        for line in system_lines[:10]:
            logger.info(line)
        if len(system_lines) > 10:
            logger.info(f"... ({len(system_lines) - 10} more lines)")
        
        logger.info("--- USER PROMPT ---")
        logger.info(user_prompt)
        
        logger.info("--- MAIN PLANNER RESPONSE ---")
        logger.info(response)
        
        if action is not None:
            logger.info(f"--- PARSED ACTION: {action.name} ---")
        else:
            logger.info("--- PARSED ACTION: Failed ---")
        
        logger.info("=" * 60)
    
    def get_last_interaction(self) -> Dict[str, Any]:
        """Get last LLM interaction for logging."""
        return {
            "system_prompt": self.last_system_prompt,
            "user_prompt": self.last_user_prompt,
            "response": self.last_response,
            "action": self.last_action.name if self.last_action is not None else None,
            "feedback": self.last_action_feedback,
            "story": self.last_story,
            "narrator_interaction": self.narrator.get_last_interaction(),
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics including both LLMs."""
        base_metrics = super().get_metrics()
        narrator_metrics = self.narrator.get_metrics()
        
        base_metrics.update({
            "rooms_discovered": len(self.room_objects),
            "rooms_classified": len(self.room_classification),
            "rooms_visited": len(self.visited_rooms),
            "room_display_names": dict(self.room_display_names),
            "unique_objects_seen": len(self.object_id_to_name),
            "conversation_turns": len(self.conversation.messages),
            
            # Main planner tokens
            "main_planner_queries": self.total_queries,
            "main_planner_input_tokens": self.total_input_tokens,
            "main_planner_output_tokens": self.total_output_tokens,
            "main_planner_total_tokens": self.total_input_tokens + self.total_output_tokens,
            
            # Narrator tokens
            "narrator_queries": narrator_metrics["total_queries"],
            "narrator_input_tokens": narrator_metrics["total_input_tokens"],
            "narrator_output_tokens": narrator_metrics["total_output_tokens"],
            "narrator_total_tokens": narrator_metrics["total_tokens"],
            
            # Combined
            "total_queries": self.total_queries + narrator_metrics["total_queries"],
            "total_tokens": (self.total_input_tokens + self.total_output_tokens + 
                           narrator_metrics["total_tokens"]),
        })
        return base_metrics
    
    def classify_rooms(
        self,
        room_objects: Dict[str, List[str]],
        open_set: bool = True
    ) -> Dict[str, str]:
        """Classify rooms based on contents."""
        if not room_objects:
            return {}
        
        room_list = "\n".join(
            f" - {room} contains [{', '.join(objs)}]."
            for room, objs in room_objects.items()
        )
        
        possible_rooms = ["bathroom", "bedroom", "closet", "corridor", "dining room",
                         "entryway", "garage", "hallway", "kitchen", "laundry room",
                         "living room", "office", "other room", "outdoor", "stairs"]
        
        if open_set:
            request = "Classify each room based on its contents. If unsure or empty, use 'other room'."
            remember = ""
        else:
            request = f"Classify rooms into: {', '.join(possible_rooms)}. If unsure, use 'other room'."
            remember = "Remember: only use the given categories."
        
        user_prompt = ROOM_CLASSIFICATION_USER_PROMPT.format(
            num_rooms=len(room_objects),
            room_object_list=room_list,
            request=request,
            remember=remember
        )
        
        response = self._query_room_classification(user_prompt)
        
        return self._parse_room_classification(
            response,
            list(room_objects.keys()),
            possible_rooms if not open_set else None
        )
    
    def _query_room_classification(self, user_prompt: str) -> str:
        """Query for room classification."""
        try:
            from openai import OpenAI
            
            client = self._openai_client
            if client is None:
                key = os.environ.get("OPENAI_API_KEY")
                if not key:
                    logger.warning("No OpenAI API key for room classification")
                    return ""
                client = OpenAI(api_key=key)
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": ROOM_CLASSIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=256,
                temperature=0.0
            )
            
            if response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
            
            return response.choices[0].message.content or ""
            
        except Exception as e:
            logger.error(f"Room classification query failed: {e}")
            return ""
    
    def _parse_room_classification(
        self,
        response: str,
        room_ids: List[str],
        possible_rooms: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Parse room classification from response."""
        classification = {}
        
        for line in response.lower().split("\n"):
            for room_id in room_ids:
                if room_id in classification:
                    continue
                
                room_key = room_id.lower()
                if room_key not in line and room_key.replace("-", " ") not in line:
                    continue
                
                if ":" in line:
                    room_type = line.split(":")[-1].strip().strip(".-")
                    
                    if possible_rooms:
                        matched = next((r for r in possible_rooms if r in room_type), None)
                        room_type = matched or "other room"
                    
                    classification[room_id] = room_type
                    break
        
        for room_id in room_ids:
            if room_id not in classification:
                classification[room_id] = "other room"
        
        return classification


# Convenience factory functions
def create_two_planner_openai_agent(
    main_model: str = "gpt-4o",
    narrator_model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    action_level: str = "high",
    **kwargs
) -> TwoPlannerAgent:
    """
    Create a Two-Planner agent using OpenAI API.
    
    Args:
        main_model: Model for main planner (default: gpt-4o)
        narrator_model: Model for narrator (default: gpt-4o-mini for cost efficiency)
        api_key: OpenAI API key
        temperature: Sampling temperature
        action_level: "low" or "high"
        **kwargs: Additional arguments
        
    Returns:
        Configured TwoPlannerAgent
    """
    return TwoPlannerAgent(
        api_type="openai",
        main_model=main_model,
        narrator_model=narrator_model,
        openai_api_key=api_key,
        temperature=temperature,
        action_level=action_level,
        **kwargs
    )


def create_two_planner_custom_agent(
    api_url: str = "http://localhost:8000/generate",
    main_model: str = "default",
    narrator_model: str = "default",
    temperature: float = 0.7,
    action_level: str = "high",
    **kwargs
) -> TwoPlannerAgent:
    """
    Create a Two-Planner agent using custom API.
    
    Args:
        api_url: Custom API URL
        main_model: Model for main planner
        narrator_model: Model for narrator
        temperature: Sampling temperature
        action_level: "low" or "high"
        **kwargs: Additional arguments
        
    Returns:
        Configured TwoPlannerAgent
    """
    return TwoPlannerAgent(
        api_type="custom",
        api_url=api_url,
        main_model=main_model,
        narrator_model=narrator_model,
        temperature=temperature,
        action_level=action_level,
        **kwargs
    )

