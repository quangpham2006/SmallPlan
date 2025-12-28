"""
MoMa-LLM Style Navigation Agent for Habitat

Uses a language model for high-level navigation decisions following
the MoMa-LLM paper's approach with dynamic scene graphs.

Reference: https://github.com/robot-learning-freiburg/MoMa-LLM

Supports both custom API endpoints and OpenAI API.
"""

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from ..base import BaseAgent, AgentState
from .prompts import (
    LOW_LEVEL_SYSTEM_PROMPT, LOW_LEVEL_USER_PROMPT,
    HIGH_LEVEL_SYSTEM_PROMPT, HIGH_LEVEL_USER_PROMPT,
    ROOM_CLASSIFICATION_SYSTEM_PROMPT, ROOM_CLASSIFICATION_USER_PROMPT,
    format_action_history, get_decision_guidance, format_retry_prompt,
    count_recent_failures, check_target_in_objects, format_discovered_rooms
)
from ...utils.actions import Action, ActionSpace, HighLevelAction, HighLevelActionSpace
from ...core.observations import ProcessedObservation

logger = logging.getLogger(__name__)


class APIType(Enum):
    """Supported API types for LLM queries."""
    CUSTOM = "custom"      # Custom API endpoint
    OPENAI = "openai"      # OpenAI API


@dataclass
class Conversation:
    """Conversation history with the LLM."""
    messages: List[Dict[str, str]]
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation."""
        self.messages.append({"role": role, "content": content})
    
    def get_prompt(self) -> str:
        """Get full conversation as a single prompt (for custom API)."""
        parts = []
        for msg in self.messages:
            role = msg["role"].upper()
            parts.append(f"[{role}]\n{msg['content']}")
        return "\n\n".join(parts)
    
    def get_openai_messages(self) -> List[Dict[str, str]]:
        """Get messages in OpenAI format."""
        return self.messages.copy()
    
    def clear(self):
        """Clear conversation history."""
        self.messages = []


class LLMAgent(BaseAgent):
    """
    MoMa-LLM style navigation agent for Habitat.
    
    Uses a language model with dynamic scene graph for navigation decisions
    following the MoMa-LLM paper's approach.
    
    Reference: https://github.com/robot-learning-freiburg/MoMa-LLM
    
    Supports two action levels:
    - "low": Low-level actions (MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, STOP)
    - "high": High-level MoMa-LLM actions (navigate, explore, go_to_and_open, done)
    
    Supports:
    - Custom API endpoints (local LLM servers, etc.)
    - OpenAI API (GPT-4, GPT-4o, GPT-3.5-turbo, etc.)
    
    Example with high-level actions (MoMa-LLM style):
        agent = LLMAgent(api_type="openai", model_name="gpt-4o", action_level="high")
    """
    
    def __init__(self,
                 api_type: str = "custom",
                 api_url: str = "http://localhost:8000/generate",
                 model_name: str = "gpt-4o",
                 openai_api_key: Optional[str] = None,
                 max_tokens: int = 256,
                 temperature: float = 0.7,
                 max_retries: int = 3,
                 action_level: str = "low",
                 name: str = "llm_agent"):
        """
        Initialize LLM agent.
        
        Args:
            api_type: Type of API - "custom" or "openai"
            api_url: URL for custom LLM API (ignored for OpenAI)
            model_name: Model identifier (e.g., "gpt-4o", "gpt-3.5-turbo")
            openai_api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-2.0)
            max_retries: Maximum retries on parse failure
            action_level: "low" for low-level actions, "high" for high-level actions
            name: Agent name
        """
        super().__init__(name=name)
        
        # API configuration
        self.api_type = APIType(api_type.lower())
        self.api_url = api_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Action level: "low" or "high"
        self.action_level = action_level.lower()
        if self.action_level not in ("low", "high"):
            raise ValueError(f"action_level must be 'low' or 'high', got '{action_level}'")
        
        # OpenAI client
        self._openai_client = None
        if self.api_type == APIType.OPENAI:
            self._init_openai(openai_api_key)
        
        # Conversation tracking
        self.conversation = Conversation(messages=[])
        
        # Scene understanding - restructured for proper ID tracking
        # Maps room_id -> set of object_ids seen in that room
        self.room_objects: Dict[str, set] = {}
        # Maps object_id -> object category name
        self.object_id_to_name: Dict[int, str] = {}
        # Maps room_id -> classified room name (e.g., "bedroom", "kitchen 2")
        self.room_classification: Dict[str, str] = {}
        # Maps room_id -> display name (classified name or room_id if unclassified)
        self.room_display_names: Dict[str, str] = {}
        # Tracks visited rooms (internal)
        self.visited_rooms: set = set()
        # Maps room_id -> is_fully_explored (based on exploration coverage, not just current view)
        self.room_exploration_status: Dict[str, bool] = {}
        # Maps room_id -> set of positions visited in that room (for proper exploration tracking)
        self.room_visited_positions: Dict[str, set] = {}
        # Current frontier info
        self.frontier_info: List[str] = []
        
        # Room classification settings
        self._pending_room_classification: bool = False
        self._last_classified_room_count: int = 0
        self._classification_name_counts: Dict[str, int] = {}  # Track how many rooms have each name
        
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_queries = 0
        
        # Last interaction tracking (for logging)
        self.last_system_prompt: str = ""
        self.last_user_prompt: str = ""
        self.last_response: str = ""
        self.last_action: Any = None  # Can be Action or HighLevelAction
        self.last_action_feedback: str = ""
        
        # Action history as list of tuples: (action_name, argument, success, feedback)
        self.action_history: List[Tuple[str, str, bool, str]] = []
        
        logger.info(f"Initialized LLMAgent: api={self.api_type.value}, model={model_name}, action_level={action_level}")
    
    def _init_openai(self, api_key: Optional[str] = None):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            
            # Get API key from parameter or environment
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                    "or pass openai_api_key parameter."
                )
            
            self._openai_client = OpenAI(api_key=key)
            logger.info("OpenAI client initialized successfully")
            
        except ImportError:
            raise ImportError(
                "OpenAI package required for OpenAI API. "
                "Install with: pip install openai"
            )
    
    def act(self,
            observation: ProcessedObservation,
            task_description: str,
            info: Dict[str, Any]):
        """
        Query LLM and return action.
        
        Args:
            observation: Current processed observation
            task_description: Natural language task description
            info: Additional info from environment
            
        Returns:
            Action (low-level) or HighLevelAction (high-level) based on action_level
        """
        # Update scene understanding
        self._update_scene_understanding(observation, info)
        
        # Build prompt with observation for object distances
        system_prompt, user_prompt = self._build_prompts(task_description, info, observation)
        
        # Query LLM
        response = self._query_llm(system_prompt, user_prompt)

        # Parse action from response
        action = self._parse_action(response)
        
        # Store and log the interaction
        self._log_interaction(system_prompt, user_prompt, response, action)
        
        # Record action
        if action:
            action_name = action.name if hasattr(action, 'name') else str(action)
            action_arg = action.argument if hasattr(action, 'argument') else ""
            self.state.add_action(action_name, action_arg, True)
        
        # Return default action if parsing failed
        if action is None:
            if self.action_level == "low":
                return Action.STOP
            else:
                # MoMa-LLM style: use "done" instead of "stop"
                return HighLevelAction("done", "", "Parse failed")
        
        return action
    
    def reset(self, target_category: str = ""):
        """Reset agent for new episode."""
        self.state.reset(target_category)
        self.conversation.clear()
        
        # Reset room and object tracking
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
        
        # Action history as list of tuples: (action_name, argument, success, feedback)
        self.action_history: List[Tuple[str, str, bool, str]] = []
        
        # Reset last interaction tracking
        self.last_system_prompt = ""
        self.last_user_prompt = ""
        self.last_response = ""
        self.last_action = None
        self.last_action_feedback = ""
        
        logger.debug(f"LLMAgent reset for target: {target_category}")
    
    def update(self,
               action,
               observation,
               info: Dict[str, Any]):
        """
        Update agent after taking an action.
        
        Records action in history with success/failure and feedback.
        Aligned with train_from_simulation_habitat action tracking.
        """
        # Call parent update
        self.state.add_observation(observation)
        
        # Determine action success
        action_success = True
        feedback_reason = ""
        
        # Extract feedback from info
        feedback_parts = []
        
        # Check for action result (from high-level action executor)
        if "action_result" in info:
            result = info["action_result"]
            if hasattr(result, 'success'):
                action_success = result.success
                if not action_success:
                    feedback_reason = getattr(result, 'feedback', '') or "action failed"
            
            if hasattr(result, 'feedback') and result.feedback:
                feedback_parts.append(result.feedback)
        
        # Check for action feedback from inference runner
        if "action_feedback" in info and info["action_feedback"]:
            feedback_parts.append(info["action_feedback"])
        
        # Check for collision status
        if "collision_status" in info and info["collision_status"]:
            feedback_parts.append(f"Collision: {info['collision_status']}")
            if not feedback_reason:
                feedback_reason = "collision detected"
        
        # Check for success/done
        if info.get("success"):
            feedback_parts.append("TASK COMPLETED - Target found!")
            action_success = True
        elif info.get("done"):
            llm_queries = info.get("llm_query_count", 0)
            step_count = info.get("step_count", 0)
            if llm_queries > 0:
                feedback_parts.append(f"Episode timeout ({llm_queries} LLM queries)")
            elif step_count >= info.get("max_steps", 500):
                feedback_parts.append(f"Episode timeout ({step_count} steps)")
            else:
                feedback_parts.append("Episode ended")
        
        # Store and log feedback
        self.last_action_feedback = " | ".join(feedback_parts) if feedback_parts else "Action executed"
        
        # Record action in history with feedback (aligned with train_from_simulation_habitat)
        if self.last_action is not None:
            action_name = self.last_action.name if hasattr(self.last_action, 'name') else str(self.last_action)
            action_arg = getattr(self.last_action, 'argument', '') if hasattr(self.last_action, 'argument') else ""
            
            # Add to action history as (action_name, argument, success, feedback_reason)
            self.action_history.append((
                action_name,
                action_arg,
                action_success,
                feedback_reason if not action_success else ""
            ))
            
            logger.info(f"ACTION RESULT: {action_name}({action_arg}) -> {'SUCCESS' if action_success else 'FAILED: ' + feedback_reason}")
    
    def _log_interaction(self, 
                        system_prompt: str, 
                        user_prompt: str, 
                        response: str,
                        action: Optional[Action]):
        """
        Store and log the LLM interaction.
        
        This logs the prompts and response for debugging and analysis.
        """
        # Store for later access
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.last_response = response
        self.last_action = action
        
        # Log with clear formatting
        logger.info("=" * 60)
        logger.info("LLM INTERACTION")
        logger.info("=" * 60)
        
        if system_prompt:
            logger.info("--- SYSTEM PROMPT ---")
            # Log first few lines of system prompt (it can be long)
            system_lines = system_prompt.strip().split('\n')
            print(system_lines)
        
        print("--- USER PROMPT ---")
        print(user_prompt)
        
        print("--- LLM RESPONSE ---")
        print(response)
        
        if action is not None:
            logger.info(f"--- PARSED ACTION ---")
            logger.info(f"Action: {action.name}")
        else:
            logger.info("--- PARSED ACTION ---")
            logger.info("Failed to parse action from response")
        
        logger.info("=" * 60)
    
    def get_last_interaction(self) -> Dict[str, Any]:
        """
        Get the last LLM interaction for external logging.
        
        Returns:
            Dictionary with prompts, response, action, and feedback
        """
        return {
            "system_prompt": self.last_system_prompt,
            "user_prompt": self.last_user_prompt,
            "response": self.last_response,
            "action": self.last_action.name if self.last_action is not None else None,
            "feedback": self.last_action_feedback,
        }
    
    def reset_token_counts(self):
        """Reset token usage counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_queries = 0
    
    def _update_scene_understanding(self, 
                                    observation: ProcessedObservation,
                                    info: Dict[str, Any]):
        """
        Update internal scene model from observation.
        
        Uses proper ID-based tracking:
        - room_objects: maps room_id -> set of object_ids
        - object_id_to_name: maps object_id -> category name
        - room_classification: maps room_id -> classified name with numbering
        """
        current_room = info.get("current_room", "unknown")
        
        # Track visited rooms
        self.visited_rooms.add(current_room)
        
        # Track position for exploration coverage
        if current_room not in self.room_visited_positions:
            self.room_visited_positions[current_room] = set()
        
        # Add current position to visited positions (discretized to 1m grid)
        pos = observation.position
        pos_key = (round(pos[0]), round(pos[2]))  # Discretize to 1m grid
        self.room_visited_positions[current_room].add(pos_key)
        
        # Collect visible objects with their IDs
        new_objects_found = False
        if observation.visible_object_info:
            for obj_info in observation.visible_object_info:
                obj_id = obj_info.obj_id
                
                # Skip if we've already seen this exact object
                if obj_id in self.object_id_to_name:
                    continue
                
                # Register this object
                self.object_id_to_name[obj_id] = obj_info.category
                new_objects_found = True
                
                # Initialize room if new
                if current_room not in self.room_objects:
                    self.room_objects[current_room] = set()
                    self._pending_room_classification = True
                    # Set initial display name as room_id until classified
                    self.room_display_names[current_room] = current_room
                
                # Add object to this room
                self.room_objects[current_room].add(obj_id)
        
        # Track observed objects in agent state
        self.state.add_observation(observation)
        
        # Update frontier info and room exploration status
        # Use position coverage for more accurate exploration tracking
        if "frontier_info" in info:
            self.frontier_info = info["frontier_info"]
            
            # Only track exploration status for rooms with objects
            if current_room in self.room_objects:
                # Check exploration based on both frontiers and position coverage
                positions_visited = len(self.room_visited_positions.get(current_room, set()))
                no_frontiers = not self.frontier_info or len(self.frontier_info) == 0
                
                # Consider room fully explored if no frontiers AND we've visited multiple positions
                if no_frontiers and positions_visited >= 3:
                    if not self.room_exploration_status.get(current_room, False):
                        self.room_exploration_status[current_room] = True
                        logger.debug(f"Marked {current_room} as fully explored ({positions_visited} positions, no frontiers)")
                else:
                    # Room has unexplored areas
                    self.room_exploration_status[current_room] = False
        
        # Run room classification if new rooms discovered or objects added
        if new_objects_found or self._pending_room_classification:
            self._maybe_classify_rooms()
    
    def _should_stop(self, observation: ProcessedObservation) -> bool:
        """Check if agent should stop (target found and close)."""
        target = self.state.target_category
        if not target:
            return False
        
        if observation.visible_objects and target in observation.visible_objects:
            # Additional check: are we close enough?
            # This is a heuristic based on depth at center
            center_depth = observation.depth[
                observation.depth.shape[0] // 2,
                observation.depth.shape[1] // 2
            ]
            if center_depth < 2.0:  # Within 2 meters
                return True
        
        return False
    
    def _maybe_classify_rooms(self):
        """
        Classify rooms if new rooms were discovered or significant objects added.
        
        Handles duplicate room names by adding numbering (e.g., "bedroom", "bedroom 2").
        Updates room_display_names with unique, human-readable names.
        """
        current_room_count = len(self.room_objects)
        
        if not self._pending_room_classification and current_room_count == self._last_classified_room_count:
            return
        
        # Get rooms that need classification (have objects but not yet classified)
        rooms_to_classify = {}
        for room_id, obj_ids in self.room_objects.items():
            if obj_ids:  # Only classify rooms with objects
                # Get object names for classification
                obj_names = [self.object_id_to_name.get(oid, "unknown") for oid in obj_ids]
                rooms_to_classify[room_id] = obj_names
        
        if not rooms_to_classify:
            return
        
        try:
            # Get raw classification from LLM
            raw_classification = self.classify_rooms(rooms_to_classify, open_set=True)
            
            # Track name counts for unique display names
            name_counts = {}
            
            # First pass: count all names (including already classified rooms)
            for room_id in self.room_objects:
                if room_id in self.room_classification:
                    base_name = self.room_classification[room_id]
                    name_counts[base_name] = name_counts.get(base_name, 0) + 1
            
            # Second pass: assign new classifications with unique numbering
            for room_id, room_type in raw_classification.items():
                if room_id not in self.room_classification or room_type != "other room":
                    # Store base classification
                    self.room_classification[room_id] = room_type
                    
                    # Generate unique display name
                    base_name = room_type
                    count = name_counts.get(base_name, 0) + 1
                    name_counts[base_name] = count
                    
                    if count == 1:
                        display_name = base_name
                    else:
                        display_name = f"{base_name} {count}"
                    
                    self.room_display_names[room_id] = display_name
            
            # Update display names for rooms that were already classified but may need renumbering
            self._update_display_names()
            
            self._pending_room_classification = False
            self._last_classified_room_count = current_room_count
            
            logger.debug(f"Room classification updated: {self.room_display_names}")
            
        except Exception as e:
            logger.warning(f"Room classification failed: {e}")
    
    def _update_display_names(self):
        """Update display names to ensure unique numbering for rooms with same classification."""
        # Group rooms by their base classification
        rooms_by_type: Dict[str, List[str]] = {}
        for room_id, room_type in self.room_classification.items():
            if room_type not in rooms_by_type:
                rooms_by_type[room_type] = []
            rooms_by_type[room_type].append(room_id)
        
        # Assign unique display names
        for room_type, room_ids in rooms_by_type.items():
            if len(room_ids) == 1:
                # Only one room of this type - no numbering needed
                self.room_display_names[room_ids[0]] = room_type
            else:
                # Multiple rooms of same type - add numbering
                for i, room_id in enumerate(sorted(room_ids)):
                    if i == 0:
                        self.room_display_names[room_id] = room_type
                    else:
                        self.room_display_names[room_id] = f"{room_type} {i + 1}"
    
    def _get_room_display_name(self, room_id: str) -> str:
        """
        Get human-readable display name for a room.
        
        Returns classified name with numbering if available,
        otherwise returns the raw room_id for unclassified rooms.
        """
        return self.room_display_names.get(room_id, room_id)
    
    def _get_room_objects_as_names(self, room_id: str) -> List[str]:
        """Get list of object category names for a room."""
        obj_ids = self.room_objects.get(room_id, set())
        return [self.object_id_to_name.get(oid, "unknown") for oid in obj_ids]
    
    def _build_prompts(self, 
                       task_description: str, 
                       info: Dict[str, Any],
                       observation: Optional[ProcessedObservation] = None) -> Tuple[str, str]:
        """
        Build system and user prompts for LLM.
        
        Uses proper display names (no IDs in prompts):
        - Classified room names with unique numbering
        - Object category names (not IDs)
        - Proper deduplication via ID tracking
        """
        # Build room info with display names and object names (not IDs)
        rooms_for_prompt = {}
        room_exploration_for_prompt = {}
        
        for room_id, obj_ids in self.room_objects.items():
            if not obj_ids:
                continue  # Skip empty rooms
            
            # Get display name (classified name or room_id if unclassified)
            display_name = self._get_room_display_name(room_id)
            
            # Get object names (deduplicated by category for display)
            obj_names = list(set(self._get_room_objects_as_names(room_id)))
            
            rooms_for_prompt[display_name] = obj_names
            room_exploration_for_prompt[display_name] = self.room_exploration_status.get(room_id, False)
        
        # Format discovered rooms using helper function
        discovered_rooms = format_discovered_rooms(
            rooms_for_prompt,
            room_exploration_for_prompt
        )
        
        # Format nearby objects with distances if available
        if observation and hasattr(observation, 'format_nearby_objects') and observation.visible_object_info:
            nearby = observation.format_nearby_objects(max_objects=15)
        elif self.state.seen_objects:
            nearby = ", ".join(sorted(self.state.seen_objects)[:15])
        else:
            nearby = "none visible"
        
        # Format action history using the new helper function
        # Convert action_history list to proper format for formatting
        action_history_formatted = format_action_history(self.action_history, max_items=5)
        
        # Unexplored info and exploration status
        if self.frontier_info:
            unexplored = ", ".join(self.frontier_info[:5])
        else:
            unexplored = "none - all nearby directions explored"
        
        # Build exploration status with room suggestions
        # Uses display names, ensures no duplicates
        exploration_status_parts = []
        
        # Collect unique room display names with their exploration status
        rooms_with_unexplored = set()
        fully_explored_set = set()
        
        for room_id, obj_ids in self.room_objects.items():
            if not obj_ids:
                continue  # Skip empty rooms
            
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
        
        # Add suggestions when current area is explored
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
        
        # Get current room with display name
        raw_current_room = info.get("current_room", "unknown")
        current_room = self._get_room_display_name(raw_current_room)
        
        # Generate decision guidance based on current state
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
        
        # Select prompts based on action level
        if self.action_level == "low":
            system_template = LOW_LEVEL_SYSTEM_PROMPT
            user_template = LOW_LEVEL_USER_PROMPT
            action_descriptions = ActionSpace.get_action_descriptions()
        else:
            system_template = HIGH_LEVEL_SYSTEM_PROMPT
            user_template = HIGH_LEVEL_USER_PROMPT
            action_descriptions = HighLevelActionSpace.get_action_descriptions()
        
        # Build system prompt
        system = system_template.format(
            task_description=task_description,
            action_descriptions=action_descriptions
        )
        
        # Get last action feedback
        last_feedback = self.last_action_feedback if self.last_action_feedback else "None"
        
        # Build user prompt with all the improved information
        user = user_template.format(
            current_room=current_room,
            nearby_objects=nearby,
            discovered_rooms=discovered_rooms,
            action_history=action_history_formatted,
            unexplored_info=unexplored,
            exploration_status=exploration_status,
            decision_guidance=decision_guidance,
            target_object=self.state.target_category,
            # last_feedback=last_feedback
        )
        
        # Update conversation for history
        self.conversation.add_message("system", system)
        self.conversation.add_message("user", user)
        
        return system, user
    
    def _query_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Query the LLM API based on configured API type."""
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
            # MoMa-LLM style: use navigate/explore/go_to_and_open/done
            return "Action: explore()"
    
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
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            # Track token usage
            if response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
            
            result = response.choices[0].message.content
            logger.debug(f"OpenAI response: {result[:100]}...")
            return result or self._get_fallback_response()
            
        except Exception as e:
            logger.error(f"OpenAI query failed: {e}")
            return self._get_fallback_response()
    
    def _query_custom_api(self, system_prompt: str, user_prompt: str) -> str:
        """Query custom LLM API."""
        try:
            # Build combined prompt for custom API
            prompt = self.conversation.get_prompt()
            
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
            
            # Handle different response formats
            text = result.get("response") or result.get("text") or result.get("content", "")
            return text or self._get_fallback_response()
            
        except Exception as e:
            logger.error(f"Custom API query failed: {e}")
            return self._get_fallback_response()
    
    def _parse_action(self, response: str):
        """Parse action from LLM response based on action_level."""
        if self.action_level == "low":
            return self._parse_low_level_action(response)
        else:
            return self._parse_high_level_action(response)
    
    def _parse_low_level_action(self, response: str) -> Optional[Action]:
        """Parse low-level action from LLM response."""
        # Look for Action: line
        action_str = None
        for line in response.split("\n"):
            line_clean = line.strip()
            if line_clean.lower().startswith("action:"):
                action_str = line.split(":", 1)[1].strip()
                break
        
        if not action_str:
            # Try to find action name anywhere in response
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
        """Parse high-level action from LLM response (MoMa-LLM style)."""
        # Look for Action: or Command: line (MoMa-LLM uses "Action:")
        command_match = None
        for line in response.split("\n"):
            line_clean = line.strip().lower()
            # MoMa-LLM style uses "Action:", but also support "Command:" for compatibility
            if line_clean.startswith("action:") or line_clean.startswith("command:"):
                command_text = line.split(":", 1)[1].strip()
                command_match = command_text
                break
        
        if not command_match:
            # Try to find action pattern anywhere
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
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics including token usage."""
        base_metrics = super().get_metrics()
        base_metrics.update({
            "rooms_discovered": len(self.room_objects),
            "rooms_classified": len(self.room_classification),
            "rooms_visited": len(self.visited_rooms),
            "room_display_names": dict(self.room_display_names),
            "unique_objects_seen": len(self.object_id_to_name),
            "conversation_turns": len(self.conversation.messages),
            "total_queries": self.total_queries,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
        })
        return base_metrics

    # =========================================================================
    # Room Classification
    # =========================================================================
    
    def classify_rooms(self, 
                       room_objects: Dict[str, List[str]],
                       open_set: bool = True) -> Dict[str, str]:
        """
        Classify rooms based on their object contents using GPT-4o.
        
        Args:
            room_objects: Dict mapping room IDs (e.g., "room-0") to list of objects
            open_set: If True, allow any room type. If False, restrict to predefined types.
            
        Returns:
            Dict mapping room IDs to classified room types (e.g., {"room-0": "kitchen"})
        
        Example:
            room_objects = {"room-0": ["sofa", "tv", "coffee table"], "room-1": ["bed", "wardrobe"]}
            classification = agent.classify_rooms(room_objects)
            # Returns: {"room-0": "living room", "room-1": "bedroom"}
        """
        if not room_objects:
            return {}
        
        # Build room object list string
        room_list = "\n".join(
            f" - {room} contains [{', '.join(objs)}]." 
            for room, objs in room_objects.items()
        )
        
        # Build request based on open_set mode
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
        
        # Query GPT-4o for classification
        response = self._query_room_classification(user_prompt)
        
        # Parse response
        return self._parse_room_classification(
            response, 
            list(room_objects.keys()), 
            possible_rooms if not open_set else None
        )
    
    def _query_room_classification(self, user_prompt: str) -> str:
        """Query GPT-4o for room classification."""
        try:
            from openai import OpenAI
            
            # Use existing client or create new one
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
    
    def _parse_room_classification(self,
                                   response: str,
                                   room_ids: List[str],
                                   possible_rooms: Optional[List[str]] = None) -> Dict[str, str]:
        """Parse room classification from LLM response."""
        classification = {}
        
        for line in response.lower().split("\n"):
            for room_id in room_ids:
                if room_id in classification:
                    continue
                
                # Check if this line mentions this room
                room_key = room_id.lower()
                if room_key not in line and room_key.replace("-", " ") not in line:
                    continue
                
                # Extract room type after colon
                if ":" in line:
                    room_type = line.split(":")[-1].strip().strip(".-")
                    
                    # If restricted mode, validate against possible rooms
                    if possible_rooms:
                        matched = next((r for r in possible_rooms if r in room_type), None)
                        room_type = matched or "other room"
                    
                    classification[room_id] = room_type
                    break
        
        # Fill missing rooms with "other room"
        for room_id in room_ids:
            if room_id not in classification:
                classification[room_id] = "other room"
        
        return classification


# Convenience factory functions
def create_openai_agent(
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
) -> LLMAgent:
    """
    Create an LLM agent using OpenAI API.
    
    Args:
        model: OpenAI model name (e.g., "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo")
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
        temperature: Sampling temperature
        **kwargs: Additional arguments for LLMAgent
        
    Returns:
        Configured LLMAgent instance
    """
    return LLMAgent(
        api_type="openai",
        model_name=model,
        openai_api_key=api_key,
        temperature=temperature,
        **kwargs
    )


def create_custom_api_agent(
    api_url: str = "http://localhost:8000/generate",
    model: str = "default",
    temperature: float = 0.7,
    **kwargs
) -> LLMAgent:
    """
    Create an LLM agent using a custom API endpoint.
    
    Args:
        api_url: Custom API endpoint URL
        model: Model identifier for the API
        temperature: Sampling temperature
        **kwargs: Additional arguments for LLMAgent
        
    Returns:
        Configured LLMAgent instance
    """
    return LLMAgent(
        api_type="custom",
        api_url=api_url,
        model_name=model,
        temperature=temperature,
        **kwargs
    )
