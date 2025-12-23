"""
LLM-based Navigation Agent

Uses a language model for high-level navigation decisions.
Supports both custom API endpoints and OpenAI API.
"""

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import requests

from .base import BaseAgent, AgentState
from ..utils.actions import Action, HighLevelAction, HighLevelActionSpace
from ..core.observations import ProcessedObservation

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


# System prompt for navigation
SYSTEM_PROMPT = """You are a navigation agent exploring an indoor environment to find a target object.

TASK: {task_description}

AVAILABLE ACTIONS:
{action_descriptions}

RULES:
1. Choose ONE action per response
2. Format your response as: <action>(<argument>)
3. Explore systematically - prefer unexplored areas
4. Call stop() only when you have found the target and are close to it
5. If you get stuck, try a different approach

OUTPUT FORMAT:
First provide brief reasoning, then on a new line starting with "Command: " give your action.
Example:
Reasoning: The target object might be in the kitchen. I should explore there.
Command: goto(kitchen)
"""

USER_PROMPT = """Current situation:
- Location: {current_room}
- Nearby objects: {nearby_objects}
- Rooms discovered: {discovered_rooms}
- Recent actions: {action_history}
- Unexplored areas: {unexplored_info}

What is your next action to find the {target_object}?"""

# Room classification prompts
ROOM_CLASSIFICATION_SYSTEM_PROMPT = """You are a helpful assistant identifying room types in an apartment based on their contents."""

ROOM_CLASSIFICATION_USER_PROMPT = """You observe {num_rooms} rooms containing the following objects:
{room_object_list}

{request}

Respond with a bullet list in this exact format:
 - room-X: room type

{remember}
Include ONLY the bullet list, no other text."""


class LLMAgent(BaseAgent):
    """
    LLM-based navigation agent.
    
    Uses a language model (via API) to make high-level navigation decisions.
    The agent outputs semantic actions like goto(kitchen) or explore(bedroom)
    which are then converted to low-level navigation commands.
    
    Supports:
    - Custom API endpoints (local LLM servers, etc.)
    - OpenAI API (GPT-4, GPT-4o, GPT-3.5-turbo, etc.)
    
    Example with OpenAI:
        agent = LLMAgent(
            api_type="openai",
            model_name="gpt-4o",
            openai_api_key="sk-..."
        )
    
    Example with custom API:
        agent = LLMAgent(
            api_type="custom",
            api_url="http://localhost:8000/generate"
        )
    """
    
    def __init__(self,
                 api_type: str = "custom",
                 api_url: str = "http://localhost:8000/generate",
                 model_name: str = "gpt-4o",
                 openai_api_key: Optional[str] = None,
                 max_tokens: int = 256,
                 temperature: float = 0.7,
                 max_retries: int = 3,
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
        
        # OpenAI client
        self._openai_client = None
        if self.api_type == APIType.OPENAI:
            self._init_openai(openai_api_key)
        
        # Conversation tracking
        self.conversation = Conversation(messages=[])
        
        # Scene understanding
        self.discovered_rooms: Dict[str, List[str]] = {}  # room_id -> objects
        self.room_classification: Dict[str, str] = {}  # room_id -> classified name (e.g., "room-0" -> "kitchen")
        self.visited_rooms: set = set()
        self.frontier_info: List[str] = []
        
        # Room classification settings
        self._pending_room_classification: bool = False
        self._last_classified_room_count: int = 0
        
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_queries = 0
        
        logger.info(f"Initialized LLMAgent with {self.api_type.value} API, model={model_name}")
    
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
            info: Dict[str, Any]) -> HighLevelAction:
        """
        Query LLM and return high-level action.
        
        Args:
            observation: Current processed observation
            task_description: Natural language task description
            info: Additional info from environment
            
        Returns:
            HighLevelAction to execute
        """
        # Update scene understanding
        self._update_scene_understanding(observation, info)
        
        # Check if target is visible and close
        if self._should_stop(observation):
            return HighLevelAction("stop", "", "Target found")
        
        # Build prompt
        system_prompt, user_prompt = self._build_prompts(task_description, info)
        print(f"\033[94m{system_prompt}\033[0m")
        print(f"\033[91m{user_prompt}\033[0m")
        # Query LLM
        response = self._query_llm(system_prompt, user_prompt)
        # print blue colored response
        print(f"\033[94m{response}\033[0m")

        # Parse action from response
        action = self._parse_action(response)
        
        # Record action
        if action:
            self.state.add_action(action.name, action.argument, True)
        
        return action or HighLevelAction("stop", "", "Parse failed")
    
    def reset(self, target_category: str = ""):
        """Reset agent for new episode."""
        self.state.reset(target_category)
        self.conversation.clear()
        self.discovered_rooms = {}
        self.room_classification = {}
        self.visited_rooms = set()
        self.frontier_info = []
        self._pending_room_classification = False
        self._last_classified_room_count = 0
        logger.debug(f"LLMAgent reset for target: {target_category}")
    
    def reset_token_counts(self):
        """Reset token usage counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_queries = 0
    
    def _update_scene_understanding(self, 
                                    observation: ProcessedObservation,
                                    info: Dict[str, Any]):
        """Update internal scene model from observation."""
        # Track observed objects
        self.state.add_observation(observation)
        
        # Update room info - track all rooms including "unknown"
        current_room = info.get("current_room", "unknown")
        
        # Track visited rooms (even "unknown" areas)
        self.visited_rooms.add(current_room)
        
        # Initialize room if new
        if current_room not in self.discovered_rooms:
            self.discovered_rooms[current_room] = []
            self._pending_room_classification = True  # New room discovered
        
        # Add visible objects to current room
        if observation.visible_objects:
            for obj in observation.visible_objects:
                if obj not in self.discovered_rooms[current_room]:
                    self.discovered_rooms[current_room].append(obj)
        
        # Update frontier info
        if "frontier_info" in info:
            self.frontier_info = info["frontier_info"]
        
        # Run room classification if new rooms discovered or objects added
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
        """Classify rooms if new rooms were discovered or significant objects added."""
        # Only classify if we have new rooms or it's been a while
        current_room_count = len(self.discovered_rooms)
        
        if not self._pending_room_classification and current_room_count == self._last_classified_room_count:
            return
        
        # Only classify rooms with objects
        rooms_with_objects = {
            room_id: objs for room_id, objs in self.discovered_rooms.items()
            if objs  # Only include rooms with objects
        }
        
        if not rooms_with_objects:
            return
        
        try:
            new_classification = self.classify_rooms(rooms_with_objects, open_set=True)
            
            # Merge with existing classification, updating as needed
            for room_id, room_type in new_classification.items():
                # Update if room wasn't classified or got more objects
                if room_id not in self.room_classification or room_type != "other room":
                    self.room_classification[room_id] = room_type
            
            self._pending_room_classification = False
            self._last_classified_room_count = current_room_count
            
            logger.debug(f"Room classification updated: {self.room_classification}")
            
        except Exception as e:
            logger.warning(f"Room classification failed: {e}")
    
    def _get_classified_room_name(self, room_id: str) -> str:
        """Get classified room name, falling back to room_id if not classified."""
        return self.room_classification.get(room_id, room_id)
    
    def _build_prompts(self, task_description: str, info: Dict[str, Any]) -> Tuple[str, str]:
        """Build system and user prompts for LLM, using classified room names."""
        # Format room info with classified names
        rooms_str = []
        for room_id, objects in self.discovered_rooms.items():
            room_name = self._get_classified_room_name(room_id)
            obj_list = ", ".join(objects[:10]) if objects else "no objects seen"
            visited = "(visited)" if room_id in self.visited_rooms else "(not visited)"
            rooms_str.append(f"  - {room_name} {visited}: [{obj_list}]")
        discovered_rooms = "\n".join(rooms_str) if rooms_str else "  - None yet"
        
        # Format nearby objects
        nearby = ", ".join(sorted(self.state.seen_objects)[:15]) if self.state.seen_objects else "none visible"
        
        # Format action history
        history = self.state.get_action_summary(max_actions=5)
        
        # Unexplored info
        unexplored = ", ".join(self.frontier_info[:5]) if self.frontier_info else "explore to discover"
        
        # Get current room with classified name
        raw_current_room = info.get("current_room", "unknown")
        current_room = self._get_classified_room_name(raw_current_room)
        
        # Build system prompt
        system = SYSTEM_PROMPT.format(
            task_description=task_description,
            action_descriptions=HighLevelActionSpace.get_action_descriptions()
        )
        
        # Build user prompt with classified room names
        user = USER_PROMPT.format(
            current_room=current_room,
            nearby_objects=nearby,
            discovered_rooms=discovered_rooms,
            action_history=history,
            unexplored_info=unexplored,
            target_object=self.state.target_category
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
    
    def _query_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Query OpenAI API."""
        if self._openai_client is None:
            logger.error("OpenAI client not initialized")
            return "Command: explore(unknown)"
        
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
            return result or "Command: explore(unknown)"
            
        except Exception as e:
            logger.error(f"OpenAI query failed: {e}")
            return "Command: explore(unknown)"
    
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
            return text or "Command: explore(unknown)"
            
        except Exception as e:
            logger.error(f"Custom API query failed: {e}")
            return "Command: explore(unknown)"
    
    def _parse_action(self, response: str) -> Optional[HighLevelAction]:
        """Parse action from LLM response."""
        # Look for Command: line
        command_match = None
        for line in response.split("\n"):
            line_clean = line.strip().lower()
            if line_clean.startswith("command:"):
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
        
        logger.warning(f"Failed to parse action from: {response[:100]}...")
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics including token usage."""
        base_metrics = super().get_metrics()
        base_metrics.update({
            "rooms_discovered": len(self.discovered_rooms),
            "rooms_classified": len(self.room_classification),
            "rooms_visited": len(self.visited_rooms),
            "room_classification": dict(self.room_classification),
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
