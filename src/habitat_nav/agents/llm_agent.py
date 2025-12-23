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
        self.discovered_rooms: Dict[str, List[str]] = {}  # room -> objects
        self.visited_rooms: set = set()
        self.frontier_info: List[str] = []
        
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
        self.visited_rooms = set()
        self.frontier_info = []
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
        
        # Update room info if available
        current_room = info.get("current_room", "unknown")
        if current_room != "unknown":
            self.visited_rooms.add(current_room)
            
            if current_room not in self.discovered_rooms:
                self.discovered_rooms[current_room] = []
            
            if observation.visible_objects:
                for obj in observation.visible_objects:
                    if obj not in self.discovered_rooms[current_room]:
                        self.discovered_rooms[current_room].append(obj)
        
        # Update frontier info
        if "frontier_info" in info:
            self.frontier_info = info["frontier_info"]
    
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
    
    def _build_prompts(self, task_description: str, info: Dict[str, Any]) -> Tuple[str, str]:
        """Build system and user prompts for LLM."""
        # Format room info
        rooms_str = []
        for room, objects in self.discovered_rooms.items():
            obj_list = ", ".join(objects[:10]) if objects else "no objects seen"
            explored = "(visited)" if room in self.visited_rooms else "(not visited)"
            rooms_str.append(f"  - {room} {explored}: [{obj_list}]")
        discovered_rooms = "\n".join(rooms_str) if rooms_str else "  - None yet"
        
        # Format nearby objects
        nearby = ", ".join(sorted(self.state.seen_objects)[:15]) if self.state.seen_objects else "none visible"
        
        # Format action history
        history = self.state.get_action_summary(max_actions=5)
        
        # Unexplored info
        unexplored = ", ".join(self.frontier_info[:5]) if self.frontier_info else "explore to discover"
        
        # Build system prompt
        system = SYSTEM_PROMPT.format(
            task_description=task_description,
            action_descriptions=HighLevelActionSpace.get_action_descriptions()
        )
        
        # Build user prompt
        user = USER_PROMPT.format(
            current_room=info.get("current_room", "unknown"),
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
            "rooms_visited": len(self.visited_rooms),
            "conversation_turns": len(self.conversation.messages),
            "total_queries": self.total_queries,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
        })
        return base_metrics


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
