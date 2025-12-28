"""
Image-Enhanced Planner Agent for Habitat Navigation

This module implements an LLM agent that uses visual context from images
to make navigation decisions. The agent collects recent images and sends
them to GPT-4o (or other vision-capable LLMs) for visual understanding.

Based on the single planner implementation but with image context.
"""

import base64
import io
import logging
import os
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..base import BaseAgent, AgentState
from .prompts import (
    IMAGE_PLANNER_SYSTEM_PROMPT,
    IMAGE_PLANNER_USER_PROMPT,
    ROOM_CLASSIFICATION_SYSTEM_PROMPT,
    ROOM_CLASSIFICATION_USER_PROMPT,
    format_action_history,
    get_decision_guidance,
    count_recent_failures,
    check_target_in_objects,
    format_discovered_rooms,
)
from ...utils.actions import Action, ActionSpace, HighLevelAction, HighLevelActionSpace
from ...core.observations import ProcessedObservation

logger = logging.getLogger(__name__)


class APIType(Enum):
    """Supported API types for LLM queries."""
    OPENAI = "openai"


@dataclass
class ImageContext:
    """Container for image context to send to the LLM."""
    images: List[np.ndarray]  # List of RGB images
    descriptions: List[str]   # Description for each image
    
    def to_base64_list(self) -> List[Tuple[str, str]]:
        """Convert images to base64 strings with descriptions."""
        result = []
        for img, desc in zip(self.images, self.descriptions):
            # Convert numpy array to PIL Image
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            
            pil_img = Image.fromarray(img)
            
            # Convert RGBA to RGB (JPEG doesn't support alpha channel)
            if pil_img.mode == 'RGBA':
                pil_img = pil_img.convert('RGB')
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=85)
            b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            result.append((b64_str, desc))
        
        return result


class ImagePlannerAgent(BaseAgent):
    """
    Image-Enhanced LLM Planner Agent.
    
    Uses GPT-4o (or other vision-capable LLMs) with image context
    to make navigation decisions. Collects recent images from
    different viewing angles to provide visual context.
    
    Features:
    - Sends up to N recent images to the LLM
    - Uses base64-encoded images in OpenAI Vision API
    - Maintains image buffer for recent views
    - Falls back to text-only if image sending fails
    
    Example:
        agent = ImagePlannerAgent(
            model_name="gpt-4o",
            max_images=3,
            action_level="high"
        )
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000/generate",
        model_name: str = "gpt-4o",
        openai_api_key: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        max_retries: int = 3,
        action_level: str = "high",
        max_images: int = 3,
        image_detail: str = "low",  # "low", "high", or "auto" for OpenAI
        name: str = "image_planner_agent"
    ):
        """
        Initialize Image Planner agent.
        
        Args:
            api_url: URL for custom LLM API (not used for OpenAI)
            model_name: Model identifier (must be vision-capable, e.g., "gpt-4o")
            openai_api_key: OpenAI API key
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            max_retries: Maximum retries on parse failure
            action_level: "low" or "high" level actions
            max_images: Maximum number of images to send (default: 3)
            image_detail: Image detail level for OpenAI ("low", "high", "auto")
            name: Agent name
        """
        super().__init__(name=name)
        
        # API configuration
        self.api_type = APIType.OPENAI
        self.api_url = api_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Action level
        self.action_level = action_level.lower()
        if self.action_level not in ("low", "high"):
            raise ValueError(f"action_level must be 'low' or 'high', got '{action_level}'")
        
        # Image settings
        self.max_images = max_images
        self.image_detail = image_detail
        
        # Image buffer - stores recent images
        self.image_buffer: deque = deque(maxlen=max_images * 2)  # Keep extra for selection
        
        # OpenAI client
        self._openai_client = None
        self._init_openai(openai_api_key)
        
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
        self.last_images_sent: int = 0
        
        # Action history: (action_name, argument, success, feedback)
        self.action_history: List[Tuple[str, str, bool, str]] = []
        
        logger.info(f"Initialized ImagePlannerAgent: model={model_name}, "
                   f"max_images={max_images}, action_level={action_level}")
    
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
            logger.info("OpenAI client initialized for image planner")
            
        except ImportError:
            raise ImportError("OpenAI package required. Install with: pip install openai")
    
    def act(
        self,
        observation: ProcessedObservation,
        task_description: str,
        info: Dict[str, Any]
    ):
        """
        Query LLM with image context and return action.
        
        Args:
            observation: Current processed observation (includes RGB image)
            task_description: Natural language task description
            info: Additional info from environment
            
        Returns:
            Action (low-level) or HighLevelAction (high-level)
        """
        # Update scene understanding
        self._update_scene_understanding(observation, info)
        
        # Add current image to buffer
        self._add_image_to_buffer(observation.rgb, "Current view")
        
        # Get images to send
        images_to_send = self._get_images_for_context()
        
        # Build prompts
        system_prompt, user_prompt = self._build_prompts(
            task_description, info, observation, len(images_to_send)
        )
        
        # Query LLM with images
        response = self._query_llm_with_images(system_prompt, user_prompt, images_to_send)
        
        # Parse action
        action = self._parse_action(response)
        
        # Log interaction
        self._log_interaction(system_prompt, user_prompt, response, action, len(images_to_send))
        
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
        
        # Clear image buffer
        self.image_buffer.clear()
        
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
        
        # Reset action history
        self.action_history = []
        
        # Reset tracking
        self.last_system_prompt = ""
        self.last_user_prompt = ""
        self.last_response = ""
        self.last_action = None
        self.last_action_feedback = ""
        self.last_images_sent = 0
        
        logger.debug(f"ImagePlannerAgent reset for target: {target_category}")
    
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
            
            logger.info(f"ACTION RESULT: {action_name}({action_arg}) -> "
                       f"{'SUCCESS' if action_success else 'FAILED: ' + feedback_reason}")
    
    def _add_image_to_buffer(self, image: np.ndarray, description: str = ""):
        """Add an image to the buffer."""
        self.image_buffer.append({
            'image': image.copy(),
            'description': description,
            'step': self.state.total_steps
        })
    
    def _get_images_for_context(self) -> List[Tuple[np.ndarray, str]]:
        """
        Get images to send as context.
        
        Selects up to max_images from the buffer, prioritizing:
        1. Current view (most recent)
        2. Diverse viewpoints (spread across recent history)
        
        Returns:
            List of (image, description) tuples
        """
        if not self.image_buffer:
            return []
        
        buffer_list = list(self.image_buffer)
        
        # Always include the most recent image
        selected = [buffer_list[-1]]
        
        # Add older images spaced out through the buffer
        remaining = self.max_images - 1
        if remaining > 0 and len(buffer_list) > 1:
            # Select evenly spaced images from history
            older_images = buffer_list[:-1]
            step = max(1, len(older_images) // remaining)
            
            for i in range(0, len(older_images), step):
                if len(selected) >= self.max_images:
                    break
                selected.insert(0, older_images[i])  # Insert at beginning (oldest first)
        
        # Convert to format for API
        result = []
        for i, item in enumerate(selected):
            desc = item['description'] or f"View from step {item['step']}"
            if i == len(selected) - 1:
                desc = "Current view (most recent)"
            result.append((item['image'], desc))
        
        return result
    
    def _update_scene_understanding(
        self,
        observation: ProcessedObservation,
        info: Dict[str, Any]
    ):
        """Update internal scene model from observation."""
        current_room = info.get("current_room", "unknown")
        
        self.visited_rooms.add(current_room)
        
        # Track position
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
            raw_classification = self._classify_rooms(rooms_to_classify, open_set=True)
            
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
            
            self._pending_room_classification = False
            self._last_classified_room_count = current_room_count
            
        except Exception as e:
            logger.warning(f"Room classification failed: {e}")
    
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
        num_images: int = 0
    ) -> Tuple[str, str]:
        """Build system and user prompts."""
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
        
        # Build system prompt
        system_prompt = IMAGE_PLANNER_SYSTEM_PROMPT.format(
            task_description=task_description,
            action_descriptions=action_descriptions
        )
        
        # Build user prompt
        last_feedback = self.last_action_feedback if self.last_action_feedback else "None"
        
        user_prompt = IMAGE_PLANNER_USER_PROMPT.format(
            current_room=current_room,
            num_images=num_images,
            nearby_objects=nearby,
            discovered_rooms=discovered_rooms,
            action_history=action_history_formatted,
            last_feedback=last_feedback,
            unexplored_info=unexplored,
            exploration_status=exploration_status,
            decision_guidance=decision_guidance,
            target_object=self.state.target_category
        )
        
        return system_prompt, user_prompt
    
    def _query_llm_with_images(
        self,
        system_prompt: str,
        user_prompt: str,
        images: List[Tuple[np.ndarray, str]]
    ) -> str:
        """Query LLM with image context."""
        self.total_queries += 1
        self.last_images_sent = len(images)
        
        try:
            return self._query_openai_with_images(system_prompt, user_prompt, images)
        except Exception as e:
            logger.error(f"Image query failed: {e}. Falling back to text-only.")
            return self._query_openai_text_only(system_prompt, user_prompt)
    
    def _query_openai_with_images(
        self,
        system_prompt: str,
        user_prompt: str,
        images: List[Tuple[np.ndarray, str]]
    ) -> str:
        """Query OpenAI with images using Vision API."""
        if self._openai_client is None:
            logger.error("OpenAI client not initialized")
            return self._get_fallback_response()
        
        try:
            # Build message content with images
            content = []
            
            # Add images first
            for img, desc in images:
                # Convert to base64
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                
                pil_img = Image.fromarray(img)
                
                # Convert RGBA to RGB (JPEG doesn't support alpha channel)
                if pil_img.mode == 'RGBA':
                    pil_img = pil_img.convert('RGB')
                
                buffer = io.BytesIO()
                pil_img.save(buffer, format="JPEG", quality=85)
                b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_str}",
                        "detail": self.image_detail
                    }
                })
            
            # Add text prompt
            content.append({
                "type": "text",
                "text": user_prompt
            })
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]
            
            response = self._openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            if response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
            
            result = response.choices[0].message.content
            logger.debug(f"Image planner response: {result[:100]}...")
            return result or self._get_fallback_response()
            
        except Exception as e:
            logger.error(f"OpenAI vision query failed: {e}")
            raise
    
    def _query_openai_text_only(self, system_prompt: str, user_prompt: str) -> str:
        """Fallback to text-only query."""
        if self._openai_client is None:
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
            
            if response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
            
            return response.choices[0].message.content or self._get_fallback_response()
            
        except Exception as e:
            logger.error(f"Text-only query also failed: {e}")
            return self._get_fallback_response()
    
    def _get_fallback_response(self) -> str:
        """Get fallback response based on action level."""
        if self.action_level == "low":
            return "Action: TURN_LEFT"
        else:
            return "Command: explore()"
    
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
                return action
        
        logger.warning(f"Failed to parse high-level action from: {response[:100]}...")
        return None
    
    def _log_interaction(
        self,
        system_prompt: str,
        user_prompt: str,
        response: str,
        action: Optional[Action],
        num_images: int
    ):
        """Store and log LLM interaction."""
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.last_response = response
        self.last_action = action
        
        logger.info("=" * 60)
        logger.info(f"IMAGE PLANNER INTERACTION ({num_images} images)")
        logger.info("=" * 60)
        
        logger.info("--- SYSTEM PROMPT ---")
        system_lines = system_prompt.strip().split('\n')
        print(system_lines)
        
        print(f"--- USER PROMPT (with {num_images} images) ---")
        print(user_prompt)
        
        print("--- LLM RESPONSE ---")
        print(response)
        
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
            "images_sent": self.last_images_sent,
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        base_metrics = super().get_metrics()
        base_metrics.update({
            "rooms_discovered": len(self.room_objects),
            "rooms_classified": len(self.room_classification),
            "rooms_visited": len(self.visited_rooms),
            "unique_objects_seen": len(self.object_id_to_name),
            "total_queries": self.total_queries,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "images_in_buffer": len(self.image_buffer),
            "max_images": self.max_images,
        })
        return base_metrics
    
    def _classify_rooms(
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
        
        try:
            response = self._openai_client.chat.completions.create(
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
            
            result = response.choices[0].message.content or ""
            return self._parse_room_classification(result, list(room_objects.keys()), 
                                                   possible_rooms if not open_set else None)
            
        except Exception as e:
            logger.error(f"Room classification query failed: {e}")
            return {room_id: "other room" for room_id in room_objects}
    
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


# Convenience factory function
def create_image_planner_agent(
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_images: int = 3,
    action_level: str = "high",
    **kwargs
) -> ImagePlannerAgent:
    """
    Create an Image Planner agent using OpenAI Vision API.
    
    Args:
        model: OpenAI model name (must be vision-capable, e.g., "gpt-4o")
        api_key: OpenAI API key
        temperature: Sampling temperature
        max_images: Maximum number of images to send
        action_level: "low" or "high"
        **kwargs: Additional arguments
        
    Returns:
        Configured ImagePlannerAgent
    """
    return ImagePlannerAgent(
        model_name=model,
        openai_api_key=api_key,
        temperature=temperature,
        max_images=max_images,
        action_level=action_level,
        **kwargs
    )

