# Habitat-compatible LLM module for SmallPlan
# Replaces iGibson object_states dependencies with simulator-agnostic approach

import os
import re
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pprint import pformat, pprint
from typing import List, Tuple, Union, Dict, Any, Optional

import inflect
import networkx as nx
import numpy as np
from openai import OpenAI, OpenAIError
from pygments import highlight
from pygments.formatters import Terminal256Formatter, TerminalFormatter
from pygments.lexers import PythonLexer
from sty import fg

from moma_llm.utils.habitat_constants import POSSIBLE_ROOMS, NODETYPE
from moma_llm.env.prompts import ROOM_CLASSIFICATION_SYSTEM_PROMPT, ROOM_CLASSIFICATION_USER_PROMPT

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List
import requests
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# Object States - Simulator Agnostic
# ============================================================================

class ObjectStateType:
    """Base class for object state types."""
    pass


class OpenState(ObjectStateType):
    """Open state for doors/containers - simulator agnostic."""
    pass


class InsideState(ObjectStateType):
    """Inside relation state."""
    pass


class OnTopState(ObjectStateType):
    """OnTop relation state."""
    pass


class object_states:
    """
    Namespace for object states compatible with both iGibson and Habitat.
    Provides a simulator-agnostic interface.
    """
    Open = OpenState
    Inside = InsideState
    OnTop = OnTopState
    
    @staticmethod
    def get_state_value(states_dict: Dict, state_type: type, default: Any = None) -> Any:
        """
        Get state value from a states dictionary.
        
        Args:
            states_dict: Dictionary mapping state types to values
            state_type: The state type to look up
            default: Default value if state not found
            
        Returns:
            State value or default
        """
        if states_dict is None:
            return default
        return states_dict.get(state_type, default)


# ============================================================================
# Client and Utilities
# ============================================================================

client = OpenAI()
inflect_engine = inflect.engine()


def pprint_color(obj, style="staroffice", width=200):
    """Pretty print with syntax highlighting."""
    txt = highlight(pformat(obj, width=width), PythonLexer(), Terminal256Formatter(style=style))
    print(txt, end="")
    return txt


# ============================================================================
# Conversation Classes
# ============================================================================

class Conversation_api:
    """Simple conversation container for API calls."""
    def __init__(self, messages: List[Dict]):
        self.messages = messages

    def add_message(self, message: Dict):
        self.messages.append(message)


class Conversation:
    """
    Conversation manager with optional environment message filtering.
    """
    def __init__(self, messages: List[Dict], include_env_messages: bool = False) -> None:
        self._messages = messages
        self._include_env_messages = include_env_messages
        
    def add_message(self, message: Dict):
        self._messages.append(message)
        
    @property
    def messages(self) -> List[Dict]:
        if self._include_env_messages:
            return self._messages
        else:
            return [m for m in self._messages if m["role"].lower() not in ["env", "environment"]]
    
    @property
    def messages_including_env(self) -> List[Dict]:
        return self._messages


# ============================================================================
# Query Functions
# ============================================================================

@lru_cache(maxsize=None)
def send_query_cached(messages: tuple, model: str, temperature: float):
    """Cached query for deterministic responses (temperature=0)."""
    assert temperature == 0.0, "Caching only works for temperature=0.0"
    messages = [dict(m) for m in messages]
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )


def send_query(messages: List[Dict], model: str, temperature: float):
    """Send query to OpenAI API."""
    if temperature == 0.0:
        hashable_messages = tuple(tuple(m.items()) for m in messages)
        return send_query_cached(messages=hashable_messages, model=model, temperature=temperature)
    else:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )


# ============================================================================
# LLM Classes
# ============================================================================

class LLM_hugging:
    """
    LLM interface using Hugging Face models via API.
    Habitat-compatible version without iGibson dependencies.
    """
    
    def __init__(self,
                 room_classification_model: str,
                 open_set_rooms: bool = True,
                 temperature: float = 0.0,
                 debug: bool = False,
                 slm_api_url: str = None,
                 use_openai: bool = True,  # TEMPORARY: Switch to use OpenAI
                 openai_model: str = "gpt-4o") -> None:  # TEMPORARY: OpenAI model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.room_classification_model = room_classification_model
        self.temperature = temperature
        self.debug = debug
        self.open_set_rooms = open_set_rooms
        self.slm_api_url = slm_api_url
        self.use_openai = use_openai  # TEMPORARY
        self.openai_model = openai_model  # TEMPORARY
        
        # Token tracking
        self._episode_input_tokens = 0
        self._episode_output_tokens = 0
        self._episode_queries = 0
    
    def reset_episode_metrics(self):
        """Reset token metrics for new episode."""
        self._episode_input_tokens = 0
        self._episode_output_tokens = 0
        self._episode_queries = 0
    
    def get_episode_metrics(self) -> Dict[str, Any]:
        """Get token usage metrics for current episode."""
        return {
            'episode_input_tokens': self._episode_input_tokens,
            'episode_output_tokens': self._episode_output_tokens,
            'episode_total_tokens': self._episode_input_tokens + self._episode_output_tokens,
            'episode_llm_queries': self._episode_queries,
            'avg_tokens_per_query': (
                (self._episode_input_tokens + self._episode_output_tokens) / max(1, self._episode_queries)
            )
        }
    
    def train_PPO(self, reward: float):
        """Send reward signal for PPO training."""
        url = f"{self.slm_api_url}/train"
        headers = {"Content-Type": "application/json"}
        data = {"reward": reward}

        response = requests.post(url, json=data, headers=headers)
        print(response.status_code)
        print(response.json())

    def train_SFT(self, conversation: Conversation):
        """Supervised fine-tuning on conversation."""
        num_attempts = 0
        while True:
            try:
                response = send_query(
                    model="gpt-4-1106-preview",
                    messages=conversation.messages[:-1],
                    temperature=self.temperature
                )
                break
            except OpenAIError as e:
                print(f"Attempting again after {e}")
                num_attempts += 1
                time.sleep(5)
            assert num_attempts < 10, "Too many openai errors"
        
        content = response.choices[0].message.content
        
        url = f"{self.slm_api_url}/train"
        conversation_api = Conversation_api(conversation.messages[:-1])
        data = {"messages": conversation_api.messages, "target_response": content}
        train_response = requests.post(url, json=data)
        print(train_response.status_code)
        print(train_response.json())

    def save_checkpoint(self, save_name: str):
        """Save model checkpoint."""
        url = f"{self.slm_api_url}/save_model"
        params = {"checkpoint_name": save_name}
        response = requests.post(url, params=params)
        print(response.status_code)
        print(response.json())
    
    def send_query(self, conversation: Conversation) -> str:
        """
        Generate response using the model via API.
        
        Args:
            conversation: Conversation object with message history
            
        Returns:
            Generated response string
        """
        # TEMPORARY: Use OpenAI instead of SLM API
        if self.use_openai:
            try:
                conversation_api = Conversation_api(conversation.messages)
                
                # Use OpenAI API
                response = client.chat.completions.create(
                    model=self.openai_model,
                    messages=conversation_api.messages,
                    temperature=self.temperature
                )
                
                assistant_response = response.choices[0].message.content
                
                # Update token tracking with actual usage
                self._episode_queries += 1
                self._episode_input_tokens += response.usage.prompt_tokens
                self._episode_output_tokens += response.usage.completion_tokens
                
            except Exception as e:
                print(f"OpenAI Error: {e}")
                assistant_response = ""
        else:
            # Original SLM API code
            url = f"{self.slm_api_url}/chat"
            conversation_api = Conversation_api(conversation.messages)
            data = {"messages": conversation_api.messages}
            
            response = requests.post(url, json=data)
            if response.status_code == 200:
                assistant_response = response.json()["output"]
                
                # Update token tracking (estimate)
                self._episode_queries += 1
                # Rough token estimation
                input_text = " ".join([m["content"] for m in conversation_api.messages])
                self._episode_input_tokens += len(input_text.split()) * 1.3  # Rough token estimate
                self._episode_output_tokens += len(assistant_response.split()) * 1.3
            else:
                print(f"Error: {response.status_code}, {response.text}")
                assistant_response = ""

        if self.debug:
            pprint_color("#################################\n", width=200)
            pprint_color(
                "\n+++++++++++++++++++++++++++++++++\n".join(
                    [f"{m['role']}: {m['content']}" for m in conversation_api.messages]
                ),
                width=200
            )
            pprint_color(
                f"==================================\n assistant: {assistant_response}",
                width=200,
                style="rrt"
            )

        return assistant_response

    @staticmethod
    def to_human_readable_object_name(object_name: str, states: Optional[Dict] = None) -> str:
        """
        Convert object name to human-readable format.
        
        Args:
            object_name: Raw object name
            states: Optional dictionary of object states
            
        Returns:
            Human-readable object name
        """
        if states is not None:
            # Check for Open state - works with both iGibson and Habitat
            open_value = None
            for state_key, state_val in states.items():
                # Handle different state key formats
                if (state_key == object_states.Open or 
                    str(state_key).endswith('Open') or
                    getattr(state_key, '__name__', '') == 'Open' or
                    (hasattr(state_key, '__class__') and 
                     state_key.__class__.__name__ in ['Open', 'OpenState'])):
                    open_value = state_val
                    break
            
            if open_value is not None:
                prefix = "opened " if open_value else "closed "
                object_name = prefix + object_name
                
        return re.sub(r'-\d+', '', re.sub(r'_\d+', '', object_name)).replace("_", "-")
    
    @staticmethod
    def human_to_graph_name(object_name: str) -> str:
        """
        Convert human-readable name back to graph node name format.
        
        Args:
            object_name: Human-readable object name
            
        Returns:
            Graph-compatible object name
        """
        object_name = object_name.replace("closed", "").replace("opened", "").replace("-", "_")
        return re.sub(r'\b\d+\b', '', object_name).strip(" ")

    @staticmethod
    def create_room_object_dict(graph: nx.DiGraph,
                                open_door_inclusion: str = "as_object",
                                room_classification: Optional[Dict] = None,
                                include_explored: bool = False) -> Dict[str, List[str]]:
        """
        Create dictionary mapping rooms to their contained objects.
        
        Args:
            graph: Room-object graph
            open_door_inclusion: How to handle open doors ("as_object", "as_edge", "ignore")
            room_classification: Optional room classification mapping
            include_explored: Whether to include exploration status
            
        Returns:
            Dictionary mapping room names to object lists
        """
        rooms = list(graph.successors("root"))
        room_dict = {}
        
        for room in rooms:
            objects = sorted(list(graph.successors(room)))
            objects_readable = [
                LLM_hugging.to_human_readable_object_name(o, states=graph.nodes[o].get("states"))
                for o in objects
            ]
            
            # Handle open doors
            open_doors_data = list(graph.nodes.get(room, {}).get("open_doors", []))
            
            if open_door_inclusion == "as_object":
                open_doors = [
                    LLM_hugging.to_human_readable_object_name(
                        d[0] if isinstance(d, tuple) else d,
                        states={object_states.Open: True}
                    )
                    for d in open_doors_data
                ]
            elif open_door_inclusion == "as_edge":
                assert room_classification is not None
                open_doors = []
                for d in open_doors_data:
                    door_name = d[0] if isinstance(d, tuple) else d
                    connected_room = d[1] if (isinstance(d, tuple) and len(d) > 1) else None
                    readable_name = LLM_hugging.to_human_readable_object_name(
                        door_name,
                        states={object_states.Open: True}
                    )
                    if connected_room is not None and connected_room in room_classification:
                        readable_name += f" to {room_classification[connected_room]}"
                    open_doors.append(readable_name)
            elif open_door_inclusion == "ignore":
                open_doors = []
            else:
                raise ValueError(f"Unknown open_door_inclusion: {open_door_inclusion}")
                
            objects_readable += open_doors
            
            if include_explored:
                frontier_points = graph.nodes.get(room, {}).get("frontier_points", set())
                if len(frontier_points) > 0:
                    objects_readable.append("unexplored area")
            
            # Count occurrences and format
            occurrences = Counter(objects_readable)
            counted_objects = [
                f"{v} {inflect_engine.plural(k) if inflect_engine.plural(k) else (k + 's')}"
                if (v > 1) else k
                for k, v in occurrences.items()
            ]
            room_dict[room] = counted_objects
            
        return room_dict

    def _parse_rooms(self, input_rooms: List[str], possible_rooms: List[str], response: str) -> Dict[str, str]:
        """Parse room classification from LLM response."""
        room_classification = {}
        
        lines = response.split("\n")
        for line in lines:
            line = line.lower()
            for room in input_rooms:
                if room in room_classification.keys():
                    continue
                    
                if not self.open_set_rooms:
                    if (room in line) or (room.replace("_", " ") in line):
                        for possible_room in possible_rooms:
                            if possible_room in line:
                                room_classification[room] = possible_room
                                break
                        if (room not in room_classification) and (":" in line):
                            room_classification[room] = line.split(":")[1].strip()
                else:
                    if (room in line) or (room.replace("_", " ") in line):
                        room_classification[room] = line.split(":")[-1].strip()
                        break
                        
        # Handle single room case
        if room_classification.keys() != set(input_rooms) and len(input_rooms) == 1 and len(lines) == 1:
            for possible_room in possible_rooms:
                if possible_room in lines[0]:
                    room_classification[input_rooms[0]] = possible_room
                    break
                            
        assert room_classification.keys() == set(input_rooms), \
            f"Did not find all input_rooms in the response: {room_classification}"
        
        if self.debug:
            pprint(room_classification)
        return room_classification

    def classify_rooms(self, obs: Dict, system_prompt: str = ROOM_CLASSIFICATION_SYSTEM_PROMPT) -> Dict[str, str]:
        """
        Classify rooms based on their contents.
        
        Args:
            obs: Observation dictionary containing room_object_graph
            system_prompt: System prompt for room classification
            
        Returns:
            Dictionary mapping room IDs to room types
        """
        graph = obs["room_object_graph"]
        rooms = list(graph.successors("root"))
        
        room_dict = self.create_room_object_dict(graph, include_explored=False)

        num_rooms = len(rooms)
        room_object_list = ""
        for room in rooms:
            room_object_list += f"- {room} contains [{', '.join(room_dict[room])}].\n"
        
        # Debug: Print room contents to help diagnose classification issues
        if self.debug:
            print("\n========== ROOM CLASSIFICATION DEBUG ==========")
            print(f"Total rooms to classify: {num_rooms}")
            for room in rooms:
                obj_count = len(room_dict[room])
                print(f"{room}: {obj_count} objects - {room_dict[room]}")
            print("=" * 50 + "\n")

        llm_request = ""
        # if not self.open_set_rooms:
        llm_request += f"Please classify the rooms into the following categories: {', '.join(POSSIBLE_ROOMS)}. "
        llm_request += "If you are unsure or the room is empty, classify them as other room.\n"
        # else:
            # llm_request += "Please classify the rooms. If you are unsure, classify them as other room.\n"

        remember = ""
        if not self.open_set_rooms:
            remember += "Remember: you can only use the given categories."
        
        user_prompt = ROOM_CLASSIFICATION_USER_PROMPT.format(
            NUM_ROOMS=num_rooms,
            ROOM_OBJECT_LIST=room_object_list,
            REQUESTS=llm_request,
            REMEMBER=remember,
        )
        
        conversation = Conversation(messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        for i in range(3):
            try:
                response = self.send_query(conversation=conversation)
                output = self._parse_rooms(input_rooms=rooms, possible_rooms=POSSIBLE_ROOMS, response=response)
                break
            except Exception as e:
                print(f"Room classification attempt {i+1} failed: {e}")
                conversation.add_message({"role": "user", "content": "Please classify all rooms listed."})
                assert i < 2, "Could not parse room classification"
        
        return output


class LLM:
    """
    LLM interface using OpenAI API directly.
    Habitat-compatible version without iGibson dependencies.
    """
    
    def __init__(self,
                 model: str,
                 room_classification_model: str,
                 open_set_rooms: bool = True,
                 temperature: float = 0.0,
                 debug: bool = False) -> None:
        assert os.environ.get("OPENAI_API_KEY", "todo") != "todo", \
            "Please set OPENAI_API_KEY environment variable"
        available_models = [m.id for m in client.models.list().data]
        assert model in available_models, f"Model {model} not available. Available: {available_models}"
        
        self.model = model
        self.room_classification_model = room_classification_model
        self.temperature = temperature
        self.debug = debug
        self.open_set_rooms = open_set_rooms
        
    def send_query(self, conversation: Conversation, model: str = None) -> str:
        """Send query to OpenAI API."""
        num_attempts = 0
        while True:
            try:
                response = send_query(
                    model=model or self.model,
                    messages=conversation.messages,
                    temperature=self.temperature
                )
                break
            except OpenAIError as e:
                print(f"Attempting again after {e}")
                num_attempts += 1
                time.sleep(5)
            assert num_attempts < 10, "Too many openai errors"
        
        role = response.choices[0].message.role
        content = response.choices[0].message.content
        
        if self.debug:
            pprint_color("#################################\n", width=200)
            pprint_color(
                "\n+++++++++++++++++++++++++++++++++\n".join(
                    [f"{m['role']}: {m['content']}" for m in conversation.messages]
                ),
                width=200
            )
            pprint_color(f"==================================\n {role}: {content}", width=200, style="rrt")

        return content

    # Use the same static methods as LLM_hugging for consistency
    to_human_readable_object_name = LLM_hugging.to_human_readable_object_name
    human_to_graph_name = LLM_hugging.human_to_graph_name
    create_room_object_dict = LLM_hugging.create_room_object_dict
    _parse_rooms = LLM_hugging._parse_rooms
    
    def classify_rooms(self, obs: Dict, system_prompt: str = ROOM_CLASSIFICATION_SYSTEM_PROMPT) -> Dict[str, str]:
        """Classify rooms based on their contents."""
        graph = obs["room_object_graph"]
        rooms = list(graph.successors("root"))
        
        room_dict = self.create_room_object_dict(graph, include_explored=False)

        num_rooms = len(rooms)
        room_object_list = ""
        for room in rooms:
            room_object_list += f"- {room} contains [{', '.join(room_dict[room])}].\n"
        
        # Debug: Print room contents to help diagnose classification issues
        if self.debug:
            print("\n========== ROOM CLASSIFICATION DEBUG ==========")
            print(f"Total rooms to classify: {num_rooms}")
            for room in rooms:
                obj_count = len(room_dict[room])
                print(f"{room}: {obj_count} objects - {room_dict[room]}")
            print("=" * 50 + "\n")

        llm_request = ""
        #  if not self.open_set_rooms:
        llm_request += f"Please classify the rooms into the following categories: {', '.join(POSSIBLE_ROOMS)}. "
        llm_request += "If you are unsure, classify them as other room.\n"
        # else:
            # llm_request += "Please classify the rooms. If you are unsure, classify them as other room.\n"

        remember = ""
        if not self.open_set_rooms:
            remember += "Remember: you can only use the given categories."
        
        user_prompt = ROOM_CLASSIFICATION_USER_PROMPT.format(
            NUM_ROOMS=num_rooms,
            ROOM_OBJECT_LIST=room_object_list,
            REQUESTS=llm_request,
            REMEMBER=remember,
        )
                           
        conversation = Conversation(messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        for i in range(3):
            try:
                response = self.send_query(conversation=conversation, model=self.room_classification_model)
                output = self._parse_rooms(self, input_rooms=rooms, possible_rooms=POSSIBLE_ROOMS, response=response)
                break
            except Exception as e:
                print(f"Room classification attempt {i+1} failed: {e}")
                conversation.add_message({"role": "user", "content": "Please classify all rooms listed."})
                assert i < 2, "Could not parse room classification"
                
        return output

