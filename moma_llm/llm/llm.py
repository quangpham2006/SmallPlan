# Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and
# proprietary rights in and to this software, related documentation and any
# modifications thereto. Any use, reproduction, disclosure or distribution of
# this software and related documentation without an express license agreement
# from Toyota Motor Europe NV/SA is strictly prohibited.
import copy
import os
import re
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pprint import pformat, pprint
from typing import List, Tuple, Union

import inflect
import networkx as nx
import numpy as np
from igibson import object_states
from openai import OpenAI, OpenAIError
from pygments import highlight
from pygments.formatters import Terminal256Formatter, TerminalFormatter
from pygments.lexers import PythonLexer
from sty import fg

from moma_llm.utils.constants import NODETYPE, POSSIBLE_ROOMS

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List
import requests 
client = OpenAI(organization=os.environ["OPENAI_ORGANIZATION"], api_key=os.environ["OPENAI_API_KEY"])
inflect_engine = inflect.engine()

# model_name_LLM_hugging = "Qwen/Qwen2.5-3B-Instruct"#Qwen/Qwen2.5-1.5B-Instruct
# tokenizer_hugging = AutoTokenizer.from_pretrained(model_name_LLM_hugging)
# model_hugging = AutoModelForCausalLM.from_pretrained(model_name_LLM_hugging)


def pprint_color(obj, style="staroffice", width=200):
    txt = highlight(pformat(obj, width=width), PythonLexer(), Terminal256Formatter(style=style))
    print(txt, end="")
    return txt

class Conversation_api:
    def __init__(self, messages):
        self.messages = messages

    def add_message(self, message):
        self.messages.append(message)

class Conversation:
    def __init__(self, messages: List[dict], include_env_messages: bool = False) -> None:
        self._messages = messages
        self._include_env_messages = include_env_messages
        
    def add_message(self, message: dict):
        self._messages.append(message)
        
    @property
    def messages(self):
        if self._include_env_messages:
            return self._messages
        else:
            return [m for m in self._messages if m["role"].lower() not in  ["env", "environment"]]
    
    @property
    def messages_including_env(self):
        return self._messages


@lru_cache(maxsize=None)
def send_query_cached(messages: list, model: str, temperature: float):
    assert temperature == 0.0, "Caching only works for temperature=0.0, as eitherwise we want to get different responses back"
    messages = [dict(m) for m in messages]
    return client.chat.completions.create(model=model,
                                          messages=messages,
                                          temperature=temperature,) 


def send_query(messages: list, model: str, temperature: float):
    if temperature == 0.0:
        hashable_messages = tuple(tuple(m.items()) for m in messages)
        return send_query_cached(messages=hashable_messages, model=model, temperature=temperature)
    else:
        return client.chat.completions.create(model=model,
                                              messages=messages,
                                              temperature=temperature,) 

class LLM_hugging:
    def __init__(self, 
                 room_classification_model: str,
                 open_set_rooms: bool = False,  
                 temperature: float = 0.0,
                 debug: bool = False) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.room_classification_model = room_classification_model
        self.temperature = temperature
        self.debug = debug
        self.open_set_rooms = open_set_rooms
    def train_PPO(self, reward):
        url = "http://127.0.0.1:7000/train"
        headers = {"Content-Type": "application/json"}
        data = {"reward": reward}

        response = requests.post(url, json=data, headers=headers)
        print(response.status_code)
        print(response.json())  # If the response contains JSON data

    def save_PPO_checkpoint(self, save_name): 
        url = "http://127.0.0.1:7000/save_model"
        params = {"checkpoint_name": save_name}  # Change this to your desired name

        response = requests.post(url, params=params)

        print(response.status_code)
        print(response.json())  # If the response contains JSON data

    def SFT_query(self, conversation: Conversation):
        """Example input:
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Knock knock."},
                {"role": "assistant", "content": "Who's there?"},
                {"role": "user", "content": "Orange."},
            ],
        """
        num_attempts = 0
        while True:
            try:
                response = send_query(model="gpt-4-1106-preview",
                                      messages=conversation.messages[: -1],
                                      temperature=self.temperature,)
                break
            except OpenAIError as e:
                print(f"Attempting again after {e}")
                num_attempts += 1
                time.sleep(5)
            assert num_attempts < 10, "Too many openai errors"
        
        role = response.choices[0].message.role
        content = response.choices[0].message.content
        
        url = "http://127.0.0.1:7000/train"
        conversation_api = Conversation_api(conversation.messages[: -1])
        data = {"messages": conversation_api.messages, "target_response": content}
        response = requests.post(url, json=data)
        print(response.status_code)
        print(response.json())

    
    def send_query(self, conversation: Conversation) -> str:
        """Generates a response using the LLaMA model from Hugging Face.

        Args:
            conversation: An object containing the conversation history.
            model: Optional model override.

        Returns:
            The generated response as a string.
        """
        url = "http://127.0.0.1:7000/chat"

        conversation_api = Conversation_api(conversation.messages)
        data = {"messages": conversation_api.messages}
        response = requests.post(url, json=data)
        if response.status_code == 200:
            assistant_response = response.json()["output"]
            
            # print("Assistant:", assistant_response)
        else:
            print(f"Error: {response.status_code}, {response.text}")

        # # Construct input prompt from conversation messages
        # text = self.tokenizer.apply_chat_template(
        #     conversation.messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        # model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # generated_ids = self.model.generate(
        #     model_inputs.input_ids,
        #     max_new_tokens=512
        # )
        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        # ]
        
        # response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        
        if self.debug:
            pprint_color("#################################\n", width=200)
            pprint_color("\n+++++++++++++++++++++++++++++++++\n".join([f"{m['role']}: {m['content']}" for m in conversation_api.messages]),  width=200)
            pprint_color(f"==================================\n assistant: {assistant_response}",  width=200, style="rrt")

        conversation.add_message({"role": "assistant", "content": assistant_response})
        # conversation.add_message({"role": "assistant", "content": response})
        # return response
        return assistant_response

    @staticmethod
    def to_human_readable_object_name(object_name: str, states=None) -> str:
        if states is not None:
            if object_states.Open in states:
                prefix = "opened " if states[object_states.Open] else "closed "
                object_name = prefix + object_name
        return re.sub(r'-\d+', '', re.sub(r'_\d+', '', object_name)).replace("_", "-")
    
    @staticmethod
    def human_to_graph_name(object_name: str) -> str:
        # strip states, replace spaces. Will still not be an exact match, as original node names include a number
        object_name = object_name.replace("closed", "").replace("opened", "").replace("-", "_")
        # replace numbers that stand alone (i.e. object counters in the beginning of the object name)
        # NOTE: not stripping the potential 's' for plural objects atm
        return re.sub(r'\b\d+\b', '', object_name).strip(" ")

    @staticmethod
    def create_room_object_dict(graph, open_door_inclusion: str = "as_object", room_classification = None, include_explored: bool = False) -> dict:
        rooms = list(graph.successors("root"))
        room_dict = {}
        for room in rooms:
            objects = sorted(list(graph.successors(room)))
            objects_readable = [LLM.to_human_readable_object_name(o, states=graph.nodes[o]["states"]) for o in objects]
            # can't add object_states to the name parsing since open doors are not a node in the graph atm
            if open_door_inclusion == "as_object":
                open_doors = [LLM.to_human_readable_object_name(d[0], states={object_states.Open: True}) for d in list(graph.nodes[room]["open_doors"])] 
            elif open_door_inclusion == "as_edge":
                assert room_classification is not None
                open_doors = [LLM.to_human_readable_object_name(d[0], states={object_states.Open: True}) + (f" to {room_classification[d[1]]}" if d[1] is not None else "") for d in list(graph.nodes[room]["open_doors"])] 
            elif open_door_inclusion == "ignore":
                open_doors = []
            else:
                raise ValueError(f"Unknown open_door_inclusion: {open_door_inclusion}")
            objects_readable += open_doors
            
            if include_explored:
                has_frontiers = len(graph.nodes.get(room)["frontier_points"])
                if has_frontiers:
                    objects_readable.append("unexplored area")
            
            occurences = Counter(objects_readable)
            counted_objects = [f"{v} {inflect_engine.plural(k) if inflect_engine.plural(k) else (k + 's')}" if (v > 1) else k for k, v in occurences.items()]
            room_dict[room] = counted_objects
        return room_dict  

    def _parse_rooms(self, input_rooms: list, possible_rooms: list, response: str):
        room_classification = {}
        
        lines = response.split("\n")
        for line in lines:
            line = line.lower()
            for room in input_rooms:
                if room in room_classification.keys():
                    # assume that the first mention of a room is the correct one, as usually it responds with a list of the rooms, then adds an explanation
                    continue
                if not self.open_set_rooms:
                    if (room in line) or (room.replace("_", " ") in line):
                        for possible_room in possible_rooms:
                            if possible_room in line:
                                room_classification[room] = possible_room
                                break
                        # Allow open-vocab answers, better than failing
                        if (room not in room_classification) and (":" in line):
                            room_classification[room] = line.split(":")[1].strip()
                else:    
                    if (room in line) or (room.replace("_", " ") in line):
                        room_classification[room] = line.split(":")[-1].strip()
                        break
        # If only one room is explored so far the LLM tends to answer in a full sentence instead of list
        # NOTE: could delete the following, have not observed this behaviour in a while
        if room_classification.keys() != set(input_rooms) and len(input_rooms) == 1 and len(lines) == 1:
            for possible_room in possible_rooms:
                if possible_room in lines[0]:
                    room_classification[room] = possible_room
                    break
                            
        assert room_classification.keys() == set(input_rooms), f"Did not find all input_rooms in the response: {room_classification}"
        
        if self.debug:
            pprint(room_classification)
        return room_classification

    def classify_rooms(self, obs, system_prompt: str = "You are a helpful assistant, visiting a new apartment.") -> dict:
        graph = obs["room_object_graph"]
        rooms = list(graph.successors("root"))
        
        room_dict = self.create_room_object_dict(graph, include_explored=False)

        prompt = f"You observe {len(rooms)} rooms, they contain the following objects:\n"
        for room in rooms:
            prompt += f"- {room} contains [{', '.join(room_dict[room])}].\n"
        if not self.open_set_rooms:
            prompt += f"Please classify the rooms into the following categories: {', '.join(POSSIBLE_ROOMS)}. If you are unsure, classify them as other room.\n"
        else:
            prompt += "Please classify the rooms. If you are unsure, classify them as other room.\n"
        prompt += "Output Response Format:\n"\
                "A list with bullet points of the form\n"\
                "- room-X: room type\n"
        if not self.open_set_rooms:
            prompt += "Remember: you can only use the given categories."
                           
        conversation = Conversation(messages=[{"role": "system", "content": system_prompt},
                                              {"role": "user", "content": prompt}])
        for i in range(3):
            try:
                response = self.send_query(conversation=conversation)
                output = self._parse_rooms(input_rooms=rooms, possible_rooms=POSSIBLE_ROOMS, response=response)
                break
            except:
                conversation.add_message({"role": "user", "content": "Please classify all rooms listed."})
                assert i < 2, "Could not parse room classification"
        
        return output
    
class LLM:
    def __init__(self, 
                 model: str, 
                 room_classification_model: str,
                 open_set_rooms: bool = False,  
                 temperature: float = 0.0,
                 debug=False) -> None:
        assert os.environ["OPENAI_API_KEY"] != "todo", "Please set OPENAI_API_KEY environment variable"
        available_models = [m.id for m in client.models.list().data]
        assert model in available_models, available_models
        self.model = model
        self.room_classification_model = room_classification_model
        self.temperature = temperature
        self.debug = debug
        self.open_set_rooms = open_set_rooms
        
    def send_query(self, conversation: Conversation, model: str=None):
        """Example input:
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Knock knock."},
                {"role": "assistant", "content": "Who's there?"},
                {"role": "user", "content": "Orange."},
            ],
        """
        num_attempts = 0
        while True:
            try:
                response = send_query(model=model or self.model,
                                      messages=conversation.messages,
                                      temperature=self.temperature,)
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
            pprint_color("\n+++++++++++++++++++++++++++++++++\n".join([f"{m['role']}: {m['content']}" for m in conversation.messages]), width=200)
            pprint_color(f"==================================\n {role}: {content}", width=200, style="rrt")

        conversation.add_message({"role": role, "content": content})

        return content

    @staticmethod
    def to_human_readable_object_name(object_name: str, states=None) -> str:
        if states is not None:
            if object_states.Open in states:
                prefix = "opened " if states[object_states.Open] else "closed "
                object_name = prefix + object_name
        return re.sub(r'-\d+', '', re.sub(r'_\d+', '', object_name)).replace("_", "-")
    
    @staticmethod
    def human_to_graph_name(object_name: str) -> str:
        # strip states, replace spaces. Will still not be an exact match, as original node names include a number
        object_name = object_name.replace("closed", "").replace("opened", "").replace("-", "_")
        # replace numbers that stand alone (i.e. object counters in the beginning of the object name)
        # NOTE: not stripping the potential 's' for plural objects atm
        return re.sub(r'\b\d+\b', '', object_name).strip(" ")

    @staticmethod
    def create_room_object_dict(graph, open_door_inclusion: str = "as_object", room_classification = None, include_explored: bool = False) -> dict:
        rooms = list(graph.successors("root"))
        room_dict = {}
        for room in rooms:
            objects = sorted(list(graph.successors(room)))
            objects_readable = [LLM.to_human_readable_object_name(o, states=graph.nodes[o]["states"]) for o in objects]
            # can't add object_states to the name parsing since open doors are not a node in the graph atm
            if open_door_inclusion == "as_object":
                open_doors = [LLM.to_human_readable_object_name(d[0], states={object_states.Open: True}) for d in list(graph.nodes[room]["open_doors"])] 
            elif open_door_inclusion == "as_edge":
                assert room_classification is not None
                open_doors = [LLM.to_human_readable_object_name(d[0], states={object_states.Open: True}) + (f" to {room_classification[d[1]]}" if d[1] is not None else "") for d in list(graph.nodes[room]["open_doors"])] 
            elif open_door_inclusion == "ignore":
                open_doors = []
            else:
                raise ValueError(f"Unknown open_door_inclusion: {open_door_inclusion}")
            objects_readable += open_doors
            
            if include_explored:
                has_frontiers = len(graph.nodes.get(room)["frontier_points"])
                if has_frontiers:
                    objects_readable.append("unexplored area")
            
            occurences = Counter(objects_readable)
            counted_objects = [f"{v} {inflect_engine.plural(k) if inflect_engine.plural(k) else (k + 's')}" if (v > 1) else k for k, v in occurences.items()]
            room_dict[room] = counted_objects
        return room_dict  

    def _parse_rooms(self, input_rooms: list, possible_rooms: list, response: str):
        room_classification = {}
        
        lines = response.split("\n")
        for line in lines:
            line = line.lower()
            for room in input_rooms:
                if room in room_classification.keys():
                    # assume that the first mention of a room is the correct one, as usually it responds with a list of the rooms, then adds an explanation
                    continue
                if not self.open_set_rooms:
                    if (room in line) or (room.replace("_", " ") in line):
                        for possible_room in possible_rooms:
                            if possible_room in line:
                                room_classification[room] = possible_room
                                break
                        # Allow open-vocab answers, better than failing
                        if (room not in room_classification) and (":" in line):
                            room_classification[room] = line.split(":")[1].strip()
                else:    
                    if (room in line) or (room.replace("_", " ") in line):
                        room_classification[room] = line.split(":")[-1].strip()
                        break
        # If only one room is explored so far the LLM tends to answer in a full sentence instead of list
        # NOTE: could delete the following, have not observed this behaviour in a while
        if room_classification.keys() != set(input_rooms) and len(input_rooms) == 1 and len(lines) == 1:
            for possible_room in possible_rooms:
                if possible_room in lines[0]:
                    room_classification[room] = possible_room
                    break
                            
        assert room_classification.keys() == set(input_rooms), f"Did not find all input_rooms in the response: {room_classification}"
        
        if self.debug:
            pprint(room_classification)
        return room_classification

    def classify_rooms(self, obs, system_prompt: str = "You are a helpful assistant, visiting a new apartment.") -> dict:
        graph = obs["room_object_graph"]
        rooms = list(graph.successors("root"))
        
        room_dict = self.create_room_object_dict(graph, include_explored=False)

        prompt = f"You observe {len(rooms)} rooms, they contain the following objects:\n"
        for room in rooms:
            prompt += f"- {room} contains [{', '.join(room_dict[room])}].\n"
        if not self.open_set_rooms:
            prompt += f"Please classify the rooms into the following categories: {', '.join(POSSIBLE_ROOMS)}. If you are unsure, classify them as other room.\n"
        else:
            prompt += "Please classify the rooms. If you are unsure, classify them as other room.\n"
        prompt += "Output Response Format:\n"\
                "A list with bullet points of the form\n"\
                "- room-X: room type\n"
        if not self.open_set_rooms:
            prompt += "Remember: you can only use the given categories."
                           
        conversation = Conversation(messages=[{"role": "system", "content": system_prompt},
                                              {"role": "user", "content": prompt}])
        for i in range(3):
            try:
                response = self.send_query(conversation=conversation, model=self.room_classification_model)
                output = self._parse_rooms(input_rooms=rooms, possible_rooms=POSSIBLE_ROOMS, response=response)
                break
            except:
                conversation.add_message({"role": "user", "content": "Please classify all rooms listed."})
                assert i < 2, "Could not parse room classification"
        return output
