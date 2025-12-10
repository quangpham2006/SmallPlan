"""
Habitat Multi-LLM Environment for SmallPlan.

This module extends HabitatLLMEnv to support a dual-LLM system where:
1. The main planning LLM decides actions (goto, explore, open, stop)
2. A storyteller LLM generates narrative summaries of the exploration

The storyteller's summary provides temporal context and helps the planning
LLM make more informed decisions by understanding what has been tried before.
"""

import re
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from moma_llm.env.habitat_env import OurHabitatEnv
from moma_llm.env.habitat_llm_env import (
    HabitatLLMEnv,
    HabitatHighLevelEnv,
    ActionHistory,
    DIST_MAPPING,
    distance_mapping,
    split_frontier_points
)
from moma_llm.llm.habitat_llm import LLM_hugging, Conversation, object_states
from moma_llm.llm.storyteller_llm import StorytellerLLM, ActionRecord

# Import prompts from different versions
from moma_llm.env import prompts as prompts_v1
from moma_llm.env import prompts_v2
from moma_llm.env import prompts_v3
from moma_llm.env import prompts_v4


class HabitatMultiLLMEnv(HabitatLLMEnv):
    """
    Multi-LLM environment with a planning LLM and a storyteller LLM.
    
    The storyteller LLM observes the planning context and generates
    narrative summaries that are fed back to the planning LLM to provide
    better temporal context and reasoning about the exploration journey.
    """
    
    def __init__(self, 
                 env: OurHabitatEnv, 
                 llm: LLM_hugging, 
                 seed: int,
                 storyteller_model: str = "gpt-4o-mini",
                 storyteller_temperature: float = 0.3,
                 enable_storyteller: bool = True,
                 debug_storyteller: bool = False,
                 prompt_version: int = 2) -> None:
        """
        Initialize the Multi-LLM environment.
        
        Args:
            env: The base Habitat environment
            llm: The main planning LLM
            seed: Random seed
            storyteller_model: Model to use for storytelling
            storyteller_temperature: Temperature for storyteller generation
            enable_storyteller: Whether to enable the storyteller (can disable for ablation)
            debug_storyteller: Whether to print storyteller debug info
            prompt_version: Prompt version to use (1, 2, or 4)
        """
        super().__init__(env=env, llm=llm, seed=seed)
        
        self.enable_storyteller = enable_storyteller
        self.prompt_version = prompt_version  # Store prompt version for _create_prompt
        
        # Initialize storyteller LLM with matching prompt version
        self.storyteller = StorytellerLLM(
            model=storyteller_model,
            temperature=storyteller_temperature,
            debug=debug_storyteller,
            prompt_version=prompt_version
        )
        
        # Track rooms before/after actions for storyteller
        self._room_before_action: str = ""
        self._prev_discovered_rooms: set = set()
    
    def reset(self, config_file: str, scene_id: str, episode_num: int) -> Dict:
        """Reset for new episode."""
        # Reset parent class
        obs = super().reset(config_file=config_file, scene_id=scene_id, episode_num=episode_num)
        
        # Reset storyteller
        self.storyteller.reset()
        self._room_before_action = ""
        self._prev_discovered_rooms = set()
        
        return obs
    
    def _get_new_discoveries(self, graph: nx.DiGraph) -> List[str]:
        """
        Get list of newly discovered rooms/objects since last action.
        
        Args:
            graph: Current room-object graph
            
        Returns:
            List of newly discovered room names
        """
        current_rooms = set(graph.successors("root")) if "root" in graph else set()
        new_rooms = current_rooms - self._prev_discovered_rooms
        self._prev_discovered_rooms = current_rooms
        return list(new_rooms)
    
    def _update_storyteller_context(self,
                                     task_description: str,
                                     current_room: str,
                                     room_dict: Dict[str, List[str]],
                                     close_objects: set,
                                     rooms_with_frontier_leading_out: List):
        """
        Update the storyteller with current context.
        
        Args:
            task_description: Current task description
            current_room: Current room name
            room_dict: Dict mapping rooms to object lists
            close_objects: Set of nearby objects
            rooms_with_frontier_leading_out: Rooms with unexplored areas
        """
        if not self.enable_storyteller:
            return
        
        # Convert room_dict values to lists if needed
        discovered_rooms = {
            room: list(objects) if isinstance(objects, (list, tuple)) else [str(objects)]
            for room, objects in room_dict.items()
        }
        
        # Get unexplored areas
        unexplored = [room for room, _ in rooms_with_frontier_leading_out]
        
        self.storyteller.update_context(
            task_description=task_description,
            current_room=current_room,
            discovered_rooms=discovered_rooms,
            nearby_objects=list(close_objects),
            unexplored_areas=unexplored
        )
    
    def _record_action_for_storyteller(self,
                                       action: str,
                                       argument: str,
                                       success: bool,
                                       feedback: str,
                                       room_after: str,
                                       new_discoveries: List[str]):
        """
        Record an action for the storyteller.
        
        Args:
            action: Action name
            argument: Action argument
            success: Whether action succeeded
            feedback: Feedback message
            room_after: Room after action
            new_discoveries: Newly discovered rooms
        """
        if not self.enable_storyteller:
            return
        
        self.storyteller.record_action(
            action=action,
            argument=argument,
            success=success,
            feedback=feedback,
            room_before=self._room_before_action,
            room_after=room_after,
            new_discoveries=new_discoveries
        )
    
    def _create_prompt(self,
                       task_description: str,
                       labelled_rooms: List[str],
                       current_room: str,
                       room_dict: Dict,
                       rooms_with_frontier_within: List,
                       rooms_with_frontier_leading_out: List,
                       rooms_with_closed_doors: List,
                       close_objects: set,
                       nlp_history: List[str],
                       room_distances: Dict,
                       target_name: str = "",
                       *args,
                       **kwargs) -> Conversation:
        """
        Create prompt for LLM with storytelling section.
        
        This extends the parent class method to include the storyteller's
        narrative summary in the prompt.
        """
        # Use the prompt version set in __init__ (not from config file)
        prompt_version = self.prompt_version
        
        # Generate tool descriptions
        tool_descriptions = ""
        for i, (action, description) in enumerate(self.possible_actions.items()):
            tool_descriptions += f"{i+1}. {action}({description[0]}): {description[1]}\n"

        list_nearby_objects = f"[{', '.join(sorted(close_objects))}]"

        # Track rooms with and without frontiers
        rooms_with_any_frontier = set([r[0] for r in rooms_with_frontier_within] + 
                                       [r[0] for r in rooms_with_frontier_leading_out])
        
        list_found_rooms_and_objects = ""
        for room in sorted(labelled_rooms):
            objects = room_dict.get(room, [])
            if room in rooms_with_any_frontier:
                objects = objects + ["(has unexplored areas)"]
            else:
                objects = objects + ["(fully explored)"]
            list_found_rooms_and_objects += f"- {room}: [{', '.join(objects)}]\n"

        rooms_with_frontier_descr = f"[{', '.join([f'{room} ({distance_mapping(dist)})' for room, dist in rooms_with_frontier_leading_out])}]"
        if not rooms_with_frontier_leading_out:
            rooms_with_frontier_descr = "[none - all rooms fully explored]"
        
        fully_explored_rooms = [r for r in labelled_rooms if r not in rooms_with_any_frontier]
        fully_explored_descr = f"[{', '.join(fully_explored_rooms)}]" if fully_explored_rooms else "[none yet]"
        
        visited_rooms_descr = f"[{', '.join(sorted(self.visited_rooms))}]" if self.visited_rooms else "[none yet]"
        
        rooms_with_closed_doors_descr = ""
        if len(rooms_with_closed_doors):
            rooms_with_closed_doors_descr = f"These rooms contain closed doors that might open up new space: [{', '.join([f'{room} ({distance_mapping(dist)})' for room, dist in rooms_with_closed_doors])}]."

        # Generate storytelling summary
        story_section = ""
        if self.enable_storyteller and len(self.action_history) > 0:
            # Generate/update the story
            story_summary = self.storyteller.generate_summary()
            if story_summary:
                story_section = self.storyteller.get_summary_for_prompt()

        if prompt_version == 4:
            # Use v4 prompts (simplified multi-LLM - no object lists, story carries context)
            # Format action history
            action_history_section = prompts_v4.format_action_history_from_dataclass(
                self.action_history,
                self.llm.to_human_readable_object_name
            )
            
            # Generate contextual decision guidance
            recent_failures = prompts_v4.count_recent_failures(self.action_history)
            has_unexplored = len(rooms_with_frontier_leading_out) > 0
            all_explored = len(fully_explored_rooms) == len(labelled_rooms) and len(labelled_rooms) > 0
            
            # Check if target is in nearby objects (simple check)
            target_found = target_name.lower() in [obj.lower() for obj in close_objects] if target_name else False
            
            decision_guidance = prompts_v4.get_decision_guidance(
                target_found=target_found,
                recent_failure_count=recent_failures,
                has_unexplored_areas=has_unexplored,
                all_rooms_explored=all_explored
            )
            
            # Extract closed door names from rooms_with_closed_doors
            # rooms_with_closed_doors is list of (room, distance) tuples
            closed_doors = [room for room, _ in rooms_with_closed_doors] if rooms_with_closed_doors else []
            
            # Use v4 prompt builder (simplified - no object lists)
            system_prompt, user_prompt = prompts_v4.build_main_llm_prompt(
                task_description=task_description,
                tool_descriptions=tool_descriptions,
                current_room=current_room,
                nearby_objects=list_nearby_objects,
                story_content=story_section.replace("=== EXPLORATION STORY ===\n", "").strip() if story_section else "",
                discovered_rooms=room_dict,  # Pass dict, v4 extracts just room names
                action_history_section=action_history_section,
                rooms_with_frontier=rooms_with_frontier_descr,
                fully_explored_rooms=fully_explored_descr,
                closed_doors=closed_doors,
                decision_guidance=decision_guidance,
                include_story=self.enable_storyteller
            )
        elif prompt_version == 2:
            # Use v3 prompts (multi-LLM version with story support)
            # Format action history with failure reasons
            action_history_section = prompts_v3.format_action_history_from_dataclass(
                self.action_history,
                self.llm.to_human_readable_object_name
            )
            
            # Generate contextual decision guidance
            target_found = prompts_v3.check_target_in_objects(target_name, room_dict) if target_name else False
            recent_failures = prompts_v3.count_recent_failures(self.action_history)
            has_unexplored = len(rooms_with_frontier_leading_out) > 0
            all_explored = len(fully_explored_rooms) == len(labelled_rooms) and len(labelled_rooms) > 0
            
            decision_guidance = prompts_v3.get_decision_guidance(
                target_found=target_found,
                target_name=target_name,
                recent_failure_count=recent_failures,
                has_unexplored_areas=has_unexplored,
                all_rooms_explored=all_explored
            )
            
            # Use v3 prompt builder for multi-LLM with story section
            system_prompt, user_prompt = prompts_v3.build_main_llm_prompt(
                task_description=task_description,
                tool_descriptions=tool_descriptions,
                current_room=current_room,
                nearby_objects=list_nearby_objects,
                story_content=story_section.replace("=== EXPLORATION STORY ===\n", "").strip() if story_section else "",
                found_rooms_and_objects=list_found_rooms_and_objects,
                action_history_section=action_history_section,
                rooms_with_frontier=rooms_with_frontier_descr,
                fully_explored_rooms=fully_explored_descr,
                visited_rooms=visited_rooms_descr,
                rooms_with_closed_doors=rooms_with_closed_doors_descr,
                decision_guidance=decision_guidance,
                include_story=self.enable_storyteller
            )
        else:
            # Use v1 prompts with storytelling added
            system_prompt = prompts_v1.SYSTEM_PROMPT.format(
                TASK_DESCRIPTION=task_description,
                TOOL_DESCRIPTIONS=tool_descriptions
            )
            
            list_previous_actions = ""
            if len(nlp_history):
                list_previous_actions = f"Your {len(nlp_history)} previous actions were: {', '.join(nlp_history)}."

            # Build user prompt with story section using v3 formatting
            formatted_story = ""
            if story_section and self.enable_storyteller:
                formatted_story = story_section

            user_prompt_parts = [
                f"Current location: {current_room}",
                f"Nearby objects: {list_nearby_objects}",
                "",
            ]
            
            if formatted_story:
                user_prompt_parts.append(formatted_story)
            
            user_prompt_parts.extend([
                "=== DISCOVERED ROOMS & OBJECTS ===",
                list_found_rooms_and_objects,
                f"=== PREVIOUS ACTIONS ===\n{list_previous_actions}" if list_previous_actions else "",
                "=== EXPLORATION STATUS ===",
                f"Can explore (has unexplored areas): {rooms_with_frontier_descr}",
                f"Fully explored: {fully_explored_descr}",
                f"Visited rooms: {visited_rooms_descr}",
                rooms_with_closed_doors_descr,
            ])
            
            user_prompt = "\n".join(user_prompt_parts)

        conversation = Conversation(messages=[
            self.last_env_feedback,
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        return conversation
    
    def take_action_inference(self, obs: Dict, task_description: str) -> Tuple[bool, bool, Dict]:
        """
        Take action based on observation (inference mode) with storytelling.
        
        This extends the parent method to:
        1. Update storyteller context before action
        2. Record actions for storytelling after execution
        """
        def _apply_room_classification(obs):
            obs["room_object_graph"] = nx.relabel_nodes(obs["room_object_graph"], self.room_classification)
            for n, d in obs["separated_voronoi_graph"].nodes(data=True):
                d["room_id"] = self.room_classification.get(
                    f"room-{d['room_id']}" if isinstance(d['room_id'], int) else d['room_id'],
                    f"room-{d['room_id']}"
                )

        try:
            self.classify_rooms(obs)
            _apply_room_classification(obs)
        except:
            print("Failed to classify rooms.")
            return False, False, self.env.episode_info

        graph = obs["room_object_graph"]
        labelled_rooms = list(graph.successors("root"))
        
        room_dict = self.llm.create_room_object_dict(
            graph,
            open_door_inclusion="ignore",
            room_classification=self.room_classification
        )
        current_room = self.room_classification.get(obs["robot_current_room"], "unknown")
        
        # Track room before action for storyteller
        self._room_before_action = current_room
        
        # Track visited rooms
        if current_room and current_room != "unknown":
            self.visited_rooms.add(current_room)

        def _get_closest_dist(points):
            if not points:
                return float('inf')
            idx, _, _costs, paths = self._find_closest_point(points)
            return self.env.slam.voxel_size * len(paths[idx])

        rooms_with_frontier_within = []
        rooms_with_frontier_leading_out = []
        rooms_with_closed_doors = []
        
        for n in labelled_rooms:
            frontier_points = graph.nodes.get(n, {}).get("frontier_points", set())
            if frontier_points:
                fp_within, fp_leading_out = split_frontier_points(frontier_points)
                if fp_within:
                    rooms_with_frontier_within.append((n, _get_closest_dist(fp_within)))
                if fp_leading_out:
                    rooms_with_frontier_leading_out.append((n, _get_closest_dist(fp_leading_out)))
                    
            closed_doors = graph.nodes.get(n, {}).get("closed_doors", [])
            if closed_doors:
                rooms_with_closed_doors.append((
                    n,
                    _get_closest_dist([graph.nodes.get(d) for d in closed_doors if d in graph.nodes])
                ))

        rooms_with_frontier_within = sorted(rooms_with_frontier_within, key=lambda x: x[1])
        rooms_with_frontier_leading_out = sorted(rooms_with_frontier_leading_out, key=lambda x: x[1])
        rooms_with_closed_doors = sorted(rooms_with_closed_doors, key=lambda x: x[1])

        close_objects = self._get_close_objects(graph=graph, current_room=current_room, closeness_thresh=2.5)
        nlp_history = self._match_action_history(graph=graph, separated_voronoi_graph=obs["separated_voronoi_graph"])

        def _calc_dist_to_room(current_room, separated_voronoi_graph):
            vnodes = defaultdict(list)
            for node_pos, node_data in separated_voronoi_graph.nodes(data=True):
                vnodes[node_data["room_id"]].append(node_pos)
            
            room_distances = {}
            for room in self.room_classification.values():
                if room == current_room:
                    room_distances[room] = "current location"
                elif room in vnodes:
                    _, _, _costs, paths = self._find_closest_point(
                        [self.env.slam.voxel2world(pos) for pos in vnodes[room]]
                    )
                    dist = self.env.slam.voxel_size * len(paths[0]) if paths else float('inf')
                    room_distances[room] = distance_mapping(dist)
            return room_distances

        room_distances = _calc_dist_to_room(current_room, obs["separated_voronoi_graph"])

        # Get target name
        target_name = ""
        if self.env.task and hasattr(self.env.task, 'target_category'):
            target_name = self.llm.to_human_readable_object_name(self.env.task.target_category)
        
        # Update storyteller context BEFORE generating prompt
        self._update_storyteller_context(
            task_description=task_description,
            current_room=current_room,
            room_dict=room_dict,
            close_objects=close_objects,
            rooms_with_frontier_leading_out=rooms_with_frontier_leading_out
        )
        
        conversation = self._create_prompt(
            task_description=task_description,
            labelled_rooms=labelled_rooms,
            current_room=current_room,
            room_dict=room_dict,
            rooms_with_frontier_within=rooms_with_frontier_within,
            rooms_with_frontier_leading_out=rooms_with_frontier_leading_out,
            rooms_with_closed_doors=rooms_with_closed_doors,
            close_objects=close_objects,
            nlp_history=nlp_history,
            graph=graph,
            room_graph=obs.get("room_graph"),
            room_distances=room_distances,
            target_name=target_name
        )
        
        # Check max LLM queries limit
        max_llm_queries = self.env.config.get("max_llm_queries", 0)
        if max_llm_queries and max_llm_queries > 0:
            current_queries = 0
            if hasattr(self.llm, 'get_episode_metrics'):
                current_queries = self.llm.get_episode_metrics().get('episode_llm_queries', 0)
            if current_queries >= max_llm_queries:
                print(f"Max LLM queries limit reached ({current_queries} >= {max_llm_queries})")
                self.env.episode_info["failure_reason"] = f"max_llm_queries limit ({max_llm_queries}) reached"
                return True, False, self.env.episode_info
        
        response, action, argument = self.send_query(conversation=conversation, mode='eval')
        conversation.add_message({"role": "assistant", "content": response})
        
        robot_pose_pre = np.concatenate(self.env.robots[0].get_position_orientation())

        try:
            subpolicy_success, done, self.last_env_feedback, _ = self.execute_action(
                action=action,
                argument=argument,
                task_desc=task_description,
                graph=graph,
                vor_graph=obs["separated_voronoi_graph"]
            )
        except:
            subpolicy_success = False
            done = False

        # Get new state to find new discoveries and new room
        new_obs = self.env.get_state(compute_scene_graph=True)
        try:
            _apply_room_classification(new_obs)
            new_room = self.room_classification.get(new_obs["robot_current_room"], "unknown")
        except:
            new_room = current_room
        
        # Get new discoveries
        new_discoveries = self._get_new_discoveries(new_obs.get("room_object_graph", nx.DiGraph()))
        
        # Record action for storyteller
        feedback_text = self.last_env_feedback.get("content", "") if isinstance(self.last_env_feedback, dict) else ""
        self._record_action_for_storyteller(
            action=action,
            argument=argument,
            success=subpolicy_success,
            feedback=feedback_text if not subpolicy_success else "",
            room_after=new_room,
            new_discoveries=new_discoveries
        )

        conversation.add_message(self.last_env_feedback)
        self.plot_conversation(conversation=conversation, action=action, argument=argument, ax=self.env.ax[0])
        
        robot_pose_post = np.concatenate(self.env.robots[0].get_position_orientation())

        # Retry loop
        num_retries = 0
        max_retries = 5
        while (not subpolicy_success) and np.all((robot_pose_post - robot_pose_pre) < 0.1) and (not done) and (num_retries < max_retries):
            obs = self.env.get_state(compute_scene_graph=True)
            try:
                _apply_room_classification(obs)
            except:
                break

            # Check max LLM queries limit before retry
            if max_llm_queries and max_llm_queries > 0:
                current_queries = 0
                if hasattr(self.llm, 'get_episode_metrics'):
                    current_queries = self.llm.get_episode_metrics().get('episode_llm_queries', 0)
                if current_queries >= max_llm_queries:
                    print(f"Max LLM queries limit reached during retry ({current_queries} >= {max_llm_queries})")
                    self.env.episode_info["failure_reason"] = f"max_llm_queries limit ({max_llm_queries}) reached"
                    done = True
                    break

            # Get last failure reason for prompts
            last_failure_reason = self.action_history[-1].feedback if self.action_history else None
            retry_prompt = self._get_retry_prompt(failure_reason=last_failure_reason)
            conversation.add_message({"role": "user", "content": retry_prompt})
            response, action, argument = self.send_query(conversation=conversation, mode='eval')
            conversation.add_message({"role": "assistant", "content": response})
            
            # Track room before retry action
            self._room_before_action = new_room
            
            try:
                subpolicy_success, done, self.last_env_feedback, _ = self.execute_action(
                    action=action,
                    argument=argument,
                    task_desc=task_description,
                    graph=graph,
                    vor_graph=obs["separated_voronoi_graph"]
                )
                
                # Record retry action for storyteller
                retry_new_obs = self.env.get_state(compute_scene_graph=True)
                try:
                    _apply_room_classification(retry_new_obs)
                    retry_new_room = self.room_classification.get(retry_new_obs["robot_current_room"], "unknown")
                except:
                    retry_new_room = new_room
                
                retry_feedback = self.last_env_feedback.get("content", "") if isinstance(self.last_env_feedback, dict) else ""
                self._record_action_for_storyteller(
                    action=action,
                    argument=argument,
                    success=subpolicy_success,
                    feedback=retry_feedback if not subpolicy_success else "",
                    room_after=retry_new_room,
                    new_discoveries=[]
                )
                new_room = retry_new_room
                
                conversation.add_message(self.last_env_feedback)
                self.plot_conversation(conversation=conversation, action=action, argument=argument, ax=self.env.ax[0])
            except:
                conversation.add_message({"role": "user", "content": self._get_format_error_prompt()})
            num_retries += 1
            
        if (num_retries == max_retries) and (not subpolicy_success) and (not done):
            done = True
            self.env.episode_info["failure_reason"] = "max retrials reached"
            
        self.env.episode_info["total_num_retrials"] = self.env.episode_info.get("total_num_retrials", 0) + num_retries
        self.env.episode_info["steps_with_retrial"] = self.env.episode_info.get("steps_with_retrial", 0) + (num_retries > 0)

        if self.evaluate_success():
            task_success = True
            done = True
        else:
            task_success = False

        if sum([response == r for r in self.prev_responses]) >= 3:
            print("WARNING: LLM response repeated. May be stuck.")
            if not self.env.episode_info.get("failure_reason"):
                self.env.episode_info["failure_reason"] = "llm stuck"
                self.env.episode_info["task_success"] = False
            done = True
            
        if len(self.prev_responses) > 6:
            del self.prev_responses[0]
        self.prev_responses.append(response)

        # Update metrics from both LLMs
        if hasattr(self.llm, 'get_episode_metrics'):
            llm_metrics = self.llm.get_episode_metrics()
            self.env.episode_info.update(llm_metrics)
        
        # Add storyteller metrics
        if self.enable_storyteller:
            storyteller_metrics = self.storyteller.get_episode_metrics()
            self.env.episode_info.update(storyteller_metrics)
            
            # Combined metrics
            total_input = llm_metrics.get('episode_input_tokens', 0) + storyteller_metrics.get('storyteller_input_tokens', 0)
            total_output = llm_metrics.get('episode_output_tokens', 0) + storyteller_metrics.get('storyteller_output_tokens', 0)
            self.env.episode_info['combined_input_tokens'] = total_input
            self.env.episode_info['combined_output_tokens'] = total_output
            self.env.episode_info['combined_total_tokens'] = total_input + total_output

        return done, task_success, self.env.episode_info
    
    def get_current_story(self) -> str:
        """
        Get the current exploration story.
        
        Returns:
            The current narrative summary from the storyteller
        """
        return self.storyteller.current_summary if self.enable_storyteller else ""

