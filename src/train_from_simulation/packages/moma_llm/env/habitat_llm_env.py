# Habitat-compatible LLM Environment for SmallPlan
# Replaces iGibson dependencies with Habitat-compatible interfaces

import re
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, NamedTuple, Literal, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from matplotlib.patches import FancyBboxPatch
from scipy.spatial import distance_matrix

from moma_llm.env.habitat_env import OurHabitatEnv, create_habitat_env
from moma_llm.env.prompts import SYSTEM_PROMPT, USER_PROMPT, RETRIAL_PROMPT, RETRIAL_PROMPT_FORMAT_ERROR
from moma_llm.llm.habitat_llm import (
    LLM_hugging, 
    Conversation, 
    inflect_engine,
    object_states
)
from moma_llm.topology.habitat_room_graph import get_closest_node
from moma_llm.utils.habitat_constants import FRONTIER_CLASSIFICATION, NODETYPE
from moma_llm.navigation.habitat_navigation import (
    drive_to_target_position,
    turn_full_circle,
    turn_to_target_point,
    plan_waypoints,
    PyAstarHelper
)

import gymnasium
import requests
import copy


DIST_MAPPING = OrderedDict({
    3.0: "very close",
    10.0: "near",
    20.0: "far",
    np.inf: "distant"
})


def distance_mapping(dist: float) -> str:
    """Map distance to human-readable description."""
    for k, v in DIST_MAPPING.items():
        if dist < k:
            return v
    return "distant"
    
    
def split_frontier_points(frontier_points: set) -> Tuple[List, List]:
    """Split frontier points by classification."""
    frontier_points_within = []
    frontier_points_leading_out = []
    
    for frontier_point in frontier_points:
        if frontier_point[1] == FRONTIER_CLASSIFICATION.WITHIN:
            frontier_points_within.append(frontier_point[0])
        elif frontier_point[1] == FRONTIER_CLASSIFICATION.LEADING_OUT:
            frontier_points_leading_out.append(frontier_point[0])
        else:
            raise ValueError(f"Unknown frontier point type {frontier_point[1]}")
            
    return frontier_points_within, frontier_points_leading_out


@dataclass
class ActionHistory:
    """Record of an action taken."""
    action: str
    object_name_graph: str
    position: tuple
    subtask_success: bool
    opendoors_roompos: Any = None
    orig_api_call: str = None


class HabitatHighLevelEnv(gymnasium.Wrapper):
    """High-level environment wrapper for Habitat."""
    
    def __init__(self, env: OurHabitatEnv, seed: int) -> None:
        super().__init__(env)
        self.seed = seed
        # Store mode from the initial environment for recreation
        self._mode = getattr(env, 'mode', 'headless')

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped env."""
        return getattr(self.env, name)
    
    @property
    def verbose(self) -> bool:
        """Check if verbose mode is enabled."""
        return self.env.config.get("verbose", False)
    
    def _debug_print(self, message: str) -> None:
        """Print debug message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def reset(self, config_file: str, scene_id: str, episode_num: int, 
              compute_scene_graph: bool = False) -> Dict:
        """Reset environment for new episode."""
        success = False
        i = 0
        
        while not success:
            if True:  # Always recreate to ensure clean state
                self.env.close()
                control_freq = self.env.config.get("control_freq", 10.0)
                self.env = create_habitat_env(
                    config_file=config_file,
                    scene_id=scene_id,
                    control_freq=control_freq,
                    seed=self.seed + 69 * episode_num + 999 * i,
                    mode=self._mode
                )
                # Attach task to the new environment
                from moma_llm.tasks.habitat_object_search_task import HabitatObjectSearchTask
                self.env.task = HabitatObjectSearchTask(self.env)
                
            _obs = self.env.reset()
            
            # Reset task to sample initial pose and target
            if self.env.task:
                self.env.task.reset_scene(self.env)
                self.env.task.reset_agent(self.env)
            self.env.episode_info.update({
                "steps_with_retrial": 0,
                "total_num_retrials": 0,
                "target_category": self.env.task.target_category if self.env.task else "",
            })

            for _ in range(10):
                _obs = self.env.step(np.zeros(self.env.action_space.shape))

            _obs, success = turn_full_circle(self.env)
            
            if success and self.env.config.get("reject_onestep_episodes", False):
                if self.env.task and self.env.task.evaluate_success(self.env):
                    success = False
            i += 1
            if i > 10:
                break
            
        return self.env.get_state(compute_scene_graph=compute_scene_graph)
    
    def navigate_to_point(self, target_pos_world: np.ndarray, 
                          success_thres_dist: float, 
                          face_target: bool, 
                          early_termination_dist: float) -> bool:
        """Navigate to target position."""
        self.env.plot_object_position(target_pos_world, color="lime", marker="*")
        # Get verbose flag from config (default to False)
        verbose = self.env.config.get("verbose", False)
        success = drive_to_target_position(
            env=self.env,
            target_pos_world=target_pos_world,
            inflation_radius_m=self.env.config.get("navigation_inflation_radius", 0.1),
            success_thres_dist=success_thres_dist,
            face_target=face_target,
            early_termination_dist=early_termination_dist,
            debug=verbose
        )
        return success

    def _find_closest_point(self, points: List, euclidean_heuristic: bool = True):
        """Find closest navigable point."""
        def _to_navpoint(point):
            if isinstance(point, dict):
                p = self.env.slam.voxel2world(point["closest_vor_node"][:2])
                euclidean_dist = np.linalg.norm(np.array(point["pos"][:2]) - p)
            else:
                p = point
                euclidean_dist = 0
            return p, euclidean_dist
        
        nav_points = []
        costs = []
        paths = []
        
        for point in points:
            p, euclidean_dist = _to_navpoint(point)
            path, cost = plan_waypoints(
                env=self.env,
                target_pos_world=p,
                inflation_radius_m=self.env.config.get("navigation_inflation_radius", 0.1),
                add_wall_avoidance_cost=False
            )
            nav_points.append(p)
            
            if not isinstance(point, dict):
                last_cells = int(0.25 / self.env.slam.voxel_size)
                mask = (cost[-last_cells:] == PyAstarHelper.UNEXPLORED_COST + 1)
            else:
                last_cells = int(np.ceil(
                    self.env.config.get("navigation_inflation_radius", 0.1) / self.env.slam.voxel_size
                ))
                mask = (cost[-last_cells:] >= PyAstarHelper.OCCUPIED_COST)
            cost[-last_cells:][mask] = 1
            
            total_cost = cost.sum()
            if euclidean_heuristic:
                total_cost += euclidean_dist
            costs.append(total_cost)
            paths.append(path)
            
        idx = np.argmin(costs)
        return idx, nav_points[idx], costs, paths

    def open_object(self, obj, nav_point) -> Tuple[bool, str]:
        """
        Open an articulated object (door/container).
        Simplified for Habitat as most objects are static.
        """
        feedback = ""
        
        # DEBUG: Log open_object attempt
        self._debug_print(f"DEBUG open_object: Attempting to open '{obj.name}'")
        self._debug_print(f"DEBUG open_object:   Nav target: [{nav_point[0]:.3f}, {nav_point[1]:.3f}]")
        
        # Navigate to object
        subtask_success_nav = self.navigate_to_point(
            np.array(nav_point), 
            success_thres_dist=1.5, 
            face_target=False, 
            early_termination_dist=0.5
        )
        
        if not subtask_success_nav:
            feedback = f"Navigation to {obj.name} failed"
            self._debug_print(f"DEBUG open_object: FAILED - Navigation to object failed")
            return False, feedback
            
        if obj.name in self.env.opened_doors:
            feedback = f"Door {obj.name} already open."
            self._debug_print(f"DEBUG open_object: Door already open")
            return True, feedback
        
        # In Habitat, most objects are static, so we simulate opening
        self.env.opened_doors.append(obj.name)
        self.env.episode_info["magic_open_actions"] = \
            self.env.episode_info.get("magic_open_actions", 0) + 1
            
        self._debug_print(f"DEBUG open_object: SUCCESS - Object {obj.name} opened (simulated).")
        return True, feedback

    def evaluate_success(self) -> bool:
        """Evaluate task success."""
        if self.env.task is None:
            return False
        task_success = self.env.task.evaluate_success(self.env)
        if (not task_success) and (not self.env.episode_info.get("failure_reason", None)):
            self.env.episode_info["failure_reason"] = "wrong termination by llm"
        return task_success
    
    def visualize(self, state: Optional[Dict] = None):
        """Visualize current state."""
        self.env.visualize(state=state)


class HabitatLLMEnv(HabitatHighLevelEnv):
    """LLM-based high-level environment for Habitat."""
    
    def __init__(self, env: OurHabitatEnv, llm: LLM_hugging, seed: int) -> None:
        super().__init__(env, seed=seed)

        self.possible_actions = {
            "navigate": ("room_name, object_name", "navigate to this object in this room."),
            "go_to_and_open": ("room_name, object_name", "go to this articulated object, door or container and open it."),
            "explore": ("room_name", "explore the unknown space near one of the rooms that is not fully explored yet."),
            "done": ("", "call when the task is completed or if you are unable to take any further actions.")
        }
        
        if not self.env.config.get("consider_open_actions", True):
            self.possible_actions.pop("go_to_and_open", None)
        
        self.llm = llm
        self.room_classification = dict()
        self.prev_responses = [""]
        self.action_history = []
        self.last_env_feedback = {"role": "user", "content": ""}

    def reset(self, config_file: str, scene_id: str, episode_num: int) -> Dict:
        """Reset for new episode."""
        self.prev_responses = [""]
        self.action_history = []
        self.last_env_feedback = {"role": "user", "content": ""}
        if hasattr(self.llm, 'reset_episode_metrics'):
            self.llm.reset_episode_metrics()
        return super().reset(config_file, scene_id, episode_num, compute_scene_graph=True)

    def classify_rooms(self, obs: Dict):
        """Classify rooms using LLM."""
        room_classification = self.llm.classify_rooms(obs)
        
        # Handle duplicate room names
        tot_counter = Counter(room_classification.values())
        counter = defaultdict(int)
        for k, v in room_classification.items():
            if tot_counter[v] > 1:
                room_classification[k] = v + f"-{counter[v] + 1}"
                counter[v] += 1
        self.room_classification = room_classification
        self._label_rooms_on_map(obs)

    def _label_rooms_on_map(self, obs: Dict, ax_idx: int = 1):
        """Label rooms on visualization."""
        for txt in self.env.ax[ax_idx].texts:
            Artist.remove(txt)

        labeled_rooms_not_found = []
        for room, labelled_room in self.room_classification.items():
            if room in obs["room_object_graph"].nodes:
                room_node = obs["room_object_graph"].nodes[room]
                room_center = room_node.get("pos_map", [0, 0])
                self.env.ax[ax_idx].text(
                    room_center[1], room_center[0], labelled_room,
                    fontsize=8, color="m", horizontalalignment='center'
                )
            else:
                labeled_rooms_not_found.append((room, labelled_room))
                
        if labeled_rooms_not_found:
            print(f"Rooms not found in graph: {labeled_rooms_not_found}")

    def parse_llm_action(self, response: str, mode: Literal['train', 'eval']) -> Tuple[str, str]:
        """Parse action from LLM response."""
        action = None
        command_found = False
        
        for line in response.split("\n"):
            command_found = command_found or line.lower().replace("'", "").replace('"', "").replace("\\", "").strip().startswith("command")
            if not command_found:
                continue
            try:
                action, argument = re.search(r"(\w+)\((.*)\)", line).groups()
                argument = argument.replace("'", "").replace('"', "").replace("\\", "")
                if action:
                    break
            except:
                continue
            
        if not action:
            print("Could not parse command. Calling done().")
            if mode == 'train':
                action = "format error"
            elif mode == 'eval':
                action = "done"
            argument = ""
            self.env.episode_info["failure_reason"] = "Unable to prompt LLM command."
            
        return action, argument

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
                       *args,
                       **kwargs) -> Conversation:
        """Create prompt for LLM."""
        tool_descriptions = ""
        for i, (action, description) in enumerate(self.possible_actions.items()):
            tool_descriptions += f"{i+1}. {action}({description[0]}): {description[1]}\n"

        system_prompt = SYSTEM_PROMPT.format(
            TASK_DESCRIPTION=task_description,
            TOOL_DESCRIPTIONS=tool_descriptions
        )

        list_nearby_objects = f"[{', '.join(sorted(close_objects))}]"

        list_found_rooms_and_objects = ""
        for room in sorted(labelled_rooms):
            objects = room_dict.get(room, []) + (
                ["unexplored area"] if room in [r[0] for r in rooms_with_frontier_within] else []
            )
            list_found_rooms_and_objects += f"- {room}: [{', '.join(objects)}].\n"

        list_previous_actions = ""
        if len(nlp_history):
            list_previous_actions = f"Your {len(nlp_history)} previous actions were: {', '.join(nlp_history)}."

        rooms_with_frontier_descr = f"[{', '.join([f'{room} ({distance_mapping(dist)})' for room, dist in rooms_with_frontier_leading_out])}]"
        
        rooms_with_closed_doors_descr = ""
        if len(rooms_with_closed_doors):
            rooms_with_closed_doors_descr = f"These rooms contain closed doors that might open up new space: [{', '.join([f'{room} ({distance_mapping(dist)})' for room, dist in rooms_with_closed_doors])}]."

        user_prompt = USER_PROMPT.format(
            CURRENT_ROOM=current_room,
            LIST_NEARBY_OBJECTS=list_nearby_objects,
            LIST_FOUND_ROOMS_AND_OBJECTS=list_found_rooms_and_objects,
            LIST_PREVIOUS_ACTIONS=list_previous_actions if len(nlp_history) else "",
            ROOMS_WITH_FRONTIER_DESCRIPTION=rooms_with_frontier_descr,
            ROOMS_WITH_CLOSED_DOORS_DESCRIPTION=rooms_with_closed_doors_descr,
        )

        conversation = Conversation(messages=[
            self.last_env_feedback,
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        return conversation

    def _get_close_objects(self, graph: nx.DiGraph, current_room: str, 
                           closeness_thresh: float) -> set:
        """Get objects close to robot."""
        # Check if current_room exists in graph
        if current_room not in graph.nodes:
            return set()
            
        current_room_objects = list(graph.successors(current_room))
        if not current_room_objects:
            return set()
            
        graph_node_pos = np.stack([
            graph.nodes.get(n, {}).get("pos_map", [0, 0])[:2] 
            for n in current_room_objects
        ])
        node_coords_world = self.env.slam.voxel2world(graph_node_pos)
        dist_matrix = distance_matrix(
            self.env.robots[0].get_position()[np.newaxis, :2], 
            node_coords_world
        )
        close_objects = np.array(current_room_objects)[
            np.squeeze(dist_matrix < closeness_thresh, 0)
        ]
        return set([self.llm.to_human_readable_object_name(o) for o in close_objects])

    def _match_action_history(self, graph: nx.DiGraph, 
                              separated_voronoi_graph: nx.Graph) -> List[str]:
        """Match action history to current graph state."""
        max_history_length = 5
        
        def _match_room_pos(position):
            if position is None:
                return None
            closest_nodes, _ = get_closest_node(
                np.array([position]), separated_voronoi_graph, self.env.slam
            )
            room_name = separated_voronoi_graph.nodes[tuple(closest_nodes[0])].get("room_id", 0)
            return room_name
        
        nlp_history = []
        last_k = len(self.action_history) - max_history_length
        
        for i, h in enumerate(self.action_history):
            room_name, object_name, invalid = None, None, False
            
            if h.object_name_graph is not None:
                object_name = self.llm.to_human_readable_object_name(h.object_name_graph)
                if (h.opendoors_roompos is not None) and (h.object_name_graph in self.env.opened_doors):
                    room_name = _match_room_pos(h.opendoors_roompos)
                elif h.object_name_graph in graph.nodes:
                    predecessors = list(graph.predecessors(h.object_name_graph))
                    room_name = predecessors[0] if predecessors else _match_room_pos(h.position)
                else:
                    room_name = _match_room_pos(h.position)
            elif h.position is not None:
                room_name = _match_room_pos(h.position)
            
            nlp = None
            if h.action == "done":
                nlp = "done()"
            elif (h.action == "explore") and (room_name is not None):
                nlp = f"explore({room_name})"
            elif (object_name is not None) and (room_name is not None):
                nlp = f"{h.action}({room_name}, {object_name})"
            else:
                nlp = f"{h.orig_api_call} - invalid argument"
                invalid = True
            
            if not invalid:
                nlp += f" - {'success' if h.subtask_success else 'failure'}"
            
            if i >= last_k or invalid:
                nlp_history.append(nlp)
                
        return nlp_history
    
    def send_query(self, conversation: Conversation, 
                   mode: Literal['train', 'eval']) -> Tuple[str, str, str]:
        """Send query to LLM and parse response."""
        response = self.llm.send_query(conversation=conversation)
        action, argument = self.parse_llm_action(response, mode)
        return response, action, argument

    def compute_reward(self, env_feedback: Dict, obs: Dict, new_obs: Dict) -> float:
        """Compute reward for RL training."""
        alpha_subtask = 0.3
        beta_exploration = 0.1
        gamma_distance_travel = 0.3
        
        if env_feedback.get("success") == False:
            return -alpha_subtask
        
        if self.evaluate_success():
            return 5
        
        reward_subtask = alpha_subtask
        reward_explore = beta_exploration * (
            len(new_obs.get("separated_voronoi_graph", [])) - 
            len(obs.get("separated_voronoi_graph", []))
        ) / 20

        old_distance_travel = np.linalg.norm(
            np.diff(np.stack(obs.get("robot_traj", [np.eye(4)]))[:, :2, 3], axis=0), axis=-1
        ).sum() if len(obs.get("robot_traj", [])) > 1 else 0
        new_distance_travel = np.linalg.norm(
            np.diff(np.stack(new_obs.get("robot_traj", [np.eye(4)]))[:, :2, 3], axis=0), axis=-1
        ).sum() if len(new_obs.get("robot_traj", [])) > 1 else 0
        reward_distance = -gamma_distance_travel * (new_distance_travel - old_distance_travel) / 30

        return reward_subtask + reward_explore + reward_distance

    def _train_by_strategy(self, reward: float, conversation: Conversation, 
                           strategy: Literal["RL-SFT", "SFT", "SFT-RL"]):
        """Apply training strategy."""
        if strategy == "RL-SFT":
            self.llm.train_PPO(reward=reward)
            self.llm.train_SFT(conversation)
        elif strategy == "SFT":
            self.llm.train_SFT(conversation)
        elif strategy == "SFT-RL":
            self.llm.train_SFT(conversation)
            self.llm.train_PPO(reward=reward)
        else:
            raise ValueError(f"Unknown training strategy {strategy}")

    @staticmethod
    def _parse_room_argument(room_name: str) -> str:
        """Parse room name from argument."""
        room_name = room_name[:-3].replace("_", " ").replace("-", " ") + room_name[-3:].replace("_", "-")
        for v in DIST_MAPPING.values():
            room_name = room_name.replace(f"({v})", "")
        return room_name

    @staticmethod
    def _parse_room_object_argument(argument: str) -> Tuple[str, str]:
        """Parse room and object from argument."""
        room_name, object_name = argument.split(", ")
        singular = inflect_engine.singular_noun(object_name)
        if singular:
            object_name = singular
        return HabitatLLMEnv._parse_room_argument(room_name), object_name

    def execute_action(self, action: str, argument: str, task_desc: str,
                       graph: nx.DiGraph, vor_graph: Optional[nx.Graph] = None) -> Tuple[bool, bool, Dict, Dict]:
        """Execute high-level action."""
        print(f"========== DEBUG execute_action ==========")
        print(f"Executing {action}({argument})")
        done = False
        feedback = ""
        
        if action == "navigate":
            assert vor_graph is not None
            try:
                room_name, object_name = self._parse_room_object_argument(argument)
                possible_nodes = [
                    n for n in graph.successors(room_name) 
                    if self.llm.human_to_graph_name(object_name) in n
                ]
            except:
                room_name, object_name = "None", "None"
                possible_nodes = []
                
            if len(possible_nodes) == 0:
                subtask_success = False
                feedback = f"Navigation failed: no {object_name} in {room_name}"
                history = ActionHistory(
                    action, 
                    object_name_graph=self.llm.human_to_graph_name(object_name),
                    position=graph.nodes.get(room_name, {}).get("pos_map"),
                    subtask_success=subtask_success
                )
            else:
                idx, point, _costs, _paths = self._find_closest_point(
                    [graph.nodes.get(n) for n in possible_nodes]
                )
                subtask_success = self.navigate_to_point(
                    np.array(point),
                    success_thres_dist=1.5,
                    face_target=True,
                    early_termination_dist=0.3
                )
                if not subtask_success:
                    feedback = "Navigation to object failed"
                history = ActionHistory(
                    action,
                    object_name_graph=possible_nodes[idx],
                    position=point,
                    subtask_success=subtask_success
                )
                
        elif action == "go_to_and_open":
            assert vor_graph is not None
            subtask_success, (obj, room_pos), feedback = self._open_graph_node(
                graph, vor_graph, argument=argument
            )
            history = ActionHistory(
                action,
                object_name_graph=obj.name if obj else None,
                position=obj.get_position()[:2] if obj else room_pos,
                subtask_success=subtask_success,
                opendoors_roompos=room_pos
            )
            
        elif action == "explore":
            self._debug_print(f"DEBUG explore: Processing explore({argument})")
            try:
                room_name = self._parse_room_argument(argument)
                self._debug_print(f"DEBUG explore:   Parsed room_name='{room_name}'")
                raw_frontier_points = graph.nodes.get(room_name, {}).get("frontier_points", set())
                self._debug_print(f"DEBUG explore:   Raw frontier points count: {len(raw_frontier_points)}")
                frontier_points_within, frontier_points_leading_out = split_frontier_points(raw_frontier_points)
                self._debug_print(f"DEBUG explore:   Frontier within: {len(frontier_points_within)}, leading out: {len(frontier_points_leading_out)}")
                frontier_points = frontier_points_leading_out if frontier_points_leading_out else frontier_points_within
            except Exception as e:
                room_name = "None"
                frontier_points = []
                self._debug_print(f"DEBUG explore:   ERROR parsing: {e}")
                
            if len(frontier_points) == 0:
                subtask_success = False
                feedback = f"Exploration failed: no frontier points in {room_name}"
                self._debug_print(f"DEBUG explore: FAILED - {feedback}")
                # Show what rooms have frontiers
                all_rooms = list(graph.successors("root")) if "root" in graph else []
                self._debug_print(f"DEBUG explore:   Available rooms: {all_rooms}")
                for r in all_rooms:
                    fp = graph.nodes.get(r, {}).get("frontier_points", set())
                    self._debug_print(f"DEBUG explore:   Room '{r}' has {len(fp)} frontier points")
                history = ActionHistory(
                    action,
                    object_name_graph=None,
                    position=graph.nodes.get(room_name, {}).get("pos_map"),
                    subtask_success=subtask_success
                )
            else:
                _idx, closest_frontier, _costs, _paths = self._find_closest_point(list(frontier_points))
                self._debug_print(f"DEBUG explore:   Navigating to frontier at [{closest_frontier[0]:.3f}, {closest_frontier[1]:.3f}]")
                subtask_success = self.navigate_to_point(
                    closest_frontier, 
                    success_thres_dist=0.5, 
                    face_target=True, 
                    early_termination_dist=0.5
                )
                if not subtask_success:
                    feedback = "Navigation to frontier failed"
                    self._debug_print(f"DEBUG explore: FAILED - Navigation to frontier failed")
                else:
                    self._debug_print(f"DEBUG explore: SUCCESS - Reached frontier")
                history = ActionHistory(
                    action,
                    object_name_graph=None,
                    position=closest_frontier,
                    subtask_success=subtask_success
                )
                
        elif action == "done":
            subtask_success = True
            done = True
            history = ActionHistory(action, object_name_graph=None, position=None, subtask_success=subtask_success)
            
        else:
            raise ValueError(f"Unknown action {action}")

        history.orig_api_call = f"{action}({argument})"
        self.action_history.append(history)

        print(f"Subtask success: {subtask_success}")
        
        if self.env.episode_info.get("failure_reason", "") == "max_high_level_steps timeout":
            done = True

        # Check task success
        if self.env.task:
            task_success = self.env.task.evaluate_success(env=self.env)
            if task_success:
                if self.env.config.get("ground_truth_done_decision", False):
                    done = True
                elif self.env.episode_info.get("num_low_level_steps_gtDone") is None:
                    self.env.episode_info["num_low_level_steps_gtDone"] = self.env.episode_info.get("num_low_level_steps", 0)
                    self.env.episode_info["num_high_level_steps_gtDone"] = self.env.episode_info.get("num_high_level_steps", 0)
                    self.env.episode_info["magic_open_actions_gtDone"] = self.env.episode_info.get("magic_open_actions", 0)

        feedback_msg = {
            "role": "user",
            "content": f"Feedback: {'; '.join([f'{action}({argument}) success: {subtask_success}', feedback])}"
        }
        engine_feedback = {"action": [action, argument], "success": subtask_success, "feedback": feedback}
        
        return subtask_success, done, feedback_msg, engine_feedback

    def _open_graph_node(self, graph: nx.DiGraph, vor_graph: nx.Graph, 
                         argument: str) -> Tuple[bool, Tuple, str]:
        """Open a node in the graph (door/container)."""
        self._debug_print(f"DEBUG _open_graph_node: Processing argument '{argument}'")
        
        try:
            room_name, object_name = self._parse_room_object_argument(argument)
            self._debug_print(f"DEBUG _open_graph_node:   Parsed room='{room_name}', object='{object_name}'")
            
            if "door" in object_name:
                possible_nodes = list(graph.nodes.get(room_name, {}).get("closed_doors", []))
                self._debug_print(f"DEBUG _open_graph_node:   Looking for closed doors in '{room_name}'")
                self._debug_print(f"DEBUG _open_graph_node:   Found {len(possible_nodes)} closed doors: {possible_nodes}")
            else:
                possible_nodes = []
                for n in graph.successors(room_name):
                    node_data = graph.nodes[n]
                    is_closed = not node_data.get('states', {}).get(object_states.Open, True)
                    graph_name = self.llm.human_to_graph_name(object_name)
                    if ((graph_name in n) or (graph_name.strip('s') in n)) and is_closed:
                        possible_nodes.append(n)
                self._debug_print(f"DEBUG _open_graph_node:   Looking for closed '{object_name}' in '{room_name}'")
                self._debug_print(f"DEBUG _open_graph_node:   Found {len(possible_nodes)} matching nodes: {possible_nodes}")
        except Exception as e:
            room_name, object_name = "None", "None"
            possible_nodes = []
            self._debug_print(f"DEBUG _open_graph_node:   ERROR parsing argument: {e}")

        if not possible_nodes:
            feedback = f"Object opening failed: no closed {object_name} in {room_name}"
            self._debug_print(f"DEBUG _open_graph_node: FAILED - {feedback}")
            # Debug: show what's available in the room
            room_data = graph.nodes.get(room_name, {})
            self._debug_print(f"DEBUG _open_graph_node:   Room data keys: {list(room_data.keys()) if room_data else 'room not found'}")
            if room_data:
                self._debug_print(f"DEBUG _open_graph_node:   closed_doors in room: {room_data.get('closed_doors', [])}")
                successors = list(graph.successors(room_name)) if room_name in graph else []
                self._debug_print(f"DEBUG _open_graph_node:   Room successors: {successors[:10]}{'...' if len(successors) > 10 else ''}")
            return False, (None, graph.nodes.get(room_name, {}).get("pos_map")), feedback

        # Get closest node
        possible_node_data = [graph.nodes.get(n) for n in possible_nodes if n in graph.nodes]
        if not possible_node_data:
            feedback = f"Could not find node data for {object_name}"
            self._debug_print(f"DEBUG _open_graph_node: FAILED - {feedback}")
            return False, (None, None), feedback
            
        idx, point, _costs, _paths = self._find_closest_point(possible_node_data)
        self._debug_print(f"DEBUG _open_graph_node:   Closest node: '{possible_nodes[idx]}' at [{point[0]:.3f}, {point[1]:.3f}]")
        
        obj = self.env.scene.objects_by_name.get(possible_nodes[idx])
        if obj is None:
            self._debug_print(f"DEBUG _open_graph_node:   Object not in scene, creating mock object")
            # Create a mock object for compatibility
            class MockObj:
                def __init__(self, name):
                    self.name = name
                def get_position(self):
                    return np.array([0, 0, 0])
            obj = MockObj(possible_nodes[idx])
            
        open_success, feedback = self.open_object(obj, point)
        return open_success, (obj, graph.nodes.get(room_name, {}).get("pos_map")), feedback

    def take_action(self, obs: Dict, task_description: str,
                    strategy: Literal["RL-SFT", "SFT", "SFT-RL"]) -> Tuple[bool, bool, Dict]:
        """Take action based on observation (training mode)."""
        def _apply_room_classification(obs):
            obs["room_object_graph"] = nx.relabel_nodes(obs["room_object_graph"], self.room_classification)
            for n, d in obs["separated_voronoi_graph"].nodes(data=True):
                d["room_id"] = self.room_classification.get(NODETYPE.roomname(d["room_id"]), f"room-{d['room_id']}")

        self.classify_rooms(obs)
        _apply_room_classification(obs)
        
        graph = obs["room_object_graph"]
        labelled_rooms = list(graph.successors("root"))
        
        room_dict = self.llm.create_room_object_dict(
            graph,
            open_door_inclusion="ignore",
            room_classification=self.room_classification
        )
        current_room = self.room_classification.get(obs["robot_current_room"], "unknown")

        def _get_closest_dist(points):
            if not points:
                return float('inf')
            idx, _, _costs, paths = self._find_closest_point(points)
            return self.env.slam.voxel_size * len(paths[idx])

        # Get frontier and door information
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
            room_distances=room_distances
        )
        
        response, action, argument = self.send_query(conversation=conversation, mode='train')
        conversation.add_message({"role": "assistant", "content": response})
        
        robot_pose_pre = np.concatenate(self.env.robots[0].get_position_orientation())
        
        try:
            subpolicy_success, done, self.last_env_feedback, engine_feedback = self.execute_action(
                action=action,
                argument=argument,
                task_desc=task_description,
                graph=graph,
                vor_graph=obs["separated_voronoi_graph"]
            )
            new_obs = self.env.get_state(compute_scene_graph=True)
            reward = self.compute_reward(engine_feedback, obs, new_obs)
            self._train_by_strategy(reward=reward, conversation=conversation, strategy=strategy)
        except Exception as e:
            print(f"Action execution failed: {e}")
            subpolicy_success = False
            done = False

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

            num_retries += 1
            conversation.add_message({"role": "user", "content": RETRIAL_PROMPT})
            response, action, argument = self.send_query(conversation=conversation, mode='train')
            conversation.add_message({"role": "assistant", "content": response})
            
            try:
                subpolicy_success, done, self.last_env_feedback, engine_feedback = self.execute_action(
                    action=action,
                    argument=argument,
                    task_desc=task_description,
                    graph=graph,
                    vor_graph=obs["separated_voronoi_graph"]
                )
                new_obs = self.env.get_state(compute_scene_graph=True)
                reward = self.compute_reward(engine_feedback, obs, new_obs)
                self._train_by_strategy(reward=reward, conversation=conversation, strategy=strategy)
                conversation.add_message(self.last_env_feedback)
                self.plot_conversation(conversation=conversation, action=action, argument=argument, ax=self.env.ax[0])
            except:
                self._train_by_strategy(reward=-0.1, conversation=conversation, strategy=strategy)
                conversation.add_message({"role": "user", "content": RETRIAL_PROMPT_FORMAT_ERROR})
                continue

        if (num_retries == max_retries) and (not subpolicy_success) and (not done):
            done = True
            self.env.episode_info["failure_reason"] = "max retrials reached"
            
        self.env.episode_info["total_num_retrials"] = self.env.episode_info.get("total_num_retrials", 0) + num_retries
        self.env.episode_info["steps_with_retrial"] = self.env.episode_info.get("steps_with_retrial", 0) + (num_retries > 0)

        if done:
            task_success = self.evaluate_success()
        else:
            task_success = False

        # Check for stuck LLM
        if sum([response == r for r in self.prev_responses]) >= 3:
            print("WARNING: LLM response repeated. May be stuck.")
            if not self.env.episode_info.get("failure_reason"):
                self.env.episode_info["failure_reason"] = "llm stuck"
                self.env.episode_info["task_success"] = False
            done = True
            
        if len(self.prev_responses) > 6:
            del self.prev_responses[0]
        self.prev_responses.append(response)

        return done, task_success, self.env.episode_info

    def take_action_inference(self, obs: Dict, task_description: str) -> Tuple[bool, bool, Dict]:
        """Take action based on observation (inference mode)."""
        def _apply_room_classification(obs):
            obs["room_object_graph"] = nx.relabel_nodes(obs["room_object_graph"], self.room_classification)
            for n, d in obs["separated_voronoi_graph"].nodes(data=True):
                d["room_id"] = self.room_classification.get(NODETYPE.roomname(d["room_id"]), f"room-{d['room_id']}")

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
            room_distances=room_distances
        )
        
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

            conversation.add_message({"role": "user", "content": RETRIAL_PROMPT})
            response, action, argument = self.send_query(conversation=conversation, mode='eval')
            conversation.add_message({"role": "assistant", "content": response})
            
            try:
                subpolicy_success, done, self.last_env_feedback, _ = self.execute_action(
                    action=action,
                    argument=argument,
                    task_desc=task_description,
                    graph=graph,
                    vor_graph=obs["separated_voronoi_graph"]
                )
                conversation.add_message(self.last_env_feedback)
                self.plot_conversation(conversation=conversation, action=action, argument=argument, ax=self.env.ax[0])
            except:
                conversation.add_message({"role": "user", "content": RETRIAL_PROMPT_FORMAT_ERROR})
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

        if hasattr(self.llm, 'get_episode_metrics'):
            llm_metrics = self.llm.get_episode_metrics()
            self.env.episode_info.update(llm_metrics)

        return done, task_success, self.env.episode_info

    def plot_conversation(self, conversation: Conversation, action: str, argument: str,
                          ax: plt.Axes, font_size: float = 6.5, add_fig_title: bool = True):
        """Plot conversation on visualization."""
        if add_fig_title:
            self.env.f.suptitle(f"{self.env.f._suptitle.get_text()}, {action}({argument})")

        ax.clear()
        ax.set_axis_off()
        colors = {"system": "wheat", "user": "orange", "assistant": "orangered", "env": "green"}
        
        h = 0.05
        for m in conversation.messages_including_env:
            orig_txt = f"{m['role']}: {m['content']}"
            txt = "".join([s for s in orig_txt.splitlines(True) if s.strip("\r").strip("\n")])
            txt = txt.strip("\n").strip("\r")
            props = dict(boxstyle='round', facecolor="white", alpha=0.0, edgecolor="white")
            t = ax.text(
                -0.05, 0.975 - h, txt,
                transform=ax.transAxes, fontsize=font_size,
                verticalalignment='top', bbox=props, wrap=True
            )
            t._get_wrap_line_width = lambda: 1.05 * ax.get_window_extent().width
            try:
                self.env.f.canvas.draw()
            except:
                pass
            last_h = t.get_bbox_patch().get_height() / ax.get_window_extent().height
            h += last_h + 0.025
            
            p = FancyBboxPatch(
                xy=[-0.05, 1.0 - h], width=1.05, height=last_h,
                transform=ax.transAxes, alpha=0.5,
                boxstyle="round,pad=0.01",
                facecolor=colors.get(m["role"], "gray"),
                edgecolor="k", clip_on=False
            )
            ax.add_patch(p)
            
        if (h > 1.025) and (font_size > 4.5):
            self.plot_conversation(
                conversation=conversation, action=action, argument=argument,
                ax=ax, font_size=font_size - 0.5, add_fig_title=False
            )

    def visualize(self, state: Optional[Dict] = None):
        """Visualize state with room labels."""
        super().visualize(state=state)
        if state:
            self._label_rooms_on_map(state, ax_idx=1)

