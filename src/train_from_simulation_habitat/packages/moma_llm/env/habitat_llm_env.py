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
# Import both prompt versions for switchable usage
from moma_llm.env import prompts as prompts_v1
from moma_llm.env import prompts_v2
# Default imports (will be overridden based on config)
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
    feedback: str = None  # Store failure reasons for v2 prompts


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
        """
        Navigate to target position.
        
        Args:
            target_pos_world: Target position in world coordinates (2D: X, Z in Habitat's Y-up system)
            success_thres_dist: Distance threshold for success
            face_target: Whether to face the target at the end
            early_termination_dist: Distance for early termination
            
        Returns:
            Whether navigation was successful
        """
        self.env.plot_object_position(target_pos_world, color="lime", marker="*")
        
        # Ensure target is 2D
        target_2d = np.array(target_pos_world[:2]) if len(target_pos_world) > 2 else target_pos_world
        
        # Get verbose flag from config
        verbose = self.env.config.get("verbose", False)
        
        success = drive_to_target_position(
            env=self.env,
            target_pos_world=target_2d,
            inflation_radius_m=self.env.config.get("navigation_inflation_radius", 0.4),
            success_thres_dist=success_thres_dist,
            face_target=face_target,
            early_termination_dist=early_termination_dist,
            debug=verbose  # Only enable debug if verbose mode is on
        )
        
        # Report final result
        final_pos = self.env.robots[0].get_position()
        final_pos_2d = np.array([final_pos[0], final_pos[2]])  # X, Z for 2D
        dist = np.linalg.norm(final_pos_2d - target_2d)
        
        if not success:
            print(f"Navigation failed: final dist={dist:.2f}m (threshold={success_thres_dist:.2f}m)")
        
        return success

    def _find_closest_point(self, points: List, euclidean_heuristic: bool = True):
        """Find closest navigable point. Optimized to first filter by Euclidean distance."""
        def _to_navpoint(point):
            if isinstance(point, dict):
                p = self.env.slam.voxel2world(point["closest_vor_node"][:2])
                euclidean_dist = np.linalg.norm(np.array(point["pos"][:2]) - p)
            else:
                p = point
                euclidean_dist = 0
            return p, euclidean_dist
        
        # Optimization: If many points, first filter by Euclidean distance
        # to avoid expensive path planning for all points
        robot_pos = self.env.robots[0].get_position()[:2]
        max_candidates = 3  # Only plan paths to the 3 closest points by Euclidean distance
        
        if len(points) > max_candidates:
            # Calculate Euclidean distances to all points
            euclidean_dists = []
            for point in points:
                p, _ = _to_navpoint(point)
                dist = np.linalg.norm(np.array(p) - robot_pos)
                euclidean_dists.append(dist)
            
            # Get indices of closest points
            sorted_indices = np.argsort(euclidean_dists)[:max_candidates]
            candidate_points = [points[i] for i in sorted_indices]
        else:
            candidate_points = points
        
        nav_points = []
        costs = []
        paths = []
        
        for point in candidate_points:
            p, euclidean_dist = _to_navpoint(point)
            path, cost = plan_waypoints(
                env=self.env,
                target_pos_world=p,
                inflation_radius_m=self.env.config.get("navigation_inflation_radius", 0.4),
                add_wall_avoidance_cost=False
            )
            nav_points.append(p)
            
            if not isinstance(point, dict):
                last_cells = int(0.25 / self.env.slam.voxel_size)
                mask = (cost[-last_cells:] == PyAstarHelper.UNEXPLORED_COST + 1)
            else:
                last_cells = int(np.ceil(
                    self.env.config.get("navigation_inflation_radius", 0.4) / self.env.slam.voxel_size
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
        
        Follows iGibson's open_object logic:
        1. First navigation to nearby point (success_thres_dist=1.5)
        2. For doors: navigate close enough to interact (success_thres_dist=0.5)
        3. Simulate opening the object
        4. For doors: optionally navigate through door frame
        
        Note: Habitat objects are mostly static, so opening is simulated.
        """
        feedback = ""
        is_door = "door" in obj.name.lower()
        
        self._debug_print(f"open_object: Attempting to open '{obj.name}'")
        
        # Check if door is already open
        if obj.name in self.env.opened_doors:
            feedback = f"Door {obj.name} already open."
            return True, feedback
        
        # Step 1: First navigation to get near the object
        # Use same parameters as iGibson: success_thres_dist=1.5, face_target=False, early_termination_dist=0.5
        subtask_success_nav = self.navigate_to_point(
            np.array(nav_point), 
            success_thres_dist=1.5,  # Match iGibson's initial approach distance
            face_target=False,
            early_termination_dist=0.5
        )
        
        if not subtask_success_nav:
            feedback = f"Navigation to {obj.name} failed"
            return False, feedback
        
        # Step 2: For doors, do a second closer navigation
        # iGibson does a precise approach with success_thres_dist=0.21
        # We'll use 0.5 since Habitat's navmesh can be less precise
        if is_door:
            # Get the door's actual position for final approach
            door_pos = obj.get_position()
            door_pos_2d = np.array([door_pos[0], door_pos[2]])  # X, Z in Habitat
            
            # Navigate closer to the door
            subtask_success_nav = self.navigate_to_point(
                door_pos_2d, 
                success_thres_dist=0.5,  # Close enough to "interact"
                face_target=True,  # Face the door
                early_termination_dist=0.0
            )
            
            if not subtask_success_nav:
                # Even if precise navigation fails, try to open if we're reasonably close
                robot_pos = self.env.robots[0].get_position()
                dist_to_door = np.linalg.norm(np.array([robot_pos[0], robot_pos[2]]) - door_pos_2d)
                if dist_to_door > 2.0:  # Too far to interact
                    feedback = f"Navigation to {obj.name} failed - too far to interact"
                    return False, feedback
                # Else: proceed anyway if we're close enough
        
        # Step 3: Delete the door from the SLAM map so navigation can path through
        # This is CRITICAL - without this, the door remains as an obstacle
        # and the robot cannot plan paths through the doorway
        if is_door:
            try:
                self.env.slam.delete_obj_from_voxel_map(obj)
                print(f"Deleted '{obj.name}' from occupancy map")
            except Exception as e:
                print(f"Warning: Could not delete door from map: {e}")
        
        # Step 4: Mark door as opened
        self.env.opened_doors.append(obj.name)
        self.env.episode_info["magic_open_actions"] = \
            self.env.episode_info.get("magic_open_actions", 0) + 1
        
        print(f"Object '{obj.name}' opened.")
        
        # Step 5: For doors, navigate through the door frame
        # This helps the robot enter the new area and see the new space
        if is_door:
            door_pos = obj.get_position()
            door_pos_2d = np.array([door_pos[0], door_pos[2]])
            
            # Navigate through the door (match iGibson: success_thres_dist=0.3)
            through_door_success = self.navigate_to_point(
                door_pos_2d, 
                success_thres_dist=0.3,
                face_target=False,
                early_termination_dist=0.0
            )
            
            if not through_door_success:
                # Not critical - door is still opened
                print(f"Note: Could not navigate through door frame")
            
            # Step 6: Turn around to see the new area
            # This is important because the robot camera only sees forward
            # Without this, the SLAM won't detect the new space
            try:
                from moma_llm.navigation.habitat_navigation import turn_full_circle
                print("Scanning new area after door opening...")
                turn_full_circle(self.env)
            except Exception as e:
                print(f"Warning: Could not scan new area: {e}")
        
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

        # Simplified action space aligned with Habitat navigation paradigm
        # Uses single-argument actions for easier parsing
        self.possible_actions = {
            "goto": ("target", "navigate to a target object or room. Example: goto(sofa), goto(kitchen)"),
            "open": ("object_name", "navigate to and open a door or container. Example: open(door), open(cabinet)"),
            "explore": ("room_name", "explore unexplored areas in a room. Example: explore(bedroom)"),
            "stop": ("", "call when the task is completed or no further actions possible.")
        }
        
        if not self.env.config.get("consider_open_actions", True):
            self.possible_actions.pop("open", None)
        
        self.llm = llm
        self.room_classification = dict()
        self.prev_responses = [""]
        self.action_history = []
        self.last_env_feedback = {"role": "user", "content": ""}
        self.visited_rooms = set()  # Track rooms the robot has visited
        
        # Track room classification stability - by position centroid, not room ID
        self._room_classification_history = {}  # {position_key: {"name": str, "objects": set, "locked": bool}}
        self._generic_room_names = {"other room", "unknown", "room", "other"}  # Names that can be re-classified
        self._room_position_tolerance = 3.0  # meters - rooms within this distance are considered same

    def reset(self, config_file: str, scene_id: str, episode_num: int) -> Dict:
        """Reset for new episode."""
        self.prev_responses = [""]
        self.action_history = []
        self.last_env_feedback = {"role": "user", "content": ""}
        self.visited_rooms = set()  # Reset visited rooms tracking
        self._room_classification_history = {}  # Reset classification history
        self.room_classification = {}  # Reset current classification
        if hasattr(self.llm, 'reset_episode_metrics'):
            self.llm.reset_episode_metrics()
        return super().reset(config_file, scene_id, episode_num, compute_scene_graph=True)

    def _get_room_position_key(self, graph: nx.DiGraph, room: str) -> Optional[Tuple[float, float]]:
        """Get a position key for a room based on its centroid."""
        room_node = graph.nodes.get(room, {})
        pos_map = room_node.get("pos_map")
        if pos_map is not None:
            # Convert to world coordinates and round to reduce precision issues
            pos_world = self.env.slam.voxel2world(np.array(pos_map[:2]))
            return (round(pos_world[0], 1), round(pos_world[1], 1))
        return None
    
    def _find_matching_history(self, room_pos: Tuple[float, float]) -> Optional[Dict]:
        """Find a matching room in history based on position proximity."""
        if room_pos is None:
            return None
        
        for hist_pos, history in self._room_classification_history.items():
            if isinstance(hist_pos, tuple) and len(hist_pos) == 2:
                dist = np.sqrt((room_pos[0] - hist_pos[0])**2 + (room_pos[1] - hist_pos[1])**2)
                if dist < self._room_position_tolerance:
                    return history
        return None

    def classify_rooms(self, obs: Dict):
        """
        Classify rooms using LLM with stability mechanism.
        
        Stability is tracked by POSITION, not room ID, because room IDs
        are regenerated each time room detection runs.
        
        Stability rules:
        1. Once a room gets a specific name (not "other room"/"unknown"), it's locked
        2. Only re-classify if current name is generic or significantly more objects found
        3. This prevents inconsistent naming as exploration progresses
        """
        graph = obs["room_object_graph"]
        rooms = list(graph.successors("root")) if "root" in graph else []
        
        # Get current objects and positions for each room
        current_room_objects = {}
        current_room_positions = {}
        for room in rooms:
            objects = set()
            for obj_node in graph.successors(room):
                obj_name = self.llm.to_human_readable_object_name(obj_node)
                objects.add(obj_name)
            current_room_objects[room] = objects
            current_room_positions[room] = self._get_room_position_key(graph, room)
        
        # Determine which rooms need re-classification
        rooms_to_classify = []
        room_to_history = {}  # Map room ID to its matching history
        
        for room in rooms:
            current_objects = current_room_objects.get(room, set())
            room_pos = current_room_positions.get(room)
            
            # Find matching history by position (not by room ID!)
            history = self._find_matching_history(room_pos)
            room_to_history[room] = history
            
            needs_classification = False
            
            if history is None:
                # New room (no position match) - needs classification
                needs_classification = True
                self._debug_print(f"Room classification: '{room}' at {room_pos} is new, will classify")
            elif history.get("locked", False):
                # Room has a stable, non-generic classification - keep it
                self._debug_print(f"Room classification: '{room}' at {room_pos} is locked as '{history['name']}', keeping")
                needs_classification = False
            else:
                # Check if current name is generic (can be re-classified)
                current_name = history.get("name", "").lower()
                is_generic = any(g in current_name for g in self._generic_room_names)
                
                if is_generic:
                    # Generic name - check if we have significantly more objects now
                    prev_objects = history.get("objects", set())
                    new_objects = current_objects - prev_objects
                    
                    # Re-classify if we found 5+ new objects or 50%+ more objects
                    if len(new_objects) >= 5 or (len(prev_objects) > 0 and len(current_objects) > len(prev_objects) * 1.5):
                        needs_classification = True
                        self._debug_print(f"Room classification: '{room}' is generic ('{current_name}') with {len(new_objects)} new objects, will re-classify")
                    else:
                        self._debug_print(f"Room classification: '{room}' is generic ('{current_name}') but not enough new objects ({len(new_objects)} new)")
                else:
                    # Non-generic name - lock it
                    history["locked"] = True
                    self._debug_print(f"Room classification: '{room}' has specific name '{current_name}', locking")
            
            if needs_classification:
                rooms_to_classify.append(room)
        
        # Build final classification
        room_classification = {}
        
        # First, keep existing stable classifications
        for room in rooms:
            history = room_to_history.get(room)
            if history and room not in rooms_to_classify:
                room_classification[room] = history["name"]
        
        # Then classify new/generic rooms
        if rooms_to_classify:
            try:
                new_classification = self.llm.classify_rooms(obs)
                
                for room in rooms_to_classify:
                    if room in new_classification:
                        new_name = new_classification[room]
                        room_classification[room] = new_name
                        
                        # Update history - use POSITION as key, not room ID
                        room_pos = current_room_positions.get(room)
                        if room_pos:
                            is_generic = any(g in new_name.lower() for g in self._generic_room_names)
                            self._room_classification_history[room_pos] = {
                                "name": new_name,
                                "objects": current_room_objects.get(room, set()).copy(),
                                "locked": not is_generic  # Lock if not generic
                            }
                            self._debug_print(f"Room classification: '{room}' at {room_pos} classified as '{new_name}' (locked={not is_generic})")
            except Exception as e:
                self._debug_print(f"Room classification error: {e}")
                # Fall back to previous classifications or generic names
                for room in rooms_to_classify:
                    history = room_to_history.get(room)
                    if history:
                        room_classification[room] = history["name"]
                    else:
                        room_classification[room] = "other room"
        
        # Handle duplicate room names (add suffix -1, -2, etc.)
        tot_counter = Counter(room_classification.values())
        counter = defaultdict(int)
        for k, v in list(room_classification.items()):
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
        """
        Parse action from LLM response.
        
        Expected format: action_name(argument)
        Valid actions: goto, open, explore, stop
        """
        action = None
        argument = ""
        command_found = False
        
        valid_actions = set(self.possible_actions.keys()) | {"navigate", "go_to_and_open", "done"}  # Include legacy
        
        for line in response.split("\n"):
            command_found = command_found or line.lower().replace("'", "").replace('"', "").replace("\\", "").strip().startswith("command")
            if not command_found:
                continue
            try:
                match = re.search(r"(\w+)\((.*)\)", line)
                if match:
                    parsed_action, parsed_arg = match.groups()
                    parsed_arg = parsed_arg.replace("'", "").replace('"', "").replace("\\", "").strip()
                    
                    # Validate action name
                    if parsed_action.lower() in [a.lower() for a in valid_actions]:
                        action = parsed_action.lower()
                        argument = parsed_arg
                        break
            except:
                continue
            
        if not action:
            print("Could not parse command. Calling stop().")
            if mode == 'train':
                action = "format_error"
            elif mode == 'eval':
                action = "stop"  # Use new action name
            argument = ""
            self.env.episode_info["failure_reason"] = "Unable to parse LLM command."
            
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
                       target_name: str = "",
                       *args,
                       **kwargs) -> Conversation:
        """Create prompt for LLM.
        
        Supports both v1 (original) and v2 (improved) prompts based on config.
        Set config["prompt_version"] = 2 to use v2 prompts.
        """
        # Check which prompt version to use
        prompt_version = self.env.config.get("prompt_version", 1)
        
        # Generate tool descriptions (same for both versions)
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
            # Add exploration status indicator
            if room in rooms_with_any_frontier:
                objects = objects + ["(has unexplored areas)"]
            else:
                objects = objects + ["(fully explored)"]
            list_found_rooms_and_objects += f"- {room}: [{', '.join(objects)}]\n"

        rooms_with_frontier_descr = f"[{', '.join([f'{room} ({distance_mapping(dist)})' for room, dist in rooms_with_frontier_leading_out])}]"
        if not rooms_with_frontier_leading_out:
            rooms_with_frontier_descr = "[none - all rooms fully explored]"
        
        # Identify fully explored rooms (no frontiers left)
        fully_explored_rooms = [r for r in labelled_rooms if r not in rooms_with_any_frontier]
        fully_explored_descr = f"[{', '.join(fully_explored_rooms)}]" if fully_explored_rooms else "[none yet]"
        
        # Format visited rooms
        visited_rooms_descr = f"[{', '.join(sorted(self.visited_rooms))}]" if self.visited_rooms else "[none yet]"
        
        rooms_with_closed_doors_descr = ""
        if len(rooms_with_closed_doors):
            rooms_with_closed_doors_descr = f"These rooms contain closed doors that might open up new space: [{', '.join([f'{room} ({distance_mapping(dist)})' for room, dist in rooms_with_closed_doors])}]."

        if prompt_version == 2:
            # Use v2 prompts with improved formatting
            system_prompt = prompts_v2.SYSTEM_PROMPT.format(
                TASK_DESCRIPTION=task_description,
                TOOL_DESCRIPTIONS=tool_descriptions
            )
            
            # Format action history with failure reasons
            action_history_section = prompts_v2.format_action_history_from_dataclass(
                self.action_history,
                self.llm.to_human_readable_object_name
            )
            
            # Generate contextual decision guidance
            target_found = prompts_v2.check_target_in_objects(target_name, room_dict) if target_name else False
            recent_failures = prompts_v2.count_recent_failures(self.action_history)
            has_unexplored = len(rooms_with_frontier_leading_out) > 0
            all_explored = len(fully_explored_rooms) == len(labelled_rooms) and len(labelled_rooms) > 0
            
            decision_guidance = prompts_v2.get_decision_guidance(
                target_found=target_found,
                target_name=target_name,
                recent_failure_count=recent_failures,
                has_unexplored_areas=has_unexplored,
                all_rooms_explored=all_explored
            )
            
            user_prompt = prompts_v2.USER_PROMPT.format(
                CURRENT_ROOM=current_room,
                LIST_NEARBY_OBJECTS=list_nearby_objects,
                LIST_FOUND_ROOMS_AND_OBJECTS=list_found_rooms_and_objects,
                ACTION_HISTORY_SECTION=action_history_section,
                ROOMS_WITH_FRONTIER_DESCRIPTION=rooms_with_frontier_descr,
                FULLY_EXPLORED_ROOMS=fully_explored_descr,
                VISITED_ROOMS=visited_rooms_descr,
                ROOMS_WITH_CLOSED_DOORS_DESCRIPTION=rooms_with_closed_doors_descr,
                DECISION_GUIDANCE=decision_guidance
            )
        else:
            # Use v1 prompts (original)
            system_prompt = prompts_v1.SYSTEM_PROMPT.format(
                TASK_DESCRIPTION=task_description,
                TOOL_DESCRIPTIONS=tool_descriptions
            )
            
            list_previous_actions = ""
            if len(nlp_history):
                list_previous_actions = f"Your {len(nlp_history)} previous actions were: {', '.join(nlp_history)}."

            user_prompt = prompts_v1.USER_PROMPT.format(
                CURRENT_ROOM=current_room,
                LIST_NEARBY_OBJECTS=list_nearby_objects,
                LIST_FOUND_ROOMS_AND_OBJECTS=list_found_rooms_and_objects,
                LIST_PREVIOUS_ACTIONS=list_previous_actions if len(nlp_history) else "",
                ROOMS_WITH_FRONTIER_DESCRIPTION=rooms_with_frontier_descr,
                FULLY_EXPLORED_ROOMS=fully_explored_descr,
                VISITED_ROOMS=visited_rooms_descr,
                ROOMS_WITH_CLOSED_DOORS_DESCRIPTION=rooms_with_closed_doors_descr,
            )

        conversation = Conversation(messages=[
            self.last_env_feedback,
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        return conversation

    def _get_retry_prompt(self, failure_reason: str = None) -> str:
        """Get retry prompt based on configured prompt version.
        
        Args:
            failure_reason: Specific failure reason (used by v2 prompts)
            
        Returns:
            Retry prompt string
        """
        prompt_version = self.env.config.get("prompt_version", 1)
        if prompt_version == 2:
            return prompts_v2.format_retry_prompt(failure_reason)
        else:
            return prompts_v1.RETRIAL_PROMPT
    
    def _get_format_error_prompt(self) -> str:
        """Get format error prompt based on configured prompt version."""
        prompt_version = self.env.config.get("prompt_version", 1)
        if prompt_version == 2:
            return prompts_v2.RETRIAL_PROMPT_FORMAT_ERROR
        else:
            return prompts_v1.RETRIAL_PROMPT_FORMAT_ERROR

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
        
        # Habitat uses Y-up coordinate system: robot position is (X, Y, Z) where Y is up
        # For 2D distance calculation, use X and Z (horizontal plane)
        robot_pos_3d = self.env.robots[0].get_position()
        robot_pos_2d = np.array([[robot_pos_3d[0], robot_pos_3d[2]]])  # X, Z for 2D
        
        dist_matrix = distance_matrix(robot_pos_2d, node_coords_world)
        close_objects = np.array(current_room_objects)[
            np.squeeze(dist_matrix < closeness_thresh, 0)
        ]
        return set([self.llm.to_human_readable_object_name(o) for o in close_objects])

    def _match_action_history(self, graph: nx.DiGraph, 
                              separated_voronoi_graph: nx.Graph) -> List[str]:
        """Match action history to current graph state for LLM context."""
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
            object_name, invalid = None, False
            
            if h.object_name_graph is not None:
                object_name = self.llm.to_human_readable_object_name(h.object_name_graph)
            
            nlp = None
            # Map to new simplified action format
            action = h.action
            
            if action in ("done", "stop"):
                nlp = "stop()"
            elif action == "explore":
                room_name = _match_room_pos(h.position)
                if room_name is not None:
                    nlp = f"explore({room_name})"
                else:
                    nlp = f"{h.orig_api_call} - invalid"
                    invalid = True
            elif action in ("goto", "navigate"):
                if object_name is not None:
                    nlp = f"goto({object_name})"
                else:
                    nlp = f"{h.orig_api_call} - invalid"
                    invalid = True
            elif action in ("open", "go_to_and_open"):
                if object_name is not None:
                    nlp = f"open({object_name})"
                else:
                    nlp = f"{h.orig_api_call} - invalid"
                    invalid = True
            else:
                nlp = f"{h.orig_api_call} - unknown action"
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
    def _parse_single_argument(argument: str) -> str:
        """
        Parse a single argument (target name) from action call.
        Simplified parsing for the new action format.
        """
        # Strip whitespace and quotes
        target = argument.strip().strip('"').strip("'")
        
        # Remove distance annotations like "(near)", "(far)"
        for v in DIST_MAPPING.values():
            target = target.replace(f"({v})", "")
        
        # Handle underscores and normalize
        target = target.replace("_", " ").strip()
        
        # Convert plural to singular for object matching
        singular = inflect_engine.singular_noun(target)
        if singular:
            target = singular
            
        return target

    @staticmethod
    def _parse_room_argument(room_name: str) -> str:
        """Parse room name from argument."""
        # Strip whitespace and quotes first
        room_name = room_name.strip().strip('"').strip("'")
        
        # Only apply suffix transformation if the name ends with a number-like pattern (e.g., "-1", "-2")
        # Don't mangle room names like "combined kitchen and living room"
        if len(room_name) >= 3 and room_name[-2] == '-' and room_name[-1].isdigit():
            room_name = room_name[:-3].replace("_", " ").replace("-", " ") + room_name[-3:].replace("_", "-")
        else:
            room_name = room_name.replace("_", " ")
        
        for v in DIST_MAPPING.values():
            room_name = room_name.replace(f"({v})", "")
        return room_name.strip()

    @staticmethod
    def _parse_room_object_argument(argument: str) -> Tuple[str, str]:
        """Parse room and object from argument (legacy support)."""
        # Handle case where argument might have extra quotes or spaces
        argument = argument.strip().strip('"').strip("'")
        
        # Split on ", " - the comma-space separator
        parts = argument.split(", ")
        if len(parts) < 2:
            # Try splitting on just comma
            parts = argument.split(",")
        
        if len(parts) >= 2:
            room_name = parts[0].strip()
            object_name = ", ".join(parts[1:]).strip()  # Handle object names with commas
        else:
            room_name = argument
            object_name = ""
        
        singular = inflect_engine.singular_noun(object_name)
        if singular:
            object_name = singular
        
        parsed_room = HabitatLLMEnv._parse_room_argument(room_name)
        return parsed_room, object_name
    
    def _find_target_in_graph(self, graph: nx.DiGraph, target: str) -> Tuple[str, Optional[str], List[str]]:
        """
        Find a target (object or room) in the scene graph.
        
        Args:
            graph: Room-object scene graph
            target: Target name to find (object or room)
            
        Returns:
            Tuple of (target_type: 'object'|'room'|'unknown', 
                     room_name: str or None,
                     matching_nodes: list of matching node names)
        """
        target_lower = target.lower()
        target_graph = self.llm.human_to_graph_name(target)
        
        # Get all rooms (direct successors of root)
        rooms = list(graph.successors("root")) if "root" in graph else []
        
        # Check if target is a room name
        for room in rooms:
            if target_lower in room.lower() or room.lower() in target_lower:
                return "room", room, [room]
        
        # Search for objects across all rooms
        matching_objects = []
        matching_room = None
        
        for room in rooms:
            for obj_node in graph.successors(room):
                # Check if target matches the object
                if target_graph in obj_node or target_lower in obj_node.lower():
                    matching_objects.append(obj_node)
                    matching_room = room
        
        if matching_objects:
            return "object", matching_room, matching_objects
        
        return "unknown", None, []

    def execute_action(self, action: str, argument: str, task_desc: str,
                       graph: nx.DiGraph, vor_graph: Optional[nx.Graph] = None) -> Tuple[bool, bool, Dict, Dict]:
        """
        Execute high-level action.
        
        Simplified action space:
        - goto(target): Navigate to an object or room
        - open(object): Navigate to and open a door/container
        - explore(room): Explore unexplored areas in a room
        - stop(): Terminate task
        
        Legacy support for old action names:
        - navigate -> goto
        - go_to_and_open -> open
        - done -> stop
        """
        print(f"Executing {action}({argument})")
        done = False
        feedback = ""
        
        # Legacy action name mapping
        action_mapping = {
            "navigate": "goto",
            "go_to_and_open": "open", 
            "done": "stop"
        }
        action = action_mapping.get(action, action)
        
        if action == "goto":
            assert vor_graph is not None
            target = self._parse_single_argument(argument)
            target_type, room_name, matching_nodes = self._find_target_in_graph(graph, target)
            
            self._debug_print(f"DEBUG goto: target='{target}', type={target_type}, room={room_name}, matches={matching_nodes}")
            
            if target_type == "room" and room_name:
                # Navigate to room center
                room_pos = graph.nodes.get(room_name, {}).get("pos_map")
                if room_pos:
                    target_world = self.env.slam.voxel2world(np.array(room_pos[:2]))
                    subtask_success = self.navigate_to_point(
                        target_world,
                        success_thres_dist=2.0,
                        face_target=True,
                        early_termination_dist=0.5
                    )
                    if not subtask_success:
                        feedback = f"Navigation to room {room_name} failed"
                    history = ActionHistory(
                        action="goto",
                        object_name_graph=None,
                        position=tuple(target_world),
                        subtask_success=subtask_success
                    )
                else:
                    subtask_success = False
                    feedback = f"Room {room_name} position unknown"
                    history = ActionHistory(
                        action="goto",
                        object_name_graph=None,
                        position=None,
                        subtask_success=subtask_success
                    )
                    
            elif target_type == "object" and matching_nodes:
                # Navigate to closest matching object
                idx, point, _costs, _paths = self._find_closest_point(
                    [graph.nodes.get(n) for n in matching_nodes]
                )
                subtask_success = self.navigate_to_point(
                    np.array(point),
                    success_thres_dist=1.5,
                    face_target=True,
                    early_termination_dist=0.3
                )
                if not subtask_success:
                    feedback = f"Navigation to {target} failed"
                history = ActionHistory(
                    action="goto",
                    object_name_graph=matching_nodes[idx],
                    position=point,
                    subtask_success=subtask_success
                )
            else:
                subtask_success = False
                feedback = f"Target '{target}' not found in known rooms/objects"
                history = ActionHistory(
                    action="goto",
                    object_name_graph=self.llm.human_to_graph_name(target),
                    position=None,
                    subtask_success=subtask_success
                )
                
        elif action == "open":
            assert vor_graph is not None
            target = self._parse_single_argument(argument)
            subtask_success, (obj, room_pos), feedback = self._open_by_target(
                graph, vor_graph, target=target
            )
            # Habitat uses Y-up coordinate system: position is (X, Y, Z) where Y is up
            # For 2D navigation, we use X and Z
            if obj:
                obj_pos = obj.get_position()
                position_2d = (obj_pos[0], obj_pos[2]) if len(obj_pos) >= 3 else obj_pos[:2]
            else:
                position_2d = room_pos
            history = ActionHistory(
                action="open",
                object_name_graph=obj.name if obj else None,
                position=position_2d,
                subtask_success=subtask_success,
                opendoors_roompos=room_pos
            )
            
        elif action == "explore":
            room_name = self._parse_single_argument(argument)
            self._debug_print(f"DEBUG explore: Processing explore({room_name})")
            
            # Find matching room (allow partial match)
            rooms = list(graph.successors("root")) if "root" in graph else []
            matched_room = None
            for r in rooms:
                if room_name.lower() in r.lower() or r.lower() in room_name.lower():
                    matched_room = r
                    break
            
            if matched_room:
                room_name = matched_room
                
            try:
                raw_frontier_points = graph.nodes.get(room_name, {}).get("frontier_points", set())
                self._debug_print(f"DEBUG explore:   Room '{room_name}' has {len(raw_frontier_points)} frontier points")
                frontier_points_within, frontier_points_leading_out = split_frontier_points(raw_frontier_points)
                self._debug_print(f"DEBUG explore:   Frontier within: {len(frontier_points_within)}, leading out: {len(frontier_points_leading_out)}")
                frontier_points = frontier_points_leading_out if frontier_points_leading_out else frontier_points_within
            except Exception as e:
                frontier_points = []
                self._debug_print(f"DEBUG explore:   ERROR parsing: {e}")
                
            if len(frontier_points) == 0:
                subtask_success = False
                feedback = f"No unexplored areas in {room_name}"
                self._debug_print(f"DEBUG explore: FAILED - {feedback}")
                # Show what rooms have frontiers
                self._debug_print(f"DEBUG explore:   Rooms with frontiers:")
                for r in rooms:
                    fp = graph.nodes.get(r, {}).get("frontier_points", set())
                    if len(fp) > 0:
                        self._debug_print(f"DEBUG explore:   - '{r}' has {len(fp)} frontier points")
                history = ActionHistory(
                    action="explore",
                    object_name_graph=None,
                    position=graph.nodes.get(room_name, {}).get("pos_map"),
                    subtask_success=subtask_success
                )
            else:
                _idx, closest_frontier, _costs, _paths = self._find_closest_point(list(frontier_points))
                self._debug_print(f"DEBUG explore:   Navigating to frontier at [{closest_frontier[0]:.3f}, {closest_frontier[1]:.3f}]")
                # Use larger threshold for frontiers since they're often near walls/obstacles
                # where the navmesh may not allow precise positioning
                subtask_success = self.navigate_to_point(
                    closest_frontier, 
                    success_thres_dist=1.5,  # Relaxed from 0.5m - frontiers are at edges
                    face_target=True, 
                    early_termination_dist=1.0  # Early terminate if reasonably close
                )
                if not subtask_success:
                    feedback = "Navigation to frontier failed"
                    self._debug_print(f"DEBUG explore: FAILED - Navigation to frontier failed")
                else:
                    self._debug_print(f"DEBUG explore: SUCCESS - Reached frontier")
                    # Do a 360-degree scan to capture all visible objects in the area
                    self._debug_print(f"DEBUG explore: Performing 360° scan to capture room objects...")
                    try:
                        turn_full_circle(self.env)
                        self._debug_print(f"DEBUG explore: 360° scan completed - objects and room info updated")
                    except Exception as e:
                        self._debug_print(f"DEBUG explore: Warning - 360° scan failed: {e}")
                history = ActionHistory(
                    action="explore",
                    object_name_graph=None,
                    position=closest_frontier,
                    subtask_success=subtask_success
                )
                
        elif action == "stop":
            subtask_success = True
            done = True
            history = ActionHistory(action="stop", object_name_graph=None, position=None, subtask_success=subtask_success)
            
        else:
            # Unknown action - return failure
            subtask_success = False
            feedback = f"Unknown action '{action}'. Available: goto, open, explore, stop"
            history = ActionHistory(action=action, object_name_graph=None, position=None, subtask_success=subtask_success)
            self.env.episode_info["failure_reason"] = f"Unknown action: {action}"

        history.orig_api_call = f"{action}({argument})"
        history.feedback = feedback  # Store failure reason for v2 prompts
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

    def _open_by_target(self, graph: nx.DiGraph, vor_graph: nx.Graph,
                        target: str) -> Tuple[bool, Tuple, str]:
        """
        Open a target object (door/container) by name.
        
        Simplified version that takes a single target name instead of room+object.
        Searches all rooms for matching closed doors or containers.
        
        Args:
            graph: Room-object scene graph
            vor_graph: Voronoi navigation graph
            target: Target object name to open (e.g., "door", "cabinet")
            
        Returns:
            Tuple of (success, (object, room_pos), feedback)
        """
        target_lower = target.lower()
        target_graph = self.llm.human_to_graph_name(target)
        is_door = "door" in target_lower
        
        self._debug_print(f"DEBUG open: Looking for '{target}' (is_door={is_door})")
        
        # Search all rooms for matching objects
        rooms = list(graph.successors("root")) if "root" in graph else []
        possible_nodes = []
        found_room = None
        
        for room in rooms:
            room_node = graph.nodes.get(room, {})
            
            if is_door:
                # Check closed doors in this room
                closed_doors = room_node.get("closed_doors", [])
                for door in closed_doors:
                    if target_graph in door.lower() or target_lower in door.lower():
                        possible_nodes.append(door)
                        found_room = room
            else:
                # Check objects in this room that match and are closed
                for obj_node in graph.successors(room):
                    node_data = graph.nodes.get(obj_node, {})
                    is_closed = not node_data.get('states', {}).get(object_states.Open, True)
                    if (target_graph in obj_node or target_lower in obj_node.lower()) and is_closed:
                        possible_nodes.append(obj_node)
                        found_room = room
        
        if not possible_nodes:
            feedback = f"No closed '{target}' found"
            self._debug_print(f"DEBUG open: FAILED - {feedback}")
            return False, (None, None), feedback
        
        self._debug_print(f"DEBUG open: Found {len(possible_nodes)} candidates: {possible_nodes}")
        
        # Delegate to legacy method with constructed argument
        if found_room:
            legacy_argument = f"{found_room}, {possible_nodes[0]}"
        else:
            legacy_argument = target
            
        return self._open_graph_node(graph, vor_graph, argument=legacy_argument)

    def _open_graph_node(self, graph: nx.DiGraph, vor_graph: nx.Graph, 
                         argument: str) -> Tuple[bool, Tuple, str]:
        """
        Open a node in the graph (door/container).
        
        Aligned with iGibson's open_graph_node logic:
        1. Parse room and object from argument
        2. Find possible nodes (closed doors or articulated objects)
        3. For doors: find closest via voronoi node proximity
        4. For other objects: find closest by navigation cost
        5. Call open_object with the closest node and navigation point
        """
        def _get_possible_nodes(room_name: str, object_name: str):
            """Get possible nodes matching the object name in the room."""
            if "door" in object_name:
                room_node = graph.nodes.get(room_name, {})
                return list(room_node.get("closed_doors", []))
            else:
                possible = []
                graph_name = self.llm.human_to_graph_name(object_name)
                for n in graph.successors(room_name):
                    node_data = graph.nodes.get(n, {})
                    is_closed = not node_data.get('states', {}).get(object_states.Open, True)
                    if ((graph_name in n) or (graph_name.strip('s') in n)) and is_closed:
                        possible.append(n)
                return possible
        
        try:
            room_name, object_name = self._parse_room_object_argument(argument)
            possible_nodes = _get_possible_nodes(room_name, object_name)
            self._debug_print(f"go_to_and_open: room='{room_name}', object='{object_name}', found {len(possible_nodes)} candidates")
        except Exception as e:
            room_name, object_name = "None", "None"
            possible_nodes = []
            print(f"ERROR _open_graph_node: Failed to parse argument '{argument}': {e}")

        if not possible_nodes:
            feedback = f"Object opening failed: no closed {object_name} in {room_name}"
            self._debug_print(f"go_to_and_open FAILED: {feedback}")
            return False, (None, graph.nodes.get(room_name, {}).get("pos_map")), feedback

        # Build node data for each possible node
        # First try to get from graph, then from scene objects
        possible_node_data = []
        for node_name in possible_nodes:
            # Check if node exists in graph with data
            if node_name in graph.nodes and graph.nodes.get(node_name):
                node_data = graph.nodes[node_name]
                possible_node_data.append({
                    "name": node_name,
                    "pos": node_data.get("pos", [0, 0, 0]),
                    "pos_map": node_data.get("pos_map", [0, 0]),
                    "closest_vor_node": node_data.get("closest_vor_node", node_data.get("pos_map", [0, 0])),
                })
            else:
                # Try to get from scene objects
                scene_obj = self.env.scene.objects_by_name.get(node_name)
                if scene_obj is not None:
                    pos = scene_obj.get_position()
                    possible_node_data.append({
                        "name": node_name,
                        "pos": tuple(pos[:3]),
                        "pos_map": tuple(self.env.slam.world2voxel(np.array([pos[0], pos[2]]))),  # Use X, Z for 2D
                        "closest_vor_node": tuple(self.env.slam.world2voxel(np.array([pos[0], pos[2]]))),
                    })
                    self._debug_print(f"go_to_and_open: Found '{node_name}' in scene at [{pos[0]:.3f}, {pos[2]:.3f}]")
        
        if not possible_node_data:
            feedback = f"Could not find node data for {object_name}"
            self._debug_print(f"go_to_and_open FAILED: {feedback}")
            return False, (None, None), feedback
        
        # For doors, use voronoi-based closest point finding (like iGibson)
        # This finds the closest navigable point NEAR the door, not the door itself
        if "door" in argument:
            # Get all voronoi nodes
            vor_nodes = np.array(list(vor_graph.nodes))
            closest_vor_nodes = {}
            
            for node_data in possible_node_data:
                node_name = node_data["name"]
                # Get door position in map coordinates
                pos = np.array(node_data["pos_map"][:2])
                
                # Find voronoi nodes close to the door
                dists_to_door = np.linalg.norm(pos - vor_nodes, axis=1)
                close_threshold = max(2.0 / self.env.slam.voxel_size, dists_to_door.min() + 1)
                is_close_to_door = dists_to_door <= close_threshold
                
                if np.any(is_close_to_door):
                    # Get world coordinates of close voronoi nodes
                    close_vor_nodes = vor_nodes[is_close_to_door]
                    points = [tuple(p) for p in self.env.slam.voxel2world(close_vor_nodes)]
                    
                    # Find closest navigable point
                    try:
                        _idx, _point, vn_costs, _paths = self._find_closest_point(points)
                        # Add distance to door as tiebreaker
                        vn_costs = np.array(vn_costs) + 2 * dists_to_door[is_close_to_door]
                        best_idx = np.argmin(vn_costs)
                        closest_vor_nodes[node_name] = (points[best_idx], vn_costs[best_idx])
                    except:
                        # Fallback to door position
                        door_world_pos = self.env.slam.voxel2world(pos)
                        closest_vor_nodes[node_name] = (tuple(door_world_pos), float('inf'))
            
            # Find the overall closest door
            if closest_vor_nodes:
                best_door = min(closest_vor_nodes.items(), key=lambda x: x[1][1])
                node_name_to_open = best_door[0]
                nav_point = np.array(best_door[1][0])
            else:
                # Fallback
                idx, nav_point, _, _ = self._find_closest_point(possible_node_data)
                node_name_to_open = possible_nodes[idx] if idx < len(possible_nodes) else possible_node_data[idx]["name"]
        else:
            # For non-door objects, use standard closest point finding
            idx, nav_point, _costs, _paths = self._find_closest_point(possible_node_data)
            node_name_to_open = possible_nodes[idx] if idx < len(possible_nodes) else possible_node_data[idx]["name"]
        
        self._debug_print(f"go_to_and_open: Opening '{node_name_to_open}' via nav point [{nav_point[0]:.3f}, {nav_point[1]:.3f}]")
        
        # Get or create the object
        obj = self.env.scene.objects_by_name.get(node_name_to_open)
        if obj is None:
            # Create mock object with position from our data
            node_data = next((d for d in possible_node_data if d["name"] == node_name_to_open), {})
            actual_pos = np.array(node_data.get("pos", [nav_point[0], 0.0, nav_point[1]]))
            
            class MockObj:
                def __init__(self, name, position):
                    self.name = name
                    self._position = position
                def get_position(self):
                    return self._position
            obj = MockObj(node_name_to_open, actual_pos)
            
        open_success, feedback = self.open_object(obj, nav_point)
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
        
        # Track visited rooms for exploration awareness
        if current_room and current_room != "unknown":
            self.visited_rooms.add(current_room)

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

        # Get target name for v2 prompts
        target_name = ""
        if self.env.task and hasattr(self.env.task, 'target_category'):
            target_name = self.llm.to_human_readable_object_name(self.env.task.target_category)
        
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
            # Get last failure reason for v2 prompts
            last_failure_reason = self.action_history[-1].feedback if self.action_history else None
            retry_prompt = self._get_retry_prompt(failure_reason=last_failure_reason)
            conversation.add_message({"role": "user", "content": retry_prompt})
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
                conversation.add_message({"role": "user", "content": self._get_format_error_prompt()})
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
        
        # Track visited rooms for exploration awareness
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

        # Get target name for v2 prompts
        target_name = ""
        if self.env.task and hasattr(self.env.task, 'target_category'):
            target_name = self.llm.to_human_readable_object_name(self.env.task.target_category)
        
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
        
        # Check max LLM queries limit before sending query
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

            # Get last failure reason for v2 prompts
            last_failure_reason = self.action_history[-1].feedback if self.action_history else None
            retry_prompt = self._get_retry_prompt(failure_reason=last_failure_reason)
            conversation.add_message({"role": "user", "content": retry_prompt})
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

