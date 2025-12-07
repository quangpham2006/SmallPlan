# Habitat-compatible Baselines for SmallPlan
# Provides GreedyBaseline and RandomBaseline without iGibson dependencies

import os
import random
from typing import Dict, List, Any, Tuple, Optional

import networkx as nx
import numpy as np

from moma_llm.llm.habitat_llm import object_states
from moma_llm.env.habitat_llm_env import HabitatLLMEnv
from moma_llm.utils.habitat_constants import NODETYPE, POSSIBLE_ROOMS


class HabitatGreedyBaseline(HabitatLLMEnv):
    """
    Greedy baseline agent for Habitat.
    Always navigates to the closest frontier or closed object.
    """
    
    def __init__(self, env, llm, seed: int) -> None:
        super().__init__(env, llm, seed=seed)
        self.nav_fails = 0
        self.max_nav_fails = 10

    def reset(self, *args, **kwargs) -> Dict:
        """Reset baseline state for new episode."""
        self.nav_fails = 0
        return super().reset(*args, **kwargs)

    def frontier_navigation(self, frontier_point: np.ndarray) -> bool:
        """
        Navigate to a frontier point.
        
        Args:
            frontier_point: Target frontier position
            
        Returns:
            Whether navigation was successful
        """
        subtask_success = self.navigate_to_point(
            frontier_point, 
            success_thres_dist=1.5, 
            face_target=True, 
            early_termination_dist=0.5
        )
        if not subtask_success:
            print(f"Navigation to the frontier point failed.")
        return subtask_success

    def _get_all_closed_doors(self, graph: nx.DiGraph) -> List[Dict]:
        """
        Get all closed door nodes from the graph.
        
        Args:
            graph: Room-object graph
            
        Returns:
            List of closed door node data
        """
        door_nodes = []
        for n, data in graph.nodes(data=True):
            if data.get("node_type") == NODETYPE.ROOM:
                closed_doors = data.get("closed_doors", [])
                for door in closed_doors:
                    door_data = graph.nodes.get(door)
                    if door_data:
                        door_nodes.append(door_data)
        return door_nodes
    
    def _get_all_closed_objects(self, graph: nx.DiGraph) -> List[Dict]:
        """
        Get all closed object nodes from the graph.
        
        Args:
            graph: Room-object graph
            
        Returns:
            List of closed object node data
        """
        closed_object_nodes = []
        for n, data in graph.nodes(data=True):
            if data.get("node_type") == NODETYPE.OBJECT:
                states = data.get("states", {})
                
                # Check if object is closed
                is_closed = False
                
                # Check Open state - handle different formats
                for state_key, state_val in states.items():
                    # Handle different state key formats
                    if (state_key == object_states.Open or 
                        str(state_key).endswith('Open') or
                        getattr(state_key, '__name__', '') == 'Open' or
                        (hasattr(state_key, '__class__') and 
                         state_key.__class__.__name__ in ['Open', 'OpenState'])):
                        if not state_val:  # If Open is False, it's closed
                            is_closed = True
                        break
                else:
                    # No Open state found - check if it's a door
                    if data.get("semantic_class_name") == "door":
                        if n not in self.env.opened_doors:
                            is_closed = True
                
                if is_closed:
                    closed_object_nodes.append(data)
                    
        return closed_object_nodes
    
    def _object_node_to_argument(self, node: Dict) -> str:
        """
        Convert object node to action argument string.
        
        Args:
            node: Object node data
            
        Returns:
            Formatted argument string "room_name, object_name"
        """
        room_id = node.get('room_id')
        if isinstance(room_id, int):
            room_name = f'room_{room_id}'
        else:
            room_name = str(room_id) if room_id else 'unknown'
        
        object_name = self.llm.to_human_readable_object_name(node.get('name', 'object'))
        return f"{room_name}, {object_name}"

    def _action_selection(self, 
                          frontier_points: List, 
                          object_nodes: List[Dict], 
                          closed_door_nodes: List[Dict], 
                          obs: Dict) -> Tuple[Any, str]:
        """
        Select action based on greedy policy (closest point).
        
        Args:
            frontier_points: List of frontier points
            object_nodes: List of closed object nodes
            closed_door_nodes: List of closed door nodes
            obs: Current observation
            
        Returns:
            Tuple of (selected target, target type)
        """
        all_points = frontier_points + object_nodes
        
        if not all_points:
            return None, "none"
            
        closest_idx, _closest_pos, _costs, _paths = self._find_closest_point(all_points)
        
        if closest_idx < len(frontier_points):
            return all_points[closest_idx], "frontier"
        else:
            return all_points[closest_idx], "object"

    def take_action(self, obs: Dict, task_description: str, **kwargs) -> Tuple[bool, bool, Dict]:
        """
        Take action using greedy policy.
        
        Args:
            obs: Current observation
            task_description: Task description string
            **kwargs: Additional arguments (ignored for baseline)
            
        Returns:
            Tuple of (done, task_success, episode_info)
        """
        # Check task success first
        task_success = self.env.task.evaluate_success(self.env) if self.env.task else False
        
        if not task_success:
            graph = obs.get("room_object_graph")
            
            if graph is None:
                self.env.episode_info["failure_reason"] = "no_graph_available"
                return True, False, self.env.episode_info

            # Get frontier points from all rooms
            rooms_with_frontier = [
                graph.nodes.get(n, {}).get("frontier_points", set()) 
                for n in graph.successors("root")
            ]
            frontier_points = list(set().union(*rooms_with_frontier))
            frontier_points = [p[0] for p in frontier_points if isinstance(p, tuple)]

            # Get closed objects and doors
            object_nodes = self._get_all_closed_objects(graph)
            closed_door_nodes = self._get_all_closed_doors(graph)
            
            if len(object_nodes + frontier_points) > 0:
                target, point_type = self._action_selection(
                    frontier_points=frontier_points, 
                    object_nodes=object_nodes, 
                    closed_door_nodes=closed_door_nodes, 
                    obs=obs
                )
                
                if target is None:
                    self.env.episode_info["failure_reason"] = "no_valid_target"
                    nav_success = False
                elif point_type == "object":
                    argument = self._object_node_to_argument(target)
                    nav_success, _done, _feedback, _ = self.execute_action(
                        "go_to_and_open", 
                        argument, 
                        task_desc=task_description, 
                        graph=graph, 
                        vor_graph=obs.get("separated_voronoi_graph")
                    )
                    self.env.f.suptitle(f"{self.env.f._suptitle.get_text()}, go_to_and_open({argument})")
                elif point_type == "frontier":
                    nav_success = self.frontier_navigation(target)
                    self.env.f.suptitle(f"{self.env.f._suptitle.get_text()}, explore({target})")
                else:
                    raise ValueError(f"Unknown point type {point_type}")
            else:
                self.env.episode_info["failure_reason"] = "no_exploration_points_left"
                nav_success = False
                print(f"take_action failed: no closed objects or frontier points left")
            
            self.nav_fails += (not nav_success)
            if self.nav_fails > self.max_nav_fails:
                self.env.episode_info["failure_reason"] = "max_nav_fails_reached"

        done = task_success or (self.env.episode_info.get("failure_reason", None) is not None)
        return done, task_success, self.env.episode_info


class HabitatRandomBaseline(HabitatGreedyBaseline):
    """
    Random baseline agent for Habitat.
    Randomly selects between frontier points and closed objects.
    """
    
    def _action_selection(self, 
                          frontier_points: List, 
                          object_nodes: List[Dict], 
                          closed_door_nodes: List[Dict], 
                          obs: Dict) -> Tuple[Any, str]:
        """
        Select action randomly.
        
        Args:
            frontier_points: List of frontier points
            object_nodes: List of closed object nodes
            closed_door_nodes: List of closed door nodes
            obs: Current observation
            
        Returns:
            Tuple of (selected target, target type)
        """
        all_points = frontier_points + object_nodes
        
        if not all_points:
            return None, "none"
            
        rnd_idx = np.random.randint(0, len(all_points))
        
        if rnd_idx < len(frontier_points):
            return all_points[rnd_idx], "frontier"
        else:
            return all_points[rnd_idx], "object"


# Compatibility aliases
GreedyBaseline = HabitatGreedyBaseline
RandomBaseline = HabitatRandomBaseline

