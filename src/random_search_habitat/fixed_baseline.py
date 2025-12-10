# Fixed Random Baseline for Habitat
# Implements comprehensive random action selection for fair comparison with LLM

from typing import Dict, List, Any, Tuple, Set
import numpy as np

from src.train_from_simulation_habitat.packages.moma_llm.env.habitat_baselines import (
    HabitatGreedyBaseline,
    HabitatRandomBaseline
)
from src.train_from_simulation_habitat.packages.moma_llm.utils.habitat_constants import NODETYPE


# Action types for random selection
class ActionType:
    GOTO_OBJECT = "goto_object"      # Navigate to a visible object
    GOTO_ROOM = "goto_room"          # Navigate to a room
    OPEN = "open"                    # Open a door/container
    EXPLORE = "explore"              # Explore frontier/unexplored area


class FixedHabitatRandomBaseline(HabitatRandomBaseline):
    """
    Comprehensive random baseline for Habitat that randomizes between ALL action types:
    
    1. goto(object) - Navigate to visible objects (sofa, table, etc.)
    2. goto(room) - Navigate to different rooms
    3. open(door/container) - Open doors and containers
    4. explore(frontier) - Navigate to unexplored areas
    
    This provides a fair comparison with the LLM-based approach which uses all these actions.
    """
    
    def __init__(self, env, llm, seed: int) -> None:
        super().__init__(env, llm, seed=seed)
        # Track failed targets to avoid repeating same failing action
        self._failed_targets: Set[str] = set()
        self._max_failures_per_target = 2  # After 2 failures, exclude target
        self._target_failure_counts: Dict[str, int] = {}
        self._visited_objects: Set[str] = set()  # Track visited objects to encourage diversity
    
    def reset(self, *args, **kwargs) -> Dict:
        """Reset baseline state for new episode."""
        self._failed_targets = set()
        self._target_failure_counts = {}
        self._visited_objects = set()
        return super().reset(*args, **kwargs)
    
    def _get_all_visible_objects(self, graph) -> List[Dict]:
        """
        Get all visible objects from the graph that can be navigated to.
        Excludes doors (handled separately) and already visited objects.
        """
        visible_objects = []
        rooms = list(graph.successors("root")) if "root" in graph else []
        
        for room in rooms:
            for obj_node in graph.successors(room):
                node_data = graph.nodes.get(obj_node, {})
                
                # Skip doors (handled by open action)
                if node_data.get("semantic_class_name") == "door":
                    continue
                
                # Skip if already visited recently (encourage diversity)
                if obj_node in self._visited_objects and len(self._visited_objects) < 20:
                    continue
                    
                # Get object position
                pos = node_data.get("pos")
                pos_map = node_data.get("pos_map")
                
                if pos is not None or pos_map is not None:
                    visible_objects.append({
                        "name": obj_node,
                        "room_id": room,
                        "pos": pos,
                        "pos_map": pos_map,
                        "node_data": node_data
                    })
        
        return visible_objects
    
    def _get_all_rooms(self, graph) -> List[Dict]:
        """Get all rooms that can be navigated to."""
        room_list = []
        rooms = list(graph.successors("root")) if "root" in graph else []
        
        for room in rooms:
            room_node = graph.nodes.get(room, {})
            pos_map = room_node.get("pos_map")
            
            if pos_map is not None:
                room_list.append({
                    "name": room,
                    "pos_map": pos_map,
                    "node_data": room_node
                })
        
        return room_list
    
    def _get_all_closed_doors_from_rooms(self, graph) -> List[Dict]:
        """
        Get closed doors directly from room node 'closed_doors' attributes.
        
        In Habitat, doors are stored as string names in the closed_doors set 
        of room nodes, not as separate graph nodes. This method handles that.
        """
        door_nodes = []
        rooms = list(graph.successors("root")) if "root" in graph else []
        
        for room in rooms:
            room_node = graph.nodes.get(room, {})
            closed_doors = room_node.get("closed_doors", set())
            
            for door_name in closed_doors:
                # Skip already opened doors
                if door_name in self.env.opened_doors:
                    continue
                    
                # Try to get door data from graph first
                door_data = graph.nodes.get(door_name, {})
                
                if door_data:
                    # Door exists as a node in the graph
                    door_nodes.append(door_data)
                else:
                    # Door is just a string - try to get from scene objects
                    scene_obj = self.env.scene.objects_by_name.get(door_name)
                    if scene_obj is not None:
                        pos = scene_obj.get_position()
                        door_nodes.append({
                            "name": door_name,
                            "room_id": room,
                            "pos": tuple(pos[:3]),
                            "semantic_class_name": "door",
                        })
                    else:
                        # Create minimal door data
                        door_nodes.append({
                            "name": door_name,
                            "room_id": room,
                            "semantic_class_name": "door",
                        })
        
        return door_nodes
    
    def take_action(self, obs: Dict, task_description: str, **kwargs) -> Tuple[bool, bool, Dict]:
        """
        Take action using comprehensive random policy.
        
        Randomly selects from ALL available action types:
        - goto(object): Navigate to visible objects
        - goto(room): Navigate to different rooms  
        - open(door/container): Open doors and containers
        - explore(frontier): Navigate to unexplored areas
        
        This provides fair comparison with LLM-based approach.
        """
        # Check task success first
        task_success = self.env.task.evaluate_success(self.env) if self.env.task else False
        
        if not task_success:
            graph = obs.get("room_object_graph")
            vor_graph = obs.get("separated_voronoi_graph")
            
            if graph is None:
                self.env.episode_info["failure_reason"] = "no_graph_available"
                return True, False, self.env.episode_info

            # Collect all available action candidates
            rooms = list(graph.successors("root")) if "root" in graph else []
            
            # 1. Frontier points (for explore action)
            rooms_with_frontier = [
                graph.nodes.get(n, {}).get("frontier_points", set()) 
                for n in graph.successors("root")
            ]
            frontier_points = list(set().union(*rooms_with_frontier))
            frontier_points = [p[0] for p in frontier_points if isinstance(p, tuple)]
            
            # 2. Closed doors/objects (for open action)
            object_nodes = self._get_all_closed_objects(graph)
            door_nodes = self._get_all_closed_doors_from_rooms(graph)
            all_closed = []
            for obj in object_nodes + door_nodes:
                obj_name = obj.get('name', '')
                if obj_name not in self._failed_targets:
                    all_closed.append(obj)
            
            # 3. Visible objects (for goto action)
            visible_objects = self._get_all_visible_objects(graph)
            
            # 4. Rooms (for goto room action)
            room_list = self._get_all_rooms(graph)
            
            # Build list of all possible actions with their candidates
            action_candidates = []
            
            # Add explore actions (frontier points)
            for fp in frontier_points:
                action_candidates.append({
                    "type": ActionType.EXPLORE,
                    "target": fp,
                    "name": "frontier"
                })
            
            # Add open actions (closed doors/objects)
            for obj in all_closed:
                action_candidates.append({
                    "type": ActionType.OPEN,
                    "target": obj,
                    "name": obj.get('name', 'object')
                })
            
            # Add goto object actions (visible objects)
            for obj in visible_objects:
                action_candidates.append({
                    "type": ActionType.GOTO_OBJECT,
                    "target": obj,
                    "name": obj.get('name', 'object')
                })
            
            # Add goto room actions (if multiple rooms)
            if len(room_list) > 1:
                for room in room_list:
                    action_candidates.append({
                        "type": ActionType.GOTO_ROOM,
                        "target": room,
                        "name": room.get('name', 'room')
                    })
            
            # Debug output
            print(f"DEBUG RandomBaseline: Available actions:")
            print(f"  - {len(frontier_points)} explore (frontier) options")
            print(f"  - {len(all_closed)} open (door/object) options")
            print(f"  - {len(visible_objects)} goto (object) options")
            print(f"  - {len(room_list)} goto (room) options")
            print(f"  - Total: {len(action_candidates)} action candidates")
            
            nav_success = False
            action_taken = False
            
            if len(action_candidates) > 0:
                # Random selection from all available actions
                rnd_idx = np.random.randint(0, len(action_candidates))
                selected = action_candidates[rnd_idx]
                action_type = selected["type"]
                target = selected["target"]
                target_name = selected["name"]
                
                print(f"DEBUG RandomBaseline: Selected action type '{action_type}' with target '{target_name}'")
                
                if action_type == ActionType.EXPLORE:
                    # Navigate to frontier point - ensure it's a numpy array
                    action_taken = True
                    frontier_point = np.array(target) if isinstance(target, tuple) else target
                    nav_success = self.frontier_navigation(frontier_point)
                    self.env.f.suptitle(f"{self.env.f._suptitle.get_text()}, explore(frontier)")
                    
                elif action_type == ActionType.OPEN:
                    # Open door or container
                    action_taken = True
                    argument = self.llm.to_human_readable_object_name(target_name)
                    
                    nav_success, _done, _feedback, _ = self.execute_action(
                        "open",
                        argument,
                        task_desc=task_description,
                        graph=graph,
                        vor_graph=vor_graph
                    )
                    self.env.f.suptitle(f"{self.env.f._suptitle.get_text()}, open({argument})")
                    
                    # Track failures
                    if not nav_success:
                        self._target_failure_counts[target_name] = self._target_failure_counts.get(target_name, 0) + 1
                        if self._target_failure_counts[target_name] >= self._max_failures_per_target:
                            self._failed_targets.add(target_name)
                            
                elif action_type == ActionType.GOTO_OBJECT:
                    # Navigate to visible object
                    action_taken = True
                    argument = self.llm.to_human_readable_object_name(target_name)
                    
                    nav_success, _done, _feedback, _ = self.execute_action(
                        "goto",
                        argument,
                        task_desc=task_description,
                        graph=graph,
                        vor_graph=vor_graph
                    )
                    self.env.f.suptitle(f"{self.env.f._suptitle.get_text()}, goto({argument})")
                    
                    # Mark as visited
                    self._visited_objects.add(target_name)
                    
                elif action_type == ActionType.GOTO_ROOM:
                    # Navigate to room
                    action_taken = True
                    room_name = target_name
                    
                    nav_success, _done, _feedback, _ = self.execute_action(
                        "goto",
                        room_name,
                        task_desc=task_description,
                        graph=graph,
                        vor_graph=vor_graph
                    )
                    self.env.f.suptitle(f"{self.env.f._suptitle.get_text()}, goto({room_name})")
            
            # Fallback: Random walk if no actions available
            if not action_taken:
                print(f"DEBUG RandomBaseline: No valid actions, trying random walk")
                
                if rooms:
                    current_room = rooms[0]
                    room_node = graph.nodes.get(current_room, {})
                    room_pos = room_node.get("pos_map")
                    
                    if room_pos is not None:
                        target_pos = self.env.slam.voxel2world(np.array(room_pos[:2]))
                        target_pos = target_pos + np.random.uniform(-2, 2, size=2)
                        
                        nav_success = self.navigate_to_point(
                            target_pos,
                            success_thres_dist=2.0,
                            face_target=True,
                            early_termination_dist=0.5
                        )
                        self.env.f.suptitle(f"{self.env.f._suptitle.get_text()}, random_walk")
                        action_taken = True
                
                if not action_taken:
                    self.env.episode_info["failure_reason"] = "no_exploration_points_left"
            
            self.nav_fails += (not nav_success)
            if self.nav_fails > self.max_nav_fails:
                self.env.episode_info["failure_reason"] = "max_nav_fails_reached"

        done = task_success or (self.env.episode_info.get("failure_reason", None) is not None)
        return done, task_success, self.env.episode_info

