# Fixed Greedy Baseline for Habitat
# Implements comprehensive greedy action selection for fair comparison with LLM and random baseline

from typing import Dict, List, Any, Tuple, Set
import numpy as np

from src.train_from_simulation_habitat.packages.moma_llm.env.habitat_baselines import (
    HabitatGreedyBaseline,
    HabitatRandomBaseline
)
from src.train_from_simulation_habitat.packages.moma_llm.utils.habitat_constants import NODETYPE


# Action types for greedy selection
class ActionType:
    GOTO_OBJECT = "goto_object"      # Navigate to a visible object
    GOTO_ROOM = "goto_room"          # Navigate to a room
    OPEN = "open"                    # Open a door/container
    EXPLORE = "explore"              # Explore frontier/unexplored area


class FixedHabitatGreedyBaseline(HabitatGreedyBaseline):
    """
    Comprehensive greedy baseline for Habitat that selects the CLOSEST action target:
    
    1. goto(object) - Navigate to visible objects (sofa, table, etc.)
    2. goto(room) - Navigate to different rooms
    3. open(door/container) - Open doors and containers
    4. explore(frontier) - Navigate to unexplored areas
    
    Unlike random baseline, this always selects the closest target based on path cost.
    This provides a fair comparison with LLM-based and random approaches.
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
    
    def _get_target_position(self, target) -> np.ndarray:
        """
        Extract world position from a target (frontier point, object, or room).
        
        Args:
            target: Target - can be numpy array, tuple (frontier point), or dict with position info
            
        Returns:
            World position as numpy array, or None if not available
        """
        # If target is directly a position array or tuple (frontier point)
        if isinstance(target, np.ndarray):
            return target[:2] if len(target) > 2 else target
        
        # Frontier points are stored as tuples like (x, z)
        if isinstance(target, tuple):
            return np.array(target[:2])
        
        # If target is a dict with position info
        if isinstance(target, dict):
            # If target has pos (world coordinates)
            pos = target.get("pos")
            if pos is not None:
                if isinstance(pos, (list, tuple)):
                    return np.array(pos[:2])  # Take x, z only
                return np.array(pos[:2])
            
            # If target has pos_map (voxel coordinates) - convert to world
            pos_map = target.get("pos_map")
            if pos_map is not None:
                world_pos = self.env.slam.voxel2world(np.array(pos_map[:2]))
                return world_pos
        
        return None
    
    def _compute_path_cost(self, target_pos: np.ndarray) -> float:
        """
        Compute path cost (distance) to a target position.
        
        Args:
            target_pos: Target world position
            
        Returns:
            Path cost (distance), or infinity if unreachable
        """
        if target_pos is None:
            return float('inf')
        
        try:
            # Get current robot position - use robots[0].get_position() like the rest of codebase
            robot_pos_3d = self.env.robots[0].get_position()
            # Extract X, Z for 2D (Habitat uses Y-up coordinate system)
            robot_pos = np.array([robot_pos_3d[0], robot_pos_3d[2]])
            
            # Ensure target is also 2D numpy array
            target_2d = np.array(target_pos[:2]) if hasattr(target_pos, '__len__') else target_pos
            
            # Simple Euclidean distance as heuristic
            # In practice, we could use A* path cost here
            euclidean_dist = np.linalg.norm(target_2d - robot_pos)
            
            return euclidean_dist
        except Exception as e:
            print(f"DEBUG _compute_path_cost: Exception computing cost: {e}")
            return float('inf')
    
    def take_action(self, obs: Dict, task_description: str, **kwargs) -> Tuple[bool, bool, Dict]:
        """
        Take action using comprehensive greedy policy.
        
        Selects the CLOSEST target from ALL available action types:
        - goto(object): Navigate to visible objects
        - goto(room): Navigate to different rooms  
        - open(door/container): Open doors and containers
        - explore(frontier): Navigate to unexplored areas
        
        This provides fair comparison with LLM-based and random approaches.
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
            
            # Build list of all possible actions with their candidates and costs
            action_candidates = []
            
            # Add explore actions (frontier points)
            for fp in frontier_points:
                cost = self._compute_path_cost(fp)
                action_candidates.append({
                    "type": ActionType.EXPLORE,
                    "target": fp,
                    "name": "frontier",
                    "cost": cost
                })
            
            # Add open actions (closed doors/objects)
            for obj in all_closed:
                target_pos = self._get_target_position(obj)
                cost = self._compute_path_cost(target_pos)
                action_candidates.append({
                    "type": ActionType.OPEN,
                    "target": obj,
                    "name": obj.get('name', 'object'),
                    "cost": cost
                })
            
            # Add goto object actions (visible objects)
            for obj in visible_objects:
                target_pos = self._get_target_position(obj)
                cost = self._compute_path_cost(target_pos)
                action_candidates.append({
                    "type": ActionType.GOTO_OBJECT,
                    "target": obj,
                    "name": obj.get('name', 'object'),
                    "cost": cost
                })
            
            # Add goto room actions (if multiple rooms)
            if len(room_list) > 1:
                for room in room_list:
                    target_pos = self._get_target_position(room)
                    cost = self._compute_path_cost(target_pos)
                    action_candidates.append({
                        "type": ActionType.GOTO_ROOM,
                        "target": room,
                        "name": room.get('name', 'room'),
                        "cost": cost
                    })
            
            # Debug output
            print(f"DEBUG GreedyBaseline: Available actions:")
            print(f"  - {len(frontier_points)} explore (frontier) options")
            print(f"  - {len(all_closed)} open (door/object) options")
            print(f"  - {len(visible_objects)} goto (object) options")
            print(f"  - {len(room_list)} goto (room) options")
            print(f"  - Total: {len(action_candidates)} action candidates")
            
            nav_success = False
            action_taken = False
            
            if len(action_candidates) > 0:
                # GREEDY selection: choose the action with minimum cost (closest target)
                action_candidates.sort(key=lambda x: x["cost"])
                selected = action_candidates[0]
                action_type = selected["type"]
                target = selected["target"]
                target_name = selected["name"]
                
                print(f"DEBUG GreedyBaseline: Selected action type '{action_type}' with target '{target_name}' (cost: {selected['cost']:.2f})")
                
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
            
            # Fallback: Navigate to closest room center if no actions available
            if not action_taken:
                print(f"DEBUG GreedyBaseline: No valid actions, navigating to closest room center")
                
                if rooms:
                    # Find closest room
                    min_cost = float('inf')
                    closest_room_pos = None
                    
                    for room in rooms:
                        room_node = graph.nodes.get(room, {})
                        room_pos = room_node.get("pos_map")
                        
                        if room_pos is not None:
                            world_pos = self.env.slam.voxel2world(np.array(room_pos[:2]))
                            cost = self._compute_path_cost(world_pos)
                            if cost < min_cost:
                                min_cost = cost
                                closest_room_pos = world_pos
                    
                    if closest_room_pos is not None:
                        # Add small offset to explore new area
                        target_pos = closest_room_pos + np.random.uniform(-1, 1, size=2)
                        
                        nav_success = self.navigate_to_point(
                            target_pos,
                            success_thres_dist=2.0,
                            face_target=True,
                            early_termination_dist=0.5
                        )
                        self.env.f.suptitle(f"{self.env.f._suptitle.get_text()}, fallback_nav")
                        action_taken = True
                
                if not action_taken:
                    self.env.episode_info["failure_reason"] = "no_exploration_points_left"
            
            self.nav_fails += (not nav_success)
            if self.nav_fails > self.max_nav_fails:
                self.env.episode_info["failure_reason"] = "max_nav_fails_reached"

        done = task_success or (self.env.episode_info.get("failure_reason", None) is not None)
        return done, task_success, self.env.episode_info

