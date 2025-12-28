"""
Action Executor for MoMa-LLM Style Navigation Actions

Converts high-level MoMa-LLM actions (navigate, explore, go_to_and_open, done) to navigation.
Uses smooth waypoint-based navigation similar to MoMa-LLM.

Reference: https://github.com/robot-learning-freiburg/MoMa-LLM
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..utils.actions import Action, HighLevelAction

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    """Result of executing a high-level action."""
    success: bool
    steps_taken: int
    distance_travelled: float
    feedback: str
    final_position: Optional[np.ndarray] = None
    observation: Any = None
    info: Optional[Dict[str, Any]] = None


@dataclass 
class NavigationConfig:
    """Configuration for navigation behavior."""
    max_steps_per_action: int = 150
    collision_distance: float = 0.4
    success_distance: float = 1.5
    explore_distance: float = 3.0
    turn_angle: float = 30.0
    max_turn_angle: float = 0.4  # Max turn angle per step (radians)
    # Replanning settings
    max_stuck_count: int = 5
    replan_on_stuck: bool = True
    # Visited position tracking
    visited_position_radius: float = 1.0
    # Smooth navigation settings
    waypoint_spacing: float = 0.5  # Meters between waypoints
    z_offset: float = 0.05  # Height offset when teleporting


class ActionExecutor:
    """
    Executes high-level actions using smooth waypoint navigation.
    
    Aligned with train_from_simulation_habitat navigation:
    - Uses direct agent teleportation along waypoints (not step-by-step actions)
    - Collects observations at each waypoint for SLAM updates
    - Much faster and smoother navigation
    """
    
    def __init__(self, env, config: Optional[NavigationConfig] = None):
        self.env = env
        self.config = config or NavigationConfig()
        
        # History tracking
        self.action_history: List[Tuple[str, str, bool]] = []
        
        # Visited positions for exploration
        self.visited_positions: List[np.ndarray] = []
        
        # Door tracking
        self.opened_doors: Set[str] = set()
    
    def execute(self, action: HighLevelAction, observation) -> ActionResult:
        """
        Execute a high-level MoMa-LLM style action.
        
        MoMa-LLM action names:
        - navigate(target): Navigate to object or room using smooth teleportation
        - explore(room): Explore unexplored areas in a room  
        - go_to_and_open(object): Navigate to and open a door/container
        - done(): Terminate task
        
        Also supports legacy names (goto, open, stop) for compatibility.
        """
        logger.info(f"Executing {action.name}({action.argument})")
        
        # MoMa-LLM action: done()
        if action.name in ("done", "stop"):
            result = self._execute_done(observation)
        # MoMa-LLM action: navigate()
        elif action.name in ("navigate", "goto"):
            result = self._execute_navigate(action.argument, observation)
        # MoMa-LLM action: explore()
        elif action.name == "explore":
            result = self._execute_explore(observation, room_name=action.argument)
        # MoMa-LLM action: go_to_and_open()
        elif action.name in ("go_to_and_open", "open"):
            result = self._execute_go_to_and_open(action.argument, observation)
        else:
            logger.warning(f"Unknown action: {action.name}")
            result = ActionResult(False, 0, 0.0, f"Unknown action: {action.name}",
                                observation=observation)
        
        self.action_history.append((action.name, action.argument, result.success))
        logger.info(f"Action result: success={result.success}, feedback={result.feedback}")
        return result
    
    def _execute_done(self, observation) -> ActionResult:
        """Execute done() action (MoMa-LLM style)."""
        return ActionResult(True, 0, 0.0, "Task terminated", observation=observation)
    
    # =========================================================================
    # Smooth Navigation (Aligned with train_from_simulation_habitat)
    # =========================================================================
    
    def _get_agent_position_2d(self) -> np.ndarray:
        """Get agent 2D position (X, Z in Habitat's Y-up system)."""
        pos, _ = self.env.simulator.get_agent_state()
        return np.array([pos[0], pos[2]])
    
    def _get_agent_yaw(self) -> float:
        """Get agent yaw angle."""
        _, quat = self.env.simulator.get_agent_state()
        r = R.from_quat(quat)
        return r.as_euler('xyz')[1]  # Y-axis rotation
    
    def _set_agent_state_2d(self, position_2d: np.ndarray, yaw: float) -> bool:
        """
        Set agent position and yaw using smooth teleportation.
        
        Aligned with train_from_simulation_habitat's set_agent_state function.
        Snaps to navmesh for valid positioning.
        Also captures a frame for video recording.
        
        IMPORTANT: Updates the environment's distance_travelled for proper SPL calculation.
        
        Args:
            position_2d: 2D position (x, z)
            yaw: Yaw angle in radians
            
        Returns:
            True if successfully moved
        """
        try:
            import habitat_sim
            
            pathfinder = self.env.simulator.pathfinder
            if pathfinder is None:
                return False
            
            # Get current position for Y coordinate
            old_pos, _ = self.env.simulator.get_agent_state()
            
            # Create 3D position
            new_pos_3d = np.array([position_2d[0], old_pos[1], position_2d[1]])
            
            # Snap to navmesh
            snapped = pathfinder.snap_point(new_pos_3d)
            if np.isnan(snapped[0]):
                return False
            
            # Check if snap moved us too far
            snap_dist = np.linalg.norm(np.array([snapped[0], snapped[2]]) - position_2d)
            if snap_dist > 1.0:
                logger.debug(f"Snap too far ({snap_dist:.2f}m), target not on navmesh")
                return False
            
            # Apply z_offset
            snapped[1] += self.config.z_offset
            
            # Create rotation quaternion
            rotation = R.from_euler('y', yaw).as_quat()
            
            # Set agent state
            self.env.simulator.set_agent_state(snapped, rotation)
            
            # IMPORTANT: Capture frame for video after teleporting
            # This ensures smooth navigation is visible in recorded videos
            self._capture_frame_after_teleport()
            
            # Verify movement and update distance_travelled for SPL calculation
            new_pos, _ = self.env.simulator.get_agent_state()
            movement = np.linalg.norm(new_pos - old_pos)
            
            # Update the environment's episode_info.distance_travelled
            # This is critical for correct SPL calculation with high-level actions
            if hasattr(self.env, 'episode_info') and self.env.episode_info is not None:
                self.env.episode_info.distance_travelled += movement
            
            # Also update the environment's _previous_position for consistency
            if hasattr(self.env, '_previous_position'):
                self.env._previous_position = new_pos.copy()
            
            return movement > 0.01
            
        except Exception as e:
            logger.debug(f"Set agent state failed: {e}")
            return False
    
    def _capture_frame_after_teleport(self):
        """Capture a frame for video recording after teleporting."""
        try:
            # Get current raw observations
            raw_obs = self.env.simulator.get_observations()
            position, rotation = self.env.simulator.get_agent_state()
            
            # Process observation
            obs = self.env.obs_processor.process(
                raw_obs, position, rotation,
                self.env.scene, self.env.simulator.pathfinder
            )
            
            # Store frame using environment's frame storage method
            if hasattr(self.env, '_store_frame'):
                self.env._store_frame(obs)
            elif hasattr(self.env, 'rgb_frames'):
                # Fallback: directly append RGB
                self.env.rgb_frames.append(obs.rgb.copy())
                
        except Exception as e:
            logger.debug(f"Failed to capture frame: {e}")
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def _turn_to_target_yaw(self, target_yaw: float):
        """Turn agent smoothly to face target yaw."""
        current_pos_2d = self._get_agent_position_2d()
        current_yaw = self._get_agent_yaw()
        
        angle_diff = self._normalize_angle(target_yaw - current_yaw)
        
        while abs(angle_diff) > self.config.max_turn_angle:
            current_yaw += np.sign(angle_diff) * self.config.max_turn_angle
            self._set_agent_state_2d(current_pos_2d, current_yaw)
            self._get_observation()  # Update observations
            angle_diff = self._normalize_angle(target_yaw - current_yaw)
        
        if abs(angle_diff) > 0.05:
            self._set_agent_state_2d(current_pos_2d, target_yaw)
            self._get_observation()
    
    def _turn_to_face_point(self, target_pos_2d: np.ndarray):
        """Turn agent to face a target point."""
        current_pos = self._get_agent_position_2d()
        dx = target_pos_2d[0] - current_pos[0]
        dz = target_pos_2d[1] - current_pos[1]
        target_yaw = np.arctan2(-dx, -dz)  # Habitat: -Z is forward
        self._turn_to_target_yaw(target_yaw)
    
    def _navigate_smooth(self, target_pos: np.ndarray, 
                         success_dist: float = 1.5,
                         face_target: bool = True) -> Tuple[bool, float, Any]:
        """
        Smooth navigation to target using waypoint teleportation.
        
        Aligned with train_from_simulation_habitat's drive_to_target_position.
        Uses pathfinder for waypoints and teleports along path.
        
        Args:
            target_pos: Target position (3D or 2D)
            success_dist: Distance threshold for success
            face_target: Whether to face target at end
            
        Returns:
            Tuple of (success, distance_travelled, final_observation)
        """
        import habitat_sim
        
        pathfinder = self.env.simulator.pathfinder
        if pathfinder is None:
            return False, 0.0, self._get_observation()
        
        # Convert to 2D if needed
        target_2d = np.array([target_pos[0], target_pos[2]]) if len(target_pos) == 3 else target_pos
        
        start_pos = self._get_agent_position_2d()
        initial_dist = np.linalg.norm(start_pos - target_2d)
        
        # Get path from pathfinder
        current_pos_3d, _ = self.env.simulator.get_agent_state()
        target_3d = np.array([target_2d[0], current_pos_3d[1], target_2d[1]])
        
        path = habitat_sim.ShortestPath()
        path.requested_start = current_pos_3d
        path.requested_end = target_3d
        
        found = pathfinder.find_path(path)
        
        if not found or path.geodesic_distance == float('inf') or len(path.points) < 2:
            logger.debug(f"No path found to target (dist={initial_dist:.2f}m)")
            return False, 0.0, self._get_observation()
        
        logger.debug(f"Navigating {path.geodesic_distance:.2f}m via {len(path.points)} waypoints")
        
        # Navigate along waypoints
        max_replans = min(int(50 * initial_dist), 250)
        replans = 0
        observation = None
        
        waypoints = [np.array([p[0], p[2]]) for p in path.points]
        
        for i, waypoint in enumerate(waypoints[1:], 1):  # Skip first (current position)
            current_pos = self._get_agent_position_2d()
            
            # Check if we've reached the goal
            dist_to_goal = np.linalg.norm(current_pos - target_2d)
            if dist_to_goal < success_dist:
                break
            
            # Calculate yaw towards waypoint
            wp_yaw = np.arctan2(-(waypoint[0] - current_pos[0]), 
                               -(waypoint[1] - current_pos[1]))
            
            # Turn towards waypoint
            self._turn_to_target_yaw(wp_yaw)
            
            # Move to waypoint
            success = self._set_agent_state_2d(waypoint, wp_yaw)
            observation = self._get_observation()
            
            if not success:
                logger.debug(f"Failed to move to waypoint {i}")
                replans += 1
                if replans > max_replans:
                    break
        
        # Face target if requested
        if face_target:
            self._turn_to_face_point(target_2d)
        
        observation = self._get_observation()
        final_pos = self._get_agent_position_2d()
        final_dist = np.linalg.norm(final_pos - target_2d)
        distance_travelled = np.linalg.norm(final_pos - start_pos)
        
        success = final_dist < success_dist
        
        logger.debug(f"Navigation {'SUCCESS' if success else 'FAILED'}: "
                    f"final_dist={final_dist:.2f}m (threshold={success_dist:.2f}m)")
        
        return success, distance_travelled, observation
    
    def _get_observation(self) -> Any:
        """Get current processed observation."""
        raw_obs = self.env.simulator.get_observations()
        position, rotation = self.env.simulator.get_agent_state()
        return self.env.obs_processor.process(
            raw_obs, position, rotation,
            self.env.scene, self.env.simulator.pathfinder
        )
    
    # =========================================================================
    # High-Level Actions
    # =========================================================================
    
    def _execute_navigate(self, target: str, observation) -> ActionResult:
        """
        Execute navigate() action (MoMa-LLM style).
        
        Navigate to the nearest visible target object using smooth navigation:
        1. Find the nearest visible object matching the target name
        2. Use smooth waypoint navigation to approach it
        """
        target_lower = target.lower().strip()
        start_pos = observation.position.copy()
        
        # Find ALL visible objects matching the target
        target_objects = self._find_all_visible_targets(observation, target_lower)
        
        if not target_objects:
            visible = ", ".join(sorted(observation.visible_objects)[:10]) if observation.visible_objects else "nothing"
            return ActionResult(
                False, 0, 0.0, 
                f"Target '{target}' not visible. Visible: {visible}. Use explore() to discover more.",
                observation=observation
            )
        
        # Sort by distance to find the closest one
        target_objects.sort(key=lambda x: x.distance)
        target_obj = target_objects[0]
        
        logger.info(f"Found {len(target_objects)} '{target}' objects, "
                   f"navigating to nearest at {target_obj.distance:.2f}m")
        
        # Check if already close enough
        if target_obj.distance < self.config.success_distance:
            return ActionResult(
                True, 0, 0.0,
                f"Already near {target} ({target_obj.distance:.1f}m). Use done() to complete task.",
                observation=observation
            )
        
        # Use smooth navigation
        target_pos = target_obj.position
        success, distance_travelled, observation = self._navigate_smooth(
            target_pos, 
            success_dist=self.config.success_distance,
            face_target=True
        )
        
        # Check final distance (target may have been visible from different angle)
        final_target = self._find_visible_target(observation, target_lower)
        if final_target:
            final_dist = final_target.distance
            success = final_dist < self.config.success_distance
        else:
            # Use direct distance to original target position
            final_pos = observation.position
            final_dist = np.linalg.norm(final_pos - target_pos)
        
        feedback = f"Reached {target} ({final_dist:.1f}m)" if success else \
                   f"Navigated towards {target}, now {final_dist:.1f}m away"
        
        return ActionResult(success, 0, distance_travelled, feedback,
                          observation.position, observation=observation)
    
    def _find_all_visible_targets(self, observation, target_lower: str) -> List:
        """Find all visible objects matching target name."""
        if not observation.visible_object_info:
            return []
        
        matches = []
        for obj_info in observation.visible_object_info:
            if target_lower in obj_info.category.lower():
                matches.append(obj_info)
        
        return matches
    
    def _find_visible_target(self, observation, target_lower: str):
        """Find closest visible object matching target name."""
        targets = self._find_all_visible_targets(observation, target_lower)
        if not targets:
            return None
        targets.sort(key=lambda x: x.distance)
        return targets[0]
    
    def _execute_explore(self, observation, room_name: str = "") -> ActionResult:
        """
        Explore unexplored areas, optionally in a specific room.
        
        Uses smooth navigation to reach frontiers.
        """
        start_pos = observation.position.copy()
        discovered_objects = set(observation.visible_objects or [])
        
        self._mark_visited(start_pos)
        
        if room_name:
            logger.info(f"Targeted exploration of room: {room_name}")
        
        # Try frontier-based exploration
        frontier_result = self._explore_to_frontier(observation, room_name=room_name)
        
        if frontier_result is not None:
            nav_distance, observation, frontier_pos = frontier_result
            
            if observation.visible_objects:
                discovered_objects.update(observation.visible_objects)
            
            if nav_distance > 0.5:
                self._mark_visited(observation.position)
                
                # Perform 360° scan at frontier
                observation = self._perform_360_scan()
                
                if observation.visible_objects:
                    discovered_objects.update(observation.visible_objects)
                
                objects_str = ", ".join(sorted(discovered_objects)[:10]) if discovered_objects else "none"
                room_info = f" in {room_name}" if room_name else ""
                feedback = f"Explored frontier{room_info} ({nav_distance:.1f}m), found: {objects_str}"
                
                return ActionResult(True, 0, nav_distance, feedback, observation.position,
                                  observation=observation)
        
        # Fallback: depth-based exploration
        fallback_result = self._explore_by_depth(observation)
        nav_distance, observation, direction = fallback_result
        
        if observation.visible_objects:
            discovered_objects.update(observation.visible_objects)
        
        if nav_distance > 0.5:
            self._mark_visited(observation.position)
        
        # Perform 360° scan
        observation = self._perform_360_scan()
        
        if observation.visible_objects:
            discovered_objects.update(observation.visible_objects)
        
        distance = float(np.linalg.norm(observation.position - start_pos))
        
        if distance < 0.1 and nav_distance < 0.5:
            objects_str = ", ".join(sorted(discovered_objects)[:10]) if discovered_objects else "none"
            room_info = f" in {room_name}" if room_name else ""
            feedback = f"No unexplored areas found{room_info}. Visible: {objects_str}. Use navigate() or go_to_and_open() a door."
            success = False
        else:
            objects_str = ", ".join(sorted(discovered_objects)[:10]) if discovered_objects else "none"
            room_info = f" in {room_name}" if room_name else ""
            feedback = f"Explored {direction}{room_info} ({distance:.1f}m), found: {objects_str}"
            success = True
        
        return ActionResult(success, 0, distance, feedback, observation.position,
                          observation=observation)
    
    def _explore_to_frontier(self, observation, room_name: str = "") -> Optional[Tuple[float, Any, np.ndarray]]:
        """
        Find and navigate to a frontier point using smooth navigation.
        
        Returns:
            Tuple of (distance, observation, frontier_pos) or None if no frontier found
        """
        import habitat_sim
        
        pathfinder = self.env.simulator.pathfinder
        if pathfinder is None:
            return None
        
        current_pos = observation.position
        frontier_candidates = []
        
        for distance in [3.0, 5.0, 7.0, 10.0]:
            for angle_offset in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                yaw = observation.yaw + angle_offset
                sample_x = current_pos[0] - np.sin(yaw) * distance
                sample_z = current_pos[2] - np.cos(yaw) * distance
                sample_pos = np.array([sample_x, current_pos[1], sample_z])
                
                try:
                    nav_point = pathfinder.snap_point(sample_pos)
                    
                    if not pathfinder.is_navigable(nav_point):
                        continue
                    
                    if self._is_visited(nav_point):
                        continue
                    
                    path = habitat_sim.ShortestPath()
                    path.requested_start = current_pos
                    path.requested_end = nav_point
                    found = pathfinder.find_path(path)
                    
                    if not found or path.geodesic_distance == float('inf') or path.geodesic_distance < 1.0:
                        continue
                    
                    frontier_candidates.append((nav_point, path.geodesic_distance))
                    
                except Exception:
                    continue
        
        if not frontier_candidates:
            logger.debug(f"No frontier points found{' in ' + room_name if room_name else ''}")
            return None
        
        # Sort by distance (prefer closer frontiers)
        frontier_candidates.sort(key=lambda x: x[1])
        
        # Try to navigate to the best frontier
        for frontier_pos, score in frontier_candidates[:3]:
            logger.debug(f"Navigating to frontier at distance {score:.1f}m")
            
            start_pos = observation.position.copy()
            
            # Use smooth navigation
            success, distance, observation = self._navigate_smooth(
                frontier_pos, 
                success_dist=1.5,
                face_target=True
            )
            
            if distance > 1.0:
                return (distance, observation, frontier_pos)
        
        return None
    
    def _explore_by_depth(self, observation) -> Tuple[float, Any, str]:
        """Fallback depth-based exploration using smooth navigation."""
        start_pos = observation.position.copy()
        
        directions = self._get_smart_frontier_directions(observation)
        
        for direction in directions:
            # Calculate target position in that direction
            yaw = observation.yaw
            if direction == "left":
                yaw += np.pi / 2
            elif direction == "right":
                yaw -= np.pi / 2
            elif direction == "behind":
                yaw += np.pi
            
            # Project position forward
            dist = self.config.explore_distance
            target_x = observation.position[0] - np.sin(yaw) * dist
            target_z = observation.position[2] - np.cos(yaw) * dist
            target_pos = np.array([target_x, observation.position[1], target_z])
            
            # Check if visited
            if self._is_visited(target_pos):
                continue
            
            # Try smooth navigation
            success, nav_dist, observation = self._navigate_smooth(
                target_pos, success_dist=1.5, face_target=True
            )
            
            if nav_dist > 0.5:
                return (nav_dist, observation, direction)
        
        distance = float(np.linalg.norm(observation.position - start_pos))
        return (distance, observation, "none")
    
    def _execute_go_to_and_open(self, target: str, observation) -> ActionResult:
        """
        Execute go_to_and_open() action (MoMa-LLM style).
        
        Navigate to and open a door/container using smooth navigation.
        """
        target_lower = target.lower().strip()
        start_pos = observation.position.copy()
        
        if target_lower in self.opened_doors:
            return ActionResult(True, 0, 0.0, f"{target} already opened.", observation=observation)
        
        target_obj = self._find_visible_target(observation, target_lower)
        if target_obj is None:
            visible = ", ".join(sorted(observation.visible_objects)[:10]) if observation.visible_objects else "nothing"
            return ActionResult(
                False, 0, 0.0,
                f"Cannot see '{target}'. Visible: {visible}",
                observation=observation
            )
        
        # Navigate to target if it's far away
        distance_travelled = 0.0
        if target_obj.distance > 2.0:
            success, dist, observation = self._navigate_smooth(
                target_obj.position, success_dist=2.0, face_target=True
            )
            distance_travelled = dist
            
            target_obj = self._find_visible_target(observation, target_lower)
            if target_obj is None or target_obj.distance > 3.0:
                dist_str = f"{target_obj.distance:.1f}m" if target_obj else "unknown"
                return ActionResult(
                    False, 0, distance_travelled,
                    f"Cannot reach {target} (still {dist_str} away). Use navigate() to get closer.",
                    observation=observation
                )
        
        self.opened_doors.add(target_lower)
        
        # Perform 360° scan
        observation = self._perform_360_scan()
        
        distance_travelled = float(np.linalg.norm(observation.position - start_pos))
        return ActionResult(
            True, 0, distance_travelled,
            f"Opened {target}. Scanning for newly accessible areas.",
            observation.position, observation=observation
        )
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _perform_360_scan(self) -> Any:
        """Perform 360° scan using smooth rotation."""
        current_pos = self._get_agent_position_2d()
        current_yaw = self._get_agent_yaw()
        observation = None
        
        # Turn in increments
        for i in range(12):
            new_yaw = current_yaw + (i + 1) * (2 * np.pi / 12)
            self._set_agent_state_2d(current_pos, new_yaw)
            observation = self._get_observation()
        
        return observation
    
    def _get_smart_frontier_directions(self, observation) -> List[str]:
        """Get frontier directions ranked by openness."""
        depth = observation.depth
        
        if depth is None or depth.size == 0:
            return ["ahead", "left", "right", "behind"]
        
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
        
        h, w = depth.shape
        mid_start, mid_end = h // 3, 2 * h // 3
        
        regions = {
            "left": depth[mid_start:mid_end, :w//3],
            "ahead": depth[mid_start:mid_end, w//3:2*w//3],
            "right": depth[mid_start:mid_end, 2*w//3:],
        }
        
        scores = {}
        for direction, region in regions.items():
            valid = region[(region > 0.1) & (region < 10.0)]
            openness = float(np.median(valid)) if len(valid) > 0 else 0.0
            
            projected = self._project_position_in_direction(observation, direction, 2.0)
            if self._is_visited(projected):
                openness *= 0.3
            
            scores[direction] = openness
        
        ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        ranked.append("behind")
        return ranked
    
    def _project_position_in_direction(self, observation, direction: str, distance: float) -> np.ndarray:
        """Project position in a given direction."""
        yaw = observation.yaw
        angle_offsets = {"ahead": 0, "left": np.pi/2, "right": -np.pi/2, "behind": np.pi}
        adjusted_yaw = yaw + angle_offsets.get(direction, 0)
        
        pos = observation.position
        forward = np.array([-np.sin(adjusted_yaw), 0, -np.cos(adjusted_yaw)])
        return pos + forward * distance
    
    def _mark_visited(self, position: np.ndarray):
        """Mark a position as visited."""
        pos_2d = np.array([position[0], position[2]])
        self.visited_positions.append(pos_2d)
        
        if len(self.visited_positions) > 100:
            self.visited_positions = self.visited_positions[-50:]
    
    def _is_visited(self, position: np.ndarray) -> bool:
        """Check if a position has been visited."""
        if not self.visited_positions:
            return False
        
        pos_2d = np.array([position[0], position[2]])
        for visited in self.visited_positions:
            if np.linalg.norm(pos_2d - visited) < self.config.visited_position_radius:
                return True
        return False
    
    def reset(self):
        """Reset executor state for new episode."""
        self.action_history = []
        self.visited_positions = []
        self.opened_doors = set()


def create_action_executor(env, config: Optional[NavigationConfig] = None) -> ActionExecutor:
    """Create an action executor."""
    return ActionExecutor(env, config or NavigationConfig())
