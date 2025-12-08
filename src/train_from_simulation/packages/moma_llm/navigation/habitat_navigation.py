# Habitat Navigation Module for SmallPlan
# Provides navigation functions compatible with the original iGibson-based code

import cv2
import numpy as np
import pyastar2d as pyastar
from scipy.spatial.transform import Rotation as R
from typing import Tuple, List, Optional

try:
    import habitat_sim
    from habitat_sim.utils.common import quat_to_magnum, quat_from_magnum
    HABITAT_SIM_AVAILABLE = True
except ImportError:
    HABITAT_SIM_AVAILABLE = False

from moma_llm.utils.constants import MAX_TURN_ANGLE, OCCUPANCY


def get_robot_pos_2d(env) -> np.ndarray:
    """
    Get robot 2D position in Habitat's coordinate system.
    
    Habitat uses Y-up coordinate system:
    - X: horizontal
    - Y: vertical (UP)
    - Z: horizontal
    
    For 2D navigation, we need X and Z (indices 0 and 2).
    """
    pos_3d = env.robots[0].get_position()
    return np.array([pos_3d[0], pos_3d[2]])  # X, Z (horizontal plane)


def cand_poses_around_object(object_pos: np.ndarray, 
                             object_yaw: float, 
                             yaw_extent: float, 
                             radius: float, 
                             num_proposals: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns candidate robot poses on a circle around the object.
    
    Args:
        object_pos: Position of the object in world frame
        object_yaw: Yaw orientation of the object
        yaw_extent: Angular extent to sample poses
        radius: Radius of the circle in meters
        num_proposals: Number of pose proposals
        
    Returns:
        Tuple of (candidate positions, candidate yaws)
    """
    object_pos = np.array(object_pos)[:2]
    angles = normalize_angle_minuspi_pi(
        np.linspace(object_yaw - yaw_extent, object_yaw + yaw_extent, num_proposals)
    )
    cand_pos = object_pos + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    cand_yaw = np.arctan2(object_pos[1] - cand_pos[:, 1], object_pos[0] - cand_pos[:, 0])
    return cand_pos, cand_yaw


def get_circular_kernel(radius: int) -> np.ndarray:
    """Get circular kernel for map dilation."""
    if radius > 2:
        img = np.zeros((2 * radius + 1, 2 * radius + 1))
        center = (radius, radius)
        return cv2.circle(img, center, radius, 1, -1).astype(np.uint8)
    elif radius > 0:
        return np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    else:
        raise ValueError(f"radius {radius} too small")


class PyAstarHelper:
    """Helper class for A* path planning."""
    UNEXPLORED_COST = 500
    OCCUPIED_COST = 10_000

    @staticmethod
    def get_inflated_map_weights(m: np.ndarray, 
                                  inflation_radius_m: float, 
                                  resolution: float) -> np.ndarray:
        """Get inflated map weights for path planning."""
        if inflation_radius_m:
            radius = round(inflation_radius_m / resolution)
            kernel = get_circular_kernel(radius)
            inflated_map = cv2.dilate(m.copy(), kernel, iterations=1)
        else:
            inflated_map = m.copy()
        assert inflated_map.min() >= 0.0 and inflated_map.max() <= 1.0
        return inflated_map.astype(np.float32)
    
    @staticmethod
    def map_to_weights(occupancy_map: np.ndarray, 
                       inflation_radius_m: float, 
                       resolution: float, 
                       add_wall_avoidance_cost: bool) -> np.ndarray:
        """Convert occupancy map to weights for A* planning."""
        binary_occupancy_map = (occupancy_map == OCCUPANCY.OCCUPIED).astype(np.uint8)
        
        weights = PyAstarHelper.UNEXPLORED_COST * PyAstarHelper.get_inflated_map_weights(
            (occupancy_map == OCCUPANCY.UNEXPLORED).astype(np.uint8),
            inflation_radius_m=inflation_radius_m,
            resolution=resolution
        )
        occupied_weights = PyAstarHelper.get_inflated_map_weights(
            binary_occupancy_map,
            inflation_radius_m=inflation_radius_m,
            resolution=resolution
        )
        weights[occupied_weights > 0] = PyAstarHelper.OCCUPIED_COST
        
        if add_wall_avoidance_cost:
            # Add graduated wall avoidance cost - higher cost closer to walls
            # This creates a strong preference for paths that stay away from obstacles
            
            # First ring: very close to obstacles (within inflation + 2 cells) - VERY HIGH COST
            avoid_wall_weights_close = PyAstarHelper.get_inflated_map_weights(
                binary_occupancy_map,
                inflation_radius_m=inflation_radius_m + 2 * resolution,
                resolution=resolution
            )
            weights[avoid_wall_weights_close > 0] += 50  # Very high cost - strongly avoid
            
            # Second ring: close to obstacles (within inflation + 4 cells)
            avoid_wall_weights_near = PyAstarHelper.get_inflated_map_weights(
                binary_occupancy_map,
                inflation_radius_m=inflation_radius_m + 4 * resolution,
                resolution=resolution
            )
            weights[avoid_wall_weights_near > 0] += 20  # High cost near walls
            
            # Third ring: moderate distance (within inflation + 6 cells)
            avoid_wall_weights_medium = PyAstarHelper.get_inflated_map_weights(
                binary_occupancy_map,
                inflation_radius_m=inflation_radius_m + 6 * resolution,
                resolution=resolution
            )
            weights[avoid_wall_weights_medium > 0] += 8  # Medium cost
            
            # Fourth ring: further away (within inflation + 10 cells) - prefer open space
            avoid_wall_weights_far = PyAstarHelper.get_inflated_map_weights(
                binary_occupancy_map,
                inflation_radius_m=inflation_radius_m + 10 * resolution,
                resolution=resolution
            )
            weights[avoid_wall_weights_far > 0] += 2  # Small cost to prefer open areas
            
        weights += 1
        return weights


def find_floor_idx(env, z: float) -> int:
    """Find floor index for given z coordinate."""
    floor_heights = np.array(env.scene.floor_heights)
    return np.argmin(np.abs(floor_heights - z))


def try_incremental_move(env, target_pos: np.ndarray, step_size: float = 0.1, debug: bool = False) -> bool:
    """
    Try to move incrementally towards target position.
    
    Args:
        env: Environment instance
        target_pos: Target 2D position
        step_size: Step size in meters
        debug: Enable debug output
        
    Returns:
        True if robot moved, False otherwise
    """
    if not HABITAT_SIM_AVAILABLE:
        return False
        
    current_pos = get_robot_pos_2d(env)
    direction = target_pos - current_pos
    distance = np.linalg.norm(direction)
    
    if distance < 0.01:
        return False  # Already at target
    
    # Normalize and take a small step
    direction = direction / distance
    step_pos = current_pos + direction * min(step_size, distance)
    
    # Calculate yaw towards target
    _, yaw = cartesian_to_polar(direction[0], direction[1])
    
    # Try to move
    return set_agent_state(env, step_pos, yaw, z_offset=0.05, debug=debug)


def get_habitat_navmesh_path(env, target_pos_world: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
    """
    Get path from Habitat's navmesh pathfinder.
    
    Args:
        env: Environment instance
        target_pos_world: Target position in world coordinates (2D)
        debug: Enable debug output
        
    Returns:
        Path waypoints in world coordinates (Nx2 array) or None if no path found
    """
    if not HABITAT_SIM_AVAILABLE:
        return None
        
    pathfinder = env.sim.pathfinder
    if not pathfinder or not pathfinder.is_loaded:
        return None
    
    # Get current position in 3D
    agent = env.sim.get_agent(0)
    start_pos_3d = np.array(agent.get_state().position)
    
    # Create target 3D position
    target_pos_3d = np.array([target_pos_world[0], start_pos_3d[1], target_pos_world[1]])
    
    # Snap target to navmesh
    target_snapped = pathfinder.snap_point(target_pos_3d)
    if np.isnan(target_snapped[0]):
        if debug:
            print(f"DEBUG NavMesh: Target not on navmesh")
        return None
    
    # Find path using Habitat's pathfinder
    path = habitat_sim.ShortestPath()
    path.requested_start = start_pos_3d
    path.requested_end = target_snapped
    
    found = pathfinder.find_path(path)
    
    if not found or len(path.points) < 2:
        if debug:
            print(f"DEBUG NavMesh: No path found from [{start_pos_3d[0]:.3f}, {start_pos_3d[2]:.3f}] to [{target_snapped[0]:.3f}, {target_snapped[2]:.3f}]")
        return None
    
    # Convert to 2D waypoints (X, Z)
    waypoints_2d = np.array([[p[0], p[2]] for p in path.points])
    
    if debug:
        print(f"DEBUG NavMesh: Found path with {len(waypoints_2d)} waypoints, geodesic_dist={path.geodesic_distance:.3f}m")
    
    return waypoints_2d


def plan_waypoints(env, 
                   target_pos_world: np.ndarray, 
                   inflation_radius_m: float, 
                   filter_collision_points: bool = False, 
                   add_wall_avoidance_cost: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plan waypoints from current robot position to target.
    
    Args:
        env: Environment instance
        target_pos_world: Target position in world coordinates
        inflation_radius_m: Robot inflation radius
        filter_collision_points: Whether to filter collision points
        add_wall_avoidance_cost: Whether to add wall avoidance cost
        
    Returns:
        Tuple of (path in world coordinates, path costs)
    """
    # Use X, Z coordinates for horizontal plane (Habitat Y-up coordinate system)
    start_pos_2d = get_robot_pos_2d(env)
    
    start_pos_pixel = tuple(env.slam.world2voxel(start_pos_2d))
    target_pos_pixel = tuple(env.slam.world2voxel(np.array(target_pos_world[:2])))
    
    occupancy_map = env.slam.bev_map_occupancy
    weights = PyAstarHelper.map_to_weights(
        occupancy_map,
        inflation_radius_m=inflation_radius_m,
        resolution=env.slam.voxel_size,
        add_wall_avoidance_cost=add_wall_avoidance_cost
    )
    
    # Ensure robot position is free
    weights[start_pos_pixel[0] - 1:start_pos_pixel[0] + 2,
            start_pos_pixel[1] - 1:start_pos_pixel[1] + 2] = np.minimum(
        weights[start_pos_pixel[0] - 1:start_pos_pixel[0] + 2,
                start_pos_pixel[1] - 1:start_pos_pixel[1] + 2],
        PyAstarHelper.UNEXPLORED_COST
    )
    weights[start_pos_pixel[0], start_pos_pixel[1]] = 1
    
    path_pixels = pyastar.astar_path(
        weights,
        tuple(start_pos_pixel),
        tuple(target_pos_pixel),
        allow_diagonal=True
    )
    path_world = env.slam.voxel2world(path_pixels)
    costs = weights[path_pixels[:, 0], path_pixels[:, 1]]
    
    if filter_collision_points:
        collision_free = costs.cumsum() < PyAstarHelper.OCCUPIED_COST
        return path_world[collision_free], costs[collision_free]
    else:
        return path_world, costs


def normalize_angle_minuspi_pi(angle: np.ndarray) -> np.ndarray:
    """Normalize angle to [-pi, pi] range."""
    two_pi = 2 * np.pi
    return angle - two_pi * np.floor((angle + np.pi) / two_pi)


def cartesian_to_polar(x: float, y: float) -> Tuple[float, float]:
    """Convert cartesian to polar coordinates."""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def set_agent_state(env, position: np.ndarray, yaw: float, z_offset: float = 0.0, debug: bool = False):
    """
    Set agent position and orientation in Habitat.
    
    Args:
        env: Environment instance
        position: 2D position [x, z] in Habitat's horizontal plane
        yaw: Yaw angle in radians
        z_offset: Vertical Y offset from navmesh (not accumulated)
        debug: Enable debug output
        
    Returns:
        bool: True if position was successfully set, False otherwise
    """
    if not HABITAT_SIM_AVAILABLE:
        return False
    
    # Get current position using the robot wrapper interface
    old_position = env.robots[0].get_position()
    
    # Create 3D position with current Y height as initial guess
    initial_y = old_position[1]
    new_position_3d = np.array([position[0], initial_y, position[1]])
    
    # Try to snap to navmesh to get correct ground height
    pathfinder = env.sim.pathfinder
    if pathfinder and pathfinder.is_loaded:
        # Check if target is navigable
        is_navigable = pathfinder.is_navigable(new_position_3d, max_y_delta=0.5)
        
        # Snap point to navmesh to get correct Y coordinate
        snapped = pathfinder.snap_point(new_position_3d)
        
        if debug:
            snap_dist = np.linalg.norm(np.array([snapped[0], snapped[2]]) - position) if not np.isnan(snapped[0]) else float('inf')
            print(f"DEBUG set_agent_state: target=[{position[0]:.3f}, {position[1]:.3f}], navigable={is_navigable}, snap_dist={snap_dist:.3f}m")
        
        if not np.isnan(snapped[0]):  # Valid snap
            # Check if snap moved us too far (indicates target not on navmesh)
            snap_distance_2d = np.linalg.norm(np.array([snapped[0], snapped[2]]) - position)
            if snap_distance_2d > 0.5:  # Snapped more than 0.5m away
                if debug:
                    print(f"DEBUG set_agent_state: Snap too far ({snap_distance_2d:.3f}m), target not on navmesh")
                return False  # Can't navigate to this point
            else:
                new_position_3d = snapped
                new_position_3d[1] += z_offset
        else:
            if debug:
                print(f"DEBUG set_agent_state: Could not snap to navmesh")
            return False
    
    # Create rotation quaternion from yaw (rotate around Y axis)
    # Try both formats to see which works
    rotation = R.from_euler('y', yaw).as_quat()  # Returns [x, y, z, w]
    
    if debug:
        print(f"DEBUG set_agent_state: Setting position [{new_position_3d[0]:.3f}, {new_position_3d[1]:.3f}, {new_position_3d[2]:.3f}], yaw={yaw:.3f}")
    
    # Use the robot wrapper's set_position_orientation method
    try:
        env.robots[0].set_position_orientation(new_position_3d, rotation)
    except Exception as e:
        if debug:
            print(f"DEBUG set_agent_state: Exception during set_position: {e}")
        # Try with swapped quaternion format [w,x,y,z]
        try:
            rotation_wxyz = np.array([rotation[3], rotation[0], rotation[1], rotation[2]])
            env.robots[0].set_position_orientation(new_position_3d, rotation_wxyz)
            if debug:
                print(f"DEBUG set_agent_state: Succeeded with wxyz format")
        except Exception as e2:
            if debug:
                print(f"DEBUG set_agent_state: Both formats failed: {e2}")
            return False
    
    # Verify the state was actually set
    new_pos = env.robots[0].get_position()
    movement_2d = np.linalg.norm(np.array([new_pos[0], new_pos[2]]) - np.array([old_position[0], old_position[2]]))
    
    if debug:
        if movement_2d < 0.01:
            print(f"DEBUG set_agent_state: Agent didn't move! Old=[{old_position[0]:.3f}, {old_position[2]:.3f}], New=[{new_pos[0]:.3f}, {new_pos[2]:.3f}]")
        else:
            print(f"DEBUG set_agent_state: Moved {movement_2d:.3f}m to [{new_pos[0]:.3f}, {new_pos[2]:.3f}]")
    
    return movement_2d > 0.01


def turn_to_target_point(env, 
                         target_pos_world: np.ndarray, 
                         max_turn_angle: float, 
                         z_offset: float) -> None:
    """
    Turn robot to face target point.
    
    Args:
        env: Environment instance
        target_pos_world: Target position in world coordinates
        max_turn_angle: Maximum turn angle per step
        z_offset: Z offset
    """
    current_pos = get_robot_pos_2d(env)
    _, target_yaw = cartesian_to_polar(
        target_pos_world[0] - current_pos[0],
        target_pos_world[1] - current_pos[1]
    )
    turn_to_target_yaw(env, target_yaw, max_turn_angle, z_offset)


def turn_to_target_yaw(env, 
                       target_yaw: float, 
                       max_turn_angle: float, 
                       z_offset: float) -> None:
    """
    Turn robot to target yaw angle.
    
    Args:
        env: Environment instance
        target_yaw: Target yaw angle
        max_turn_angle: Maximum turn angle per step
        z_offset: Z offset
    """
    current_pos = get_robot_pos_2d(env)
    current_yaw = env.robots[0].get_rpy()[2]
    
    angle_diff = normalize_angle_minuspi_pi(np.array([target_yaw - current_yaw]))[0]
    
    while abs(angle_diff) > max_turn_angle:
        current_yaw += np.sign(angle_diff) * max_turn_angle
        set_agent_state(env, current_pos, current_yaw, z_offset)
        angle_diff = normalize_angle_minuspi_pi(np.array([target_yaw - current_yaw]))[0]
        env.get_state()
        
    if abs(angle_diff) > 0.05:
        set_agent_state(env, current_pos, target_yaw, z_offset)
        env.get_state()


def drive_to_target_position(env,
                             target_pos_world: np.ndarray,
                             inflation_radius_m: float,
                             max_turn_angle: float = MAX_TURN_ANGLE,
                             success_thres_dist: float = 0.5,
                             early_termination_dist: float = 0.0,
                             face_target: bool = True,
                             debug: bool = True) -> bool:
    """
    Drive robot to target position.
    
    Args:
        env: Environment instance
        target_pos_world: Target position in world coordinates
        inflation_radius_m: Robot inflation radius for planning
        max_turn_angle: Maximum turn angle per step
        success_thres_dist: Distance threshold for success
        early_termination_dist: Distance for early termination
        face_target: Whether to face target at the end
        debug: Enable debug output
        
    Returns:
        Whether navigation was successful
    """
    def _calc_dist(point_a, point_b):
        return np.linalg.norm(np.array(point_a)[:2] - np.array(point_b)[:2])
    
    z_offset = 0.05
    current_pos = get_robot_pos_2d(env)
    initial_dist = _calc_dist(current_pos, target_pos_world)
    
    # Check if target is navigable on Habitat's navmesh
    use_navmesh_path = False
    if HABITAT_SIM_AVAILABLE:
        pathfinder = env.sim.pathfinder
        if pathfinder and pathfinder.is_loaded:
            target_3d = np.array([target_pos_world[0], 0.0, target_pos_world[1]])
            target_navigable = pathfinder.is_navigable(target_3d, max_y_delta=0.5)
            
            # Also check start position
            start_3d = np.array([current_pos[0], 0.0, current_pos[1]])
            start_navigable = pathfinder.is_navigable(start_3d, max_y_delta=0.5)
            
            if debug:
                print(f"DEBUG Nav: NavMesh check - start_navigable={start_navigable}, target_navigable={target_navigable}")
            
            # If target not directly navigable, try to snap to closest navigable point
            if not target_navigable:
                snapped_target = pathfinder.snap_point(target_3d)
                if not np.isnan(snapped_target[0]):
                    snap_dist = np.linalg.norm(np.array([snapped_target[0], snapped_target[2]]) - target_pos_world[:2])
                    if debug:
                        print(f"DEBUG Nav: Target snapped to navmesh, distance={snap_dist:.3f}m")
                    # Only use snapped target if reasonably close
                    if snap_dist < success_thres_dist:
                        target_pos_world = np.array([snapped_target[0], snapped_target[2]])
                        if debug:
                            print(f"DEBUG Nav: Using snapped target: [{target_pos_world[0]:.3f}, {target_pos_world[1]:.3f}]")
    
    # Try Habitat's navmesh path first (it's usually more reliable)
    navmesh_waypoints = get_habitat_navmesh_path(env, target_pos_world, debug=debug)
    if navmesh_waypoints is not None and len(navmesh_waypoints) > 1:
        waypoints = navmesh_waypoints
        costs = np.ones(len(waypoints))  # All waypoints are navigable
        use_navmesh_path = True
        if debug:
            print(f"DEBUG Nav: Using Habitat navmesh path ({len(waypoints)} waypoints)")
    else:
        # Fallback to A* planning
        waypoints, costs = plan_waypoints(
            env=env,
            target_pos_world=target_pos_world,
            inflation_radius_m=inflation_radius_m,
            filter_collision_points=True,
            add_wall_avoidance_cost=True
        )
        if debug:
            print(f"DEBUG Nav: Using A* path ({len(waypoints)} waypoints)")
    
    max_replans = max(min(int(50 * initial_dist), 250), 50)
    replans = 0
    prev_pos = current_pos.copy()
    termination_reason = "completed"  # Track why navigation ended
    stuck_counter = 0  # Count consecutive stuck iterations
    max_stuck_iterations = 5  # Allow 5 consecutive stuck iterations before giving up
    
    # DEBUG: Log initial navigation state
    if debug:
        print(f"DEBUG Nav: Starting navigation")
        print(f"DEBUG Nav:   Start pos: [{current_pos[0]:.3f}, {current_pos[1]:.3f}]")
        print(f"DEBUG Nav:   Target pos: [{target_pos_world[0]:.3f}, {target_pos_world[1]:.3f}]")
        print(f"DEBUG Nav:   Initial distance: {initial_dist:.3f}m")
        print(f"DEBUG Nav:   Success threshold: {success_thres_dist:.3f}m")
        print(f"DEBUG Nav:   Initial waypoints: {len(waypoints)}")
        print(f"DEBUG Nav:   Using navmesh path: {use_navmesh_path}")
        if len(waypoints) > 0 and not use_navmesh_path:
            max_cost = costs.max() if len(costs) > 0 else 0
            occupied_waypoints = np.sum(costs >= PyAstarHelper.OCCUPIED_COST)
            unexplored_waypoints = np.sum((costs >= PyAstarHelper.UNEXPLORED_COST) & (costs < PyAstarHelper.OCCUPIED_COST))
            print(f"DEBUG Nav:   Path max cost: {max_cost:.0f} (occupied={occupied_waypoints}, unexplored={unexplored_waypoints})")
    
    # Check if path planning returned empty or very short path
    if len(waypoints) <= 1:
        termination_reason = "no_valid_path"
        if debug:
            print(f"DEBUG Nav: No valid path found (waypoints={len(waypoints)})")
    
    consecutive_failed_moves = 0
    max_failed_moves = 3  # Try multiple waypoints before giving up on this path
    
    loop_iteration = 0
    while len(waypoints) > 1:
        loop_iteration += 1
        current_pos = get_robot_pos_2d(env)
        next_wp, next_wp_cost = waypoints[1], costs[1]
        remaining_waypoints = len(waypoints)
        waypoints, costs = waypoints[2:], costs[2:]
        
        if debug:
            wp_dist = np.linalg.norm(next_wp - current_pos)
            print(f"DEBUG Nav: Loop {loop_iteration}: Moving to waypoint [{next_wp[0]:.3f}, {next_wp[1]:.3f}], dist={wp_dist:.3f}m, remaining={remaining_waypoints-2}")
        
        # Allow navigation through unexplored areas (cost < OCCUPIED_COST)
        # Only block truly occupied cells (walls, obstacles)
        next_wp_free = next_wp_cost < PyAstarHelper.OCCUPIED_COST
        
        if next_wp_free:
            # Turn towards next waypoint
            _, next_wp_yaw = cartesian_to_polar(
                next_wp[0] - current_pos[0],
                next_wp[1] - current_pos[1]
            )
            turn_to_target_yaw(env, next_wp_yaw, max_turn_angle, z_offset)
            
            # Move to next waypoint
            move_success = set_agent_state(env, next_wp, next_wp_yaw, z_offset, debug=debug)
            env.get_state()
            
            new_pos = get_robot_pos_2d(env)
            actual_move_dist = np.linalg.norm(new_pos - current_pos)
            if debug:
                print(f"DEBUG Nav: Loop {loop_iteration}: Move result={move_success}, actual_dist={actual_move_dist:.3f}m, new_pos=[{new_pos[0]:.3f}, {new_pos[1]:.3f}]")
            
            if not move_success:
                consecutive_failed_moves += 1
                if debug:
                    print(f"DEBUG Nav: Failed to move to waypoint [{next_wp[0]:.3f}, {next_wp[1]:.3f}] (attempt {consecutive_failed_moves}/{max_failed_moves})")
                
                # Try skipping to next waypoint, or replan if too many failures
                if consecutive_failed_moves >= max_failed_moves:
                    if debug:
                        print(f"DEBUG Nav: Too many failed moves, forcing replan...")
                    consecutive_failed_moves = 0
                    break  # Force replan
            else:
                consecutive_failed_moves = 0  # Reset on successful move
        else:
            if debug:
                print(f"DEBUG Nav: Waypoint blocked by obstacle (cost={next_wp_cost:.0f}), replanning...")
            # Don't consume more waypoints if current one is blocked, trigger replan immediately
            break
            
        # Check early termination
        current_dist = _calc_dist(get_robot_pos_2d(env), target_pos_world)
        if early_termination_dist and current_dist <= early_termination_dist:
            termination_reason = "early_termination_reached"
            if debug:
                print(f"DEBUG Nav: Early termination at dist={current_dist:.3f}m")
            break
        
        # Check if we've reached the goal within success threshold
        if current_dist <= success_thres_dist:
            termination_reason = "goal_reached"
            if debug:
                print(f"DEBUG Nav: Goal reached at dist={current_dist:.3f}m")
            break
        
        # Only replan if we're running low on waypoints (less than 3 remaining)
        # This avoids expensive replanning after every single move
        if len(waypoints) <= 3:
            waypoints, costs = plan_waypoints(
                env=env,
                target_pos_world=target_pos_world,
                inflation_radius_m=inflation_radius_m,
                filter_collision_points=True,
                add_wall_avoidance_cost=True
            )
            replans += 1
            if debug:
                print(f"DEBUG Nav: Replanned path, now {len(waypoints)} waypoints")
        
        # Check if we're stuck
        new_pos = get_robot_pos_2d(env)
        movement = np.linalg.norm(new_pos - prev_pos)
        
        if movement < 0.01:
            stuck_counter += 1
            if debug and stuck_counter > 1:
                print(f"DEBUG Nav: Limited movement ({movement:.4f}m), stuck counter: {stuck_counter}/{max_stuck_iterations}")
        else:
            stuck_counter = 0  # Reset counter if robot moved
            
        if stuck_counter >= max_stuck_iterations:
            # Try using Habitat's navmesh pathfinder as fallback
            if debug:
                print(f"DEBUG Nav: Stuck after {stuck_counter} iterations, trying Habitat navmesh fallback...")
            
            navmesh_path = get_habitat_navmesh_path(env, target_pos_world, debug=debug)
            
            if navmesh_path is not None and len(navmesh_path) > 1:
                # Use navmesh path instead
                waypoints = navmesh_path
                costs = np.ones(len(waypoints))  # Assume all costs are 1 (free)
                stuck_counter = 0  # Reset stuck counter
                if debug:
                    print(f"DEBUG Nav: Switching to navmesh path with {len(waypoints)} waypoints")
                continue  # Try with new path
            else:
                # Last resort: try incremental movement towards target
                if debug:
                    print(f"DEBUG Nav: Navmesh fallback failed, trying incremental movement...")
                
                moved = try_incremental_move(env, target_pos_world[:2], step_size=0.2, debug=debug)
                if moved:
                    stuck_counter = 0  # Reset if we managed to move
                    if debug:
                        print(f"DEBUG Nav: Incremental move succeeded, continuing...")
                    # Re-plan from new position
                    waypoints, costs = plan_waypoints(
                        env=env,
                        target_pos_world=target_pos_world,
                        inflation_radius_m=inflation_radius_m,
                        filter_collision_points=True,
                        add_wall_avoidance_cost=True
                    )
                    continue
                
                termination_reason = "stuck"
                if debug:
                    print(f"DEBUG Nav: Robot stuck at [{new_pos[0]:.3f}, {new_pos[1]:.3f}] after {replans} replans")
                    print(f"DEBUG Nav: All movement attempts failed")
                break
            
        prev_pos = new_pos
        
        if replans > max_replans:
            termination_reason = "max_replans"
            if debug:
                print(f"DEBUG Nav: Max replans ({max_replans}) exceeded")
            break
            
    # Turn to face target
    if face_target:
        turn_to_target_point(env, target_pos_world, max_turn_angle, z_offset)
    
    final_pos = get_robot_pos_2d(env)
    final_dist = _calc_dist(final_pos, target_pos_world)
    success = final_dist <= success_thres_dist
    
    # DEBUG: Log final navigation state
    if debug:
        print(f"DEBUG Nav: Navigation {'SUCCESS' if success else 'FAILED'}")
        print(f"DEBUG Nav:   Final pos: [{final_pos[0]:.3f}, {final_pos[1]:.3f}]")
        print(f"DEBUG Nav:   Final distance: {final_dist:.3f}m (threshold: {success_thres_dist:.3f}m)")
        print(f"DEBUG Nav:   Distance traveled: {initial_dist - final_dist:.3f}m")
        print(f"DEBUG Nav:   Total replans: {replans}")
        print(f"DEBUG Nav:   Termination reason: {termination_reason}")
        
    return success


def turn_full_circle(env) -> Tuple[dict, bool]:
    """
    Turn robot in a full circle to observe surroundings.
    
    Args:
        env: Environment instance
        
    Returns:
        Tuple of (observation, success)
    """
    current_pos = get_robot_pos_2d(env)
    z_offset = 0.05
    max_turn_angle = MAX_TURN_ANGLE
    
    i, max_i = 0, 500
    current_yaw = env.robots[0].get_rpy()[2]
    dist_travelled = 0.0
    
    while dist_travelled < 2 * np.pi:
        new_yaw = current_yaw + max_turn_angle
        set_agent_state(env, current_pos, new_yaw, z_offset)
        obs = env.get_state()
        
        dist_travelled += max_turn_angle
        current_yaw = new_yaw
        
        i += 1
        if i >= max_i:
            print("Robot failed to turn full circle")
            break
            
    success = i < max_i
    return obs, success

