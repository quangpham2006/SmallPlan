# =============================================================================
# HIGH-LEVEL ACTION PROMPTS (Aligned with train_from_simulation_habitat v2 prompts)
# =============================================================================

HIGH_LEVEL_SYSTEM_PROMPT = """You are a robot navigating an unexplored house. Your task is to {task_description}.

Available actions:
{action_descriptions}

Response format (follow STRICTLY):
Analysis: Brief assessment of current situation and where the target might be.
Reasoning: Why this specific action is the best choice right now.
Command: function_name(argument)

Important rules:
- If you see the target object in the visible objects list, immediately call stop(). You do not need to navigate to it or interact with it.
- Learn from failed actions - don't repeat the same failing action.
- When stuck, try opening doors to discover new rooms.
- Use exact object/room names as shown in the visible objects or discovered rooms list.
"""

HIGH_LEVEL_USER_PROMPT = """Current location: {current_room}

=== VISIBLE OBJECTS ===
{nearby_objects}

=== DISCOVERED ROOMS & OBJECTS ===
{discovered_rooms}

=== ACTION HISTORY ===
{action_history}
Last action feedback: {last_feedback}

=== EXPLORATION STATUS ===
Unexplored directions: {unexplored_info}
{exploration_status}

=== DECISION GUIDANCE ===
{decision_guidance}

=== TARGET ===
Find the {target_object}

What action should you take next?"""


# =============================================================================
# LOW-LEVEL ACTION PROMPTS
# =============================================================================

LOW_LEVEL_SYSTEM_PROMPT = """You are a robot navigating an indoor environment. Your task is to {task_description}.

Available actions:
{action_descriptions}

Response format (follow strictly):
Analysis: Brief assessment of current situation and where the target might be.
Reasoning: Why this specific action is the best choice right now.
Action: ACTION_NAME

Important rules:
- Output exactly ONE action per response.
- If you see the target object nearby, navigate towards it using MOVE_FORWARD and turns.
- Use TURN_LEFT or TURN_RIGHT to look around and find the target.
- Use MOVE_BACKWARD if you need to back away from obstacles.
- Call STOP only when you are close to the target object.
"""

LOW_LEVEL_USER_PROMPT = """Current location: {current_room}

=== VISIBLE OBJECTS ===
{nearby_objects}

=== ROOMS DISCOVERED ===
{discovered_rooms}

=== EXPLORATION STATUS ===
Unexplored directions: {unexplored_info}
{exploration_status}

=== RECENT ACTIONS ===
{action_history}
Last action feedback: {last_feedback}

=== TARGET ===
Find the {target_object}

What is your next action?"""


# =============================================================================
# ACTION HISTORY FORMATTING (from train_from_simulation_habitat prompts_v2.py)
# =============================================================================

# Template for successful actions
ACTION_SUCCESS_TEMPLATE = "✓ {action}({argument})"

# Template for failed actions with reason
ACTION_FAILED_TEMPLATE = "✗ {action}({argument}) - FAILED: {reason}"

# When no actions have been taken yet
NO_HISTORY_MESSAGE = "No actions taken yet."

# Header for action history section
ACTION_HISTORY_HEADER = "Recent actions (newest first):"


# =============================================================================
# DECISION GUIDANCE TEMPLATES - Dynamic based on state
# =============================================================================

# When target object is found
GUIDANCE_TARGET_FOUND = """🎯 TARGET FOUND! The target "{target}" is in the visible objects list.
→ Immediately call stop() to complete the task. You do not need to navigate to it or interact with it."""

# When there are repeated failures
GUIDANCE_REPEATED_FAILURES = """⚠️ Recent actions have been failing repeatedly.
→ Try a DIFFERENT approach: open a door, explore a different room, or use a different target.
→ If you struggle too long and cannot find the target, call stop() to terminate the task."""

# When all rooms are explored but target not found
GUIDANCE_ALL_EXPLORED = """All discovered rooms have been fully explored but target not found.
→ Focus on opening closed doors to discover new rooms that may contain the target."""

# When there are unexplored areas
GUIDANCE_UNEXPLORED = """There are still unexplored areas in discovered rooms.
→ Consider exploring rooms with unexplored areas OR opening doors to find new rooms."""

# Default guidance
GUIDANCE_DEFAULT = """Choose the most efficient action to find the target:
- If target is in visible objects → goto(target)
- If target might be in unexplored areas → explore(room)
- If target might be behind closed doors → open(door)"""


# =============================================================================
# RETRY/FAILURE PROMPTS
# =============================================================================

# When last action failed - includes specific feedback
RETRIAL_PROMPT = """The last action failed: {failure_reason}

Please choose a different action. Consider:
- Trying a different target (door, room, or object)
- Using a different action type
- The target might be inaccessible from your current position

Remember to include "Command:" before your action."""

# Generic retry without specific reason
RETRIAL_PROMPT_GENERIC = """The last action failed. Please try a different approach.

Tips:
- Don't repeat the exact same action that just failed
- Try a different target or action type
- Check if there are alternative paths or objects

Remember to include "Command:" before your action."""

# Format/parsing error
RETRIAL_PROMPT_FORMAT_ERROR = """Your response could not be parsed due to format errors.

Required format:
Analysis: [your analysis]
Reasoning: [your reasoning]  
Command: function_name(argument)

Use only available functions with exact object/room names from the list above."""


# =============================================================================
# ROOM CLASSIFICATION PROMPTS (shared)
# =============================================================================

ROOM_CLASSIFICATION_SYSTEM_PROMPT = """You are a helpful assistant identifying room types in an apartment based on their contents."""

ROOM_CLASSIFICATION_USER_PROMPT = """You observe {num_rooms} rooms containing the following objects:
{room_object_list}

{request}

Respond with a bullet list in this exact format:
 - room-X: room type

{remember}
Include ONLY the bullet list, no other text."""


# =============================================================================
# HELPER FUNCTIONS (from train_from_simulation_habitat prompts_v2.py)
# =============================================================================

def format_action_history(action_history: list, max_items: int = 5) -> str:
    """
    Format action history with success/failure status and failure reasons.
    
    Args:
        action_history: List of tuples (action_name, argument, success, feedback)
        max_items: Maximum number of history items to show
        
    Returns:
        Formatted string for the prompt
    """
    if not action_history:
        return NO_HISTORY_MESSAGE
    
    # Take most recent actions (reversed so newest first)
    recent = action_history[-max_items:][::-1]
    
    lines = [ACTION_HISTORY_HEADER]
    for item in recent:
        if len(item) >= 4:
            action, argument, success, feedback = item[:4]
        elif len(item) >= 3:
            action, argument, success = item[:3]
            feedback = "unknown error"
        else:
            continue
        
        if success:
            lines.append(ACTION_SUCCESS_TEMPLATE.format(action=action, argument=argument))
        else:
            reason = feedback if feedback else "unknown error"
            lines.append(ACTION_FAILED_TEMPLATE.format(action=action, argument=argument, reason=reason))
    
    return '\n'.join(lines)


def get_decision_guidance(
    target_found: bool = False,
    target_name: str = "",
    recent_failure_count: int = 0,
    has_unexplored_areas: bool = True,
    all_rooms_explored: bool = False
) -> str:
    """
    Generate contextual decision guidance based on current state.
    
    Args:
        target_found: Whether the target object has been found in visible objects
        target_name: Name of the target object
        recent_failure_count: Number of consecutive recent failures
        has_unexplored_areas: Whether there are still unexplored areas
        all_rooms_explored: Whether all discovered rooms are fully explored
        
    Returns:
        Guidance string for the prompt
    """
    if target_found:
        return GUIDANCE_TARGET_FOUND.format(target=target_name)
    
    if recent_failure_count >= 3:
        return GUIDANCE_REPEATED_FAILURES
    
    if all_rooms_explored:
        return GUIDANCE_ALL_EXPLORED
    
    if has_unexplored_areas:
        return GUIDANCE_UNEXPLORED
    
    return GUIDANCE_DEFAULT


def format_retry_prompt(failure_reason: str = None) -> str:
    """
    Generate retry prompt with optional failure reason.
    
    Args:
        failure_reason: Specific reason for the failure (optional)
        
    Returns:
        Formatted retry prompt string
    """
    if failure_reason:
        return RETRIAL_PROMPT.format(failure_reason=failure_reason)
    return RETRIAL_PROMPT_GENERIC


def count_recent_failures(action_history: list, lookback: int = 5) -> int:
    """
    Count consecutive failures in recent action history.
    
    Args:
        action_history: List of tuples (action, argument, success, feedback)
        lookback: Number of recent actions to check
        
    Returns:
        Count of consecutive failures from the most recent action
    """
    if not action_history:
        return 0
    
    count = 0
    for item in reversed(action_history[-lookback:]):
        success = item[2] if len(item) > 2 else True
        if not success:
            count += 1
        else:
            break  # Stop at first success
    return count


def check_target_in_objects(target_name: str, visible_objects: list) -> bool:
    """
    Check if target object is in the visible objects list.
    
    Args:
        target_name: Name of target to find (can be partial match)
        visible_objects: List of visible object names or categories
        
    Returns:
        True if target is found
    """
    if not target_name or not visible_objects:
        return False
    
    target_lower = target_name.lower()
    for obj in visible_objects:
        obj_name = obj if isinstance(obj, str) else getattr(obj, 'category', str(obj))
        if target_lower in obj_name.lower():
            return True
    return False


def format_discovered_rooms(
    discovered_rooms: dict,
    room_exploration_status: dict = None
) -> str:
    """
    Format discovered rooms with exploration status.
    
    Args:
        discovered_rooms: Dict mapping room display name -> list of object names
        room_exploration_status: Dict mapping room display name -> is_fully_explored
        
    Returns:
        Formatted string for the prompt with human-readable names (no IDs)
    """
    if not discovered_rooms:
        return "  - No rooms discovered yet. Explore to discover rooms."
    
    room_exploration_status = room_exploration_status or {}
    
    fully_explored_rooms = []
    unexplored_rooms = []
    
    for room_name, objects in sorted(discovered_rooms.items()):
        # Skip rooms without objects
        if not objects:
            continue
        
        # Sort and deduplicate objects, show max 10
        unique_objects = sorted(set(objects))[:10]
        obj_list = ", ".join(unique_objects)
        
        is_fully_explored = room_exploration_status.get(room_name, False)
        
        if is_fully_explored:
            fully_explored_rooms.append(f"  - {room_name}: [{obj_list}] ✓ fully explored")
        else:
            unexplored_rooms.append(f"  - {room_name}: [{obj_list}] (has unexplored areas)")
    
    # Format with unexplored rooms first (more relevant for decision making)
    parts = []
    if unexplored_rooms:
        parts.append("📍 Rooms with unexplored areas:\n" + "\n".join(unexplored_rooms))
    if fully_explored_rooms:
        parts.append("✓ Fully explored rooms:\n" + "\n".join(fully_explored_rooms))
    
    return "\n\n".join(parts) if parts else "  - No rooms discovered yet. Explore to discover rooms."


# =============================================================================
# ALIASES FOR BACKWARD COMPATIBILITY
# =============================================================================

# Default to high-level prompts for the main import
SYSTEM_PROMPT = HIGH_LEVEL_SYSTEM_PROMPT
USER_PROMPT = HIGH_LEVEL_USER_PROMPT
