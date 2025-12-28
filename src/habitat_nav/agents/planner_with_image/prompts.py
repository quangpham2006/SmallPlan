"""
Prompts for Image-Enhanced Planner Agent

This module contains prompts for an LLM agent that uses visual context
from images to make better navigation decisions. The agent receives
recent images from its camera to understand the visual scene.
"""

from typing import List, Dict, Optional

# =============================================================================
# IMAGE PLANNER - SYSTEM PROMPT
# =============================================================================

IMAGE_PLANNER_SYSTEM_PROMPT = """You are a robot navigating an unexplored house. Your task is to {task_description}.

You are provided with images showing your current view and recent views from different angles.
Use these images to understand your environment, identify objects, and plan your navigation.

Available actions:
{action_descriptions}

Response format (follow strictly):
Visual Analysis: What do you see in the images? Identify key objects, rooms, and paths.
Analysis: Brief assessment of current situation and where the target might be.
Reasoning: Why this specific action is the best choice based on what you see.
Command: function_name(argument)

Important rules:
- If you see the target object in the visible objects list, immediately call stop(). You do not need to navigate to it or interact with it.
- Use visual cues from images to identify rooms and objects.
- Learn from failed actions - don't repeat the same failing action.
- When stuck, try opening doors to discover new rooms.
- Use exact object/room names as shown in the visible objects list.
"""

# =============================================================================
# IMAGE PLANNER - USER PROMPT (with image context)
# =============================================================================

IMAGE_PLANNER_USER_PROMPT = """Current location: {current_room}

=== IMAGES ===
[You are provided with {num_images} image(s) showing your current view and recent views]

=== VISIBLE OBJECTS (from detection) ===
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

Based on the images and the information above, what action should you take next?"""

# =============================================================================
# ACTION HISTORY FORMATTING
# =============================================================================

ACTION_SUCCESS_TEMPLATE = "✓ {action}({argument})"
ACTION_FAILED_TEMPLATE = "✗ {action}({argument}) - FAILED: {reason}"
NO_HISTORY_MESSAGE = "No actions taken yet."
ACTION_HISTORY_HEADER = "Recent actions (newest first):"

# =============================================================================
# DECISION GUIDANCE TEMPLATES
# =============================================================================

GUIDANCE_TARGET_FOUND = """🎯 TARGET FOUND! The target "{target}" is in the visible objects list.
→ Immediately call stop() to complete the task. You do not need to navigate to it or interact with it."""

GUIDANCE_REPEATED_FAILURES = """⚠️ Recent actions have been failing repeatedly.
→ Look at the images carefully - you may be stuck or facing a wall.
→ Try a DIFFERENT approach: open a door, explore a different room, or turn around.
→ If you struggle too long and cannot find the target, call stop() to terminate the task."""

GUIDANCE_ALL_EXPLORED = """All discovered rooms have been fully explored but target not found.
→ Look at the images for any doors or passages you might have missed.
→ Focus on opening closed doors to discover new rooms that may contain the target."""

GUIDANCE_UNEXPLORED = """There are still unexplored areas in discovered rooms.
→ Use the images to identify promising directions to explore.
→ Consider exploring rooms with unexplored areas OR opening doors to find new rooms."""

GUIDANCE_DEFAULT = """Choose the most efficient action to find the target:
- If you see the target in images → goto(target)
- If you see unexplored areas → explore(room)
- If you see closed doors → open(door)
- Use visual cues to make informed decisions."""

# =============================================================================
# ROOM CLASSIFICATION PROMPTS
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
# HELPER FUNCTIONS
# =============================================================================

def format_action_history(action_history: list, max_items: int = 5) -> str:
    """Format action history with success/failure status."""
    if not action_history:
        return NO_HISTORY_MESSAGE
    
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
    """Generate contextual decision guidance."""
    if target_found:
        return GUIDANCE_TARGET_FOUND.format(target=target_name)
    
    if recent_failure_count >= 3:
        return GUIDANCE_REPEATED_FAILURES
    
    if all_rooms_explored:
        return GUIDANCE_ALL_EXPLORED
    
    if has_unexplored_areas:
        return GUIDANCE_UNEXPLORED
    
    return GUIDANCE_DEFAULT


def count_recent_failures(action_history: list, lookback: int = 5) -> int:
    """Count consecutive failures in recent action history."""
    if not action_history:
        return 0
    
    count = 0
    for item in reversed(action_history[-lookback:]):
        success = item[2] if len(item) > 2 else True
        if not success:
            count += 1
        else:
            break
    return count


def check_target_in_objects(target_name: str, visible_objects: list) -> bool:
    """Check if target object is in the visible objects list."""
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
    """Format discovered rooms with exploration status."""
    if not discovered_rooms:
        return "  - No rooms discovered yet. Explore to discover rooms."
    
    room_exploration_status = room_exploration_status or {}
    
    fully_explored_rooms = []
    unexplored_rooms = []
    
    for room_name, objects in sorted(discovered_rooms.items()):
        if not objects:
            continue
        
        unique_objects = sorted(set(objects))[:10]
        obj_list = ", ".join(unique_objects)
        
        is_fully_explored = room_exploration_status.get(room_name, False)
        
        if is_fully_explored:
            fully_explored_rooms.append(f"  - {room_name}: [{obj_list}] ✓ fully explored")
        else:
            unexplored_rooms.append(f"  - {room_name}: [{obj_list}] (has unexplored areas)")
    
    parts = []
    if unexplored_rooms:
        parts.append("📍 Rooms with unexplored areas:\n" + "\n".join(unexplored_rooms))
    if fully_explored_rooms:
        parts.append("✓ Fully explored rooms:\n" + "\n".join(fully_explored_rooms))
    
    return "\n\n".join(parts) if parts else "  - No rooms discovered yet. Explore to discover rooms."


__all__ = [
    'IMAGE_PLANNER_SYSTEM_PROMPT',
    'IMAGE_PLANNER_USER_PROMPT',
    'ACTION_SUCCESS_TEMPLATE',
    'ACTION_FAILED_TEMPLATE',
    'NO_HISTORY_MESSAGE',
    'ACTION_HISTORY_HEADER',
    'GUIDANCE_TARGET_FOUND',
    'GUIDANCE_REPEATED_FAILURES',
    'GUIDANCE_ALL_EXPLORED',
    'GUIDANCE_UNEXPLORED',
    'GUIDANCE_DEFAULT',
    'ROOM_CLASSIFICATION_SYSTEM_PROMPT',
    'ROOM_CLASSIFICATION_USER_PROMPT',
    'format_action_history',
    'get_decision_guidance',
    'count_recent_failures',
    'check_target_in_objects',
    'format_discovered_rooms',
]

