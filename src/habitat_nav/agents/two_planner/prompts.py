"""
Two-Planner Prompts for Habitat Navigation

This module contains prompts for a dual-LLM system with:
1. Main Planning LLM - Decides navigation actions
2. Narrator LLM - Generates narrative summaries of exploration journey

Based on prompts_v3.py from train_from_simulation_habitat.
"""

import re
from typing import List, Dict, Optional

# =============================================================================
# MAIN PLANNING LLM - SYSTEM PROMPT
# =============================================================================

MAIN_PLANNER_SYSTEM_PROMPT = """You are a robot navigating an unexplored house. Your task is to {task_description}.

Available actions:
{action_descriptions}

Response format (follow strictly):
Analysis: Brief assessment of current situation and where the target might be.
Reasoning: Why this specific action is the best choice right now.
Command: function_name(argument)

Important rules:
- If you find the target object, call goto() to get close to the object.
- Learn from failed actions - don't repeat the same failing action.
- When stuck, try opening doors to discover new rooms.
- Use the exploration story to understand your journey and avoid repeating mistakes.
"""

# =============================================================================
# MAIN PLANNING LLM - USER PROMPT (with story section)
# =============================================================================

MAIN_PLANNER_USER_PROMPT = """Current location: {current_room}

=== VISIBLE OBJECTS ===
{nearby_objects}

{story_section}=== DISCOVERED ROOMS & OBJECTS ===
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

# User prompt without story section (for fallback or ablation)
MAIN_PLANNER_USER_PROMPT_NO_STORY = """Current location: {current_room}

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
# NARRATOR/STORYTELLER LLM - SYSTEM PROMPT
# =============================================================================

NARRATOR_SYSTEM_PROMPT = """You are a narrative assistant helping a robot understand its exploration journey.
Your task is to summarize the robot's previous actions and observations in a clear, story-like manner.

Guidelines:
- Write in second person ("You did X", "You found Y")
- Focus on what the robot discovered, where it went, and what it tried
- Highlight relevant findings that might help with the current task
- Keep the narrative concise but informative (3-5 sentences typically)
- Mention failed actions and what was learned from them
- Connect observations to potential next steps

The summary should help the robot remember its journey and make better decisions."""

# =============================================================================
# NARRATOR LLM - USER PROMPTS
# =============================================================================

# Initial story generation prompt
NARRATOR_USER_PROMPT = """The robot is tasked with: {task_description}

=== EXPLORATION JOURNEY ===
Starting location: {starting_room}
Current location: {current_room}

=== ROOMS DISCOVERED ===
{discovered_rooms}

=== ACTION HISTORY (oldest to newest) ===
{action_history}

=== CURRENT OBSERVATIONS ===
Nearby objects: {nearby_objects}
Unexplored areas: {unexplored_areas}

Please provide a brief narrative summary (3-5 sentences) of the robot's exploration journey so far.
Focus on what it has tried, what it found, and any patterns or insights that might help with the task."""

# Story update prompt (for incremental updates)
NARRATOR_UPDATE_PROMPT = """The robot just took a new action.

Previous summary: {previous_summary}

New action: {new_action}
Action result: {action_result}
New observations: {new_observations}

Please update the narrative summary to include this new development. Keep the total summary to 3-6 sentences."""

# =============================================================================
# STORY SECTION FORMATTING
# =============================================================================

STORY_SECTION_HEADER = "=== EXPLORATION STORY ==="

STORY_SECTION_TEMPLATE = """=== EXPLORATION STORY ===
{story_content}

"""

STORY_EMPTY = "You have just started your exploration."

# =============================================================================
# ACTION HISTORY FORMATTING
# =============================================================================

ACTION_SUCCESS_TEMPLATE = "✓ {action}({argument})"
ACTION_FAILED_TEMPLATE = "✗ {action}({argument}) - FAILED: {reason}"
NO_HISTORY_MESSAGE = "No actions taken yet."
ACTION_HISTORY_HEADER = "Recent actions (newest first):"

# Storyteller action formatting
STORY_ACTION_SUCCESS = "- {action}({argument}): successfully"
STORY_ACTION_FAILED = "- {action}({argument}): unsuccessfully ({feedback})"
STORY_ACTION_ROOM_CHANGE = " [moved from {room_before} to {room_after}]"
STORY_ACTION_DISCOVERIES = " [discovered: {discoveries}]"

# =============================================================================
# DECISION GUIDANCE TEMPLATES
# =============================================================================

GUIDANCE_TARGET_FOUND = """🎯 TARGET FOUND! The target "{target}" is in the visible objects list.
→ Call goto({target}) to get close and complete the task."""

GUIDANCE_REPEATED_FAILURES = """⚠️ Recent actions have been failing repeatedly.
→ Try a DIFFERENT approach: open a door, explore a different room, or use a different target.
→ Review the exploration story above to avoid repeating past mistakes.
→ If you struggle too long and cannot find the target, call stop() to terminate the task."""

GUIDANCE_ALL_EXPLORED = """All discovered rooms have been fully explored but target not found.
→ Focus on opening closed doors to discover new rooms that may contain the target.
→ Check the exploration story - you may have missed something."""

GUIDANCE_UNEXPLORED = """There are still unexplored areas in discovered rooms.
→ Consider exploring rooms with unexplored areas OR opening doors to find new rooms.
→ Use the exploration story to prioritize areas you haven't thoroughly checked."""

GUIDANCE_DEFAULT = """Choose the most efficient action to find the target:
- If target is in visible objects → goto(target)
- If target might be in unexplored areas → explore(room)
- If target might be behind closed doors → open(door)
- Review the exploration story for context on what you've tried."""

# =============================================================================
# RETRY/FAILURE PROMPTS
# =============================================================================

RETRIAL_PROMPT = """The last action failed: {failure_reason}

Please choose a different action. Consider:
- Trying a different target (door, room, or object)
- Using a different action type
- The target might be inaccessible from your current position
- Review the exploration story for alternative approaches

Remember to include "Command:" before your action."""

RETRIAL_PROMPT_GENERIC = """The last action failed. Please try a different approach.

Tips:
- Don't repeat the exact same action that just failed
- Try a different target or action type
- Check if there are alternative paths or objects
- Think about what the exploration story tells you about this area

Remember to include "Command:" before your action."""

RETRIAL_PROMPT_FORMAT_ERROR = """Your response could not be parsed due to format errors.

Required format:
Analysis: [your analysis]
Reasoning: [your reasoning]  
Command: function_name(argument)

Use only available functions with exact object/room names from the list above."""

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
# HELPER FUNCTIONS - MAIN PLANNER
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
    """Generate retry prompt with optional failure reason."""
    if failure_reason:
        return RETRIAL_PROMPT.format(failure_reason=failure_reason)
    return RETRIAL_PROMPT_GENERIC


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


# =============================================================================
# HELPER FUNCTIONS - NARRATOR
# =============================================================================

def format_story_section(story_content: str) -> str:
    """Format story content for inclusion in the main planner prompt."""
    if not story_content or story_content.strip() == "":
        return ""
    
    return STORY_SECTION_TEMPLATE.format(story_content=story_content)


def format_action_for_narrator(
    action: str,
    argument: str,
    success: bool,
    feedback: str = "",
    room_before: str = "",
    room_after: str = "",
    discoveries: List[str] = None
) -> str:
    """Format a single action record for narrator context."""
    if success:
        line = STORY_ACTION_SUCCESS.format(action=action, argument=argument)
    else:
        line = STORY_ACTION_FAILED.format(action=action, argument=argument, feedback=feedback or "unknown")
    
    if room_before and room_after and room_before != room_after:
        line += STORY_ACTION_ROOM_CHANGE.format(room_before=room_before, room_after=room_after)
    
    if discoveries:
        line += STORY_ACTION_DISCOVERIES.format(discoveries=", ".join(discoveries))
    
    return line


def format_discovered_rooms_for_narrator(room_dict: Dict[str, List[str]], max_objects: int = 10) -> str:
    """Format discovered rooms for narrator prompt."""
    if not room_dict:
        return "No rooms discovered yet."
    
    lines = []
    for room, objects in room_dict.items():
        obj_list = ", ".join(objects[:max_objects])
        if len(objects) > max_objects:
            obj_list += f" (and {len(objects) - max_objects} more)"
        lines.append(f"- {room}: [{obj_list}]")
    
    return "\n".join(lines)


def format_action_history_for_narrator(action_records: list) -> str:
    """Format action history for narrator prompt (oldest to newest)."""
    if not action_records:
        return "No actions taken yet."
    
    lines = []
    for i, record in enumerate(action_records, 1):
        if isinstance(record, dict):
            action = record.get('action', 'unknown')
            argument = record.get('argument', '')
            success = record.get('success', False)
            feedback = record.get('feedback', '')
            room_before = record.get('room_before', '')
            room_after = record.get('room_after', '')
            discoveries = record.get('new_discoveries', [])
        elif isinstance(record, tuple):
            # Handle tuple format: (action, argument, success, feedback)
            action = record[0] if len(record) > 0 else 'unknown'
            argument = record[1] if len(record) > 1 else ''
            success = record[2] if len(record) > 2 else False
            feedback = record[3] if len(record) > 3 else ''
            room_before = ''
            room_after = ''
            discoveries = []
        else:
            action = getattr(record, 'action', 'unknown')
            argument = getattr(record, 'argument', '')
            success = getattr(record, 'success', False)
            feedback = getattr(record, 'feedback', '')
            room_before = getattr(record, 'room_before', '')
            room_after = getattr(record, 'room_after', '')
            discoveries = getattr(record, 'new_discoveries', [])
        
        formatted = format_action_for_narrator(
            action=action,
            argument=argument,
            success=success,
            feedback=feedback,
            room_before=room_before,
            room_after=room_after,
            discoveries=discoveries
        )
        lines.append(f"{i}. {formatted}")
    
    return "\n".join(lines)


def generate_fallback_story(
    starting_room: str,
    current_room: str,
    action_records: list
) -> str:
    """Generate a simple fallback story without LLM (for error cases)."""
    if not action_records:
        return STORY_EMPTY
    
    lines = []
    lines.append(f"You started in {starting_room or 'an unknown room'}.")
    
    # Count action types
    successful_explores = sum(1 for r in action_records 
                             if (r[0] if isinstance(r, tuple) else getattr(r, 'action', '')) == 'explore' 
                             and (r[2] if isinstance(r, tuple) else getattr(r, 'success', False)))
    
    successful_gotos = [r for r in action_records 
                        if (r[0] if isinstance(r, tuple) else getattr(r, 'action', '')) == 'goto' 
                        and (r[2] if isinstance(r, tuple) else getattr(r, 'success', False))]
    
    failed_actions = sum(1 for r in action_records 
                         if not (r[2] if isinstance(r, tuple) else getattr(r, 'success', True)))
    
    if successful_explores:
        lines.append(f"You explored {successful_explores} area(s).")
    
    if successful_gotos:
        targets = [(r[1] if isinstance(r, tuple) else getattr(r, 'argument', '')) for r in successful_gotos[-3:]]
        lines.append(f"You navigated to: {', '.join(targets)}.")
    
    if failed_actions:
        lines.append(f"Some actions failed ({failed_actions} total).")
    
    if current_room:
        lines.append(f"You are currently in {current_room}.")
    
    return " ".join(lines)


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

def build_narrator_prompt(
    task_description: str,
    starting_room: str,
    current_room: str,
    discovered_rooms: Dict[str, List[str]],
    action_history: list,
    nearby_objects: List[str],
    unexplored_areas: List[str]
) -> tuple:
    """
    Build system and user prompts for narrator.
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    user_prompt = NARRATOR_USER_PROMPT.format(
        task_description=task_description,
        starting_room=starting_room or "unknown",
        current_room=current_room or "unknown",
        discovered_rooms=format_discovered_rooms_for_narrator(discovered_rooms),
        action_history=format_action_history_for_narrator(action_history),
        nearby_objects=", ".join(nearby_objects) if nearby_objects else "none visible",
        unexplored_areas=", ".join(unexplored_areas) if unexplored_areas else "none"
    )
    
    return NARRATOR_SYSTEM_PROMPT, user_prompt


def build_narrator_update_prompt(
    previous_summary: str,
    new_action: str,
    action_result: str,
    new_observations: str
) -> tuple:
    """
    Build prompt for updating an existing story.
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    user_prompt = NARRATOR_UPDATE_PROMPT.format(
        previous_summary=previous_summary,
        new_action=new_action,
        action_result=action_result,
        new_observations=new_observations or "no new observations"
    )
    
    return NARRATOR_SYSTEM_PROMPT, user_prompt


def build_main_planner_prompt(
    task_description: str,
    action_descriptions: str,
    current_room: str,
    nearby_objects: str,
    story_content: str,
    discovered_rooms: str,
    action_history: str,
    unexplored_info: str,
    exploration_status: str,
    decision_guidance: str,
    target_object: str,
    last_feedback: str = "None",
    include_story: bool = True
) -> tuple:
    """
    Build system and user prompts for main planning LLM.
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = MAIN_PLANNER_SYSTEM_PROMPT.format(
        task_description=task_description,
        action_descriptions=action_descriptions
    )
    
    story_section = ""
    if include_story and story_content:
        story_section = format_story_section(story_content)
    
    user_prompt = MAIN_PLANNER_USER_PROMPT.format(
        current_room=current_room,
        nearby_objects=nearby_objects,
        story_section=story_section,
        discovered_rooms=discovered_rooms,
        action_history=action_history,
        unexplored_info=unexplored_info,
        exploration_status=exploration_status,
        decision_guidance=decision_guidance,
        target_object=target_object,
        last_feedback=last_feedback
    )
    
    return system_prompt, user_prompt


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main planner prompts
    'MAIN_PLANNER_SYSTEM_PROMPT',
    'MAIN_PLANNER_USER_PROMPT',
    'MAIN_PLANNER_USER_PROMPT_NO_STORY',
    
    # Narrator prompts
    'NARRATOR_SYSTEM_PROMPT',
    'NARRATOR_USER_PROMPT',
    'NARRATOR_UPDATE_PROMPT',
    
    # Story formatting
    'STORY_SECTION_HEADER',
    'STORY_SECTION_TEMPLATE',
    'STORY_EMPTY',
    
    # Action history formatting
    'ACTION_SUCCESS_TEMPLATE',
    'ACTION_FAILED_TEMPLATE',
    'NO_HISTORY_MESSAGE',
    'ACTION_HISTORY_HEADER',
    
    # Narrator action formatting
    'STORY_ACTION_SUCCESS',
    'STORY_ACTION_FAILED',
    'STORY_ACTION_ROOM_CHANGE',
    'STORY_ACTION_DISCOVERIES',
    
    # Decision guidance
    'GUIDANCE_TARGET_FOUND',
    'GUIDANCE_REPEATED_FAILURES',
    'GUIDANCE_ALL_EXPLORED',
    'GUIDANCE_UNEXPLORED',
    'GUIDANCE_DEFAULT',
    
    # Retry prompts
    'RETRIAL_PROMPT',
    'RETRIAL_PROMPT_GENERIC',
    'RETRIAL_PROMPT_FORMAT_ERROR',
    
    # Room classification
    'ROOM_CLASSIFICATION_SYSTEM_PROMPT',
    'ROOM_CLASSIFICATION_USER_PROMPT',
    
    # Helper functions - Main planner
    'format_action_history',
    'get_decision_guidance',
    'format_retry_prompt',
    'count_recent_failures',
    'check_target_in_objects',
    'format_discovered_rooms',
    
    # Helper functions - Narrator
    'format_story_section',
    'format_action_for_narrator',
    'format_discovered_rooms_for_narrator',
    'format_action_history_for_narrator',
    'generate_fallback_story',
    
    # Prompt builders
    'build_narrator_prompt',
    'build_narrator_update_prompt',
    'build_main_planner_prompt',
]

