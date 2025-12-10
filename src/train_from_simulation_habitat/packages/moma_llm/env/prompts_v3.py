"""
Multi-LLM Prompts for Habitat LLM Environment (v3).

This version introduces a dual-LLM system with:
1. Main Planning LLM - Decides navigation actions
2. Storyteller LLM - Generates narrative summaries of exploration

Key features:
- All prompts from v2 for the main planning LLM
- New storyteller prompts for narrative generation
- User prompts with integrated story section
- Helper functions for both LLMs
"""

import re
from typing import List, Dict, Any, Optional

# =============================================================================
# MAIN PLANNING LLM - SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = '''You are a robot navigating an unexplored house. Your task is to {TASK_DESCRIPTION}.

Available actions:
{TOOL_DESCRIPTIONS}

Response format (follow strictly):
Analysis: Brief assessment of current situation and where the target might be.
Reasoning: Why this specific action is the best choice right now.
Command: function_name(argument)

Important rules:
- If you find the target object, call stop() immediately - do NOT navigate to it.
- Learn from failed actions - don't repeat the same failing action.
- When stuck, try opening doors to discover new rooms.
- Use the exploration story to understand your journey and avoid repeating mistakes.
'''

# =============================================================================
# MAIN PLANNING LLM - USER PROMPT (with story section)
# =============================================================================

USER_PROMPT = '''Current location: {CURRENT_ROOM}
Nearby objects: {LIST_NEARBY_OBJECTS}

{STORY_SECTION}=== DISCOVERED ROOMS & OBJECTS ===
{LIST_FOUND_ROOMS_AND_OBJECTS}
=== ACTION HISTORY ===
{ACTION_HISTORY_SECTION}
=== EXPLORATION STATUS ===
Can explore (has unexplored areas): {ROOMS_WITH_FRONTIER_DESCRIPTION}
Fully explored: {FULLY_EXPLORED_ROOMS}
Visited rooms: {VISITED_ROOMS}
{ROOMS_WITH_CLOSED_DOORS_DESCRIPTION}
=== DECISION GUIDANCE ===
{DECISION_GUIDANCE}
'''

# User prompt without story section (for fallback or ablation)
USER_PROMPT_NO_STORY = '''Current location: {CURRENT_ROOM}
Nearby objects: {LIST_NEARBY_OBJECTS}

=== DISCOVERED ROOMS & OBJECTS ===
{LIST_FOUND_ROOMS_AND_OBJECTS}
=== ACTION HISTORY ===
{ACTION_HISTORY_SECTION}
=== EXPLORATION STATUS ===
Can explore (has unexplored areas): {ROOMS_WITH_FRONTIER_DESCRIPTION}
Fully explored: {FULLY_EXPLORED_ROOMS}
Visited rooms: {VISITED_ROOMS}
{ROOMS_WITH_CLOSED_DOORS_DESCRIPTION}
=== DECISION GUIDANCE ===
{DECISION_GUIDANCE}
'''

# =============================================================================
# STORYTELLER LLM - SYSTEM PROMPT
# =============================================================================

STORYTELLER_SYSTEM_PROMPT = '''You are a narrative assistant helping a robot understand its exploration journey.
Your task is to summarize the robot's previous actions and observations in a clear, story-like manner.

Guidelines:
- Write in second person ("You did X", "You found Y")
- Focus on what the robot discovered, where it went, and what it tried
- Highlight relevant findings that might help with the current task
- Keep the narrative concise but informative (3-5 sentences typically)
- Mention failed actions and what was learned from them
- Connect observations to potential next steps

The summary should help the robot remember its journey and make better decisions.'''

# =============================================================================
# STORYTELLER LLM - USER PROMPTS
# =============================================================================

# Initial story generation prompt
STORYTELLER_USER_PROMPT = '''The robot is tasked with: {TASK_DESCRIPTION}

=== EXPLORATION JOURNEY ===
Starting location: {STARTING_ROOM}
Current location: {CURRENT_ROOM}

=== ROOMS DISCOVERED ===
{DISCOVERED_ROOMS}

=== ACTION HISTORY (oldest to newest) ===
{ACTION_HISTORY}

=== CURRENT OBSERVATIONS ===
Nearby objects: {NEARBY_OBJECTS}
Unexplored areas: {UNEXPLORED_AREAS}

Please provide a brief narrative summary (3-5 sentences) of the robot's exploration journey so far.
Focus on what it has tried, what it found, and any patterns or insights that might help with the task.'''

# Story update prompt (for incremental updates)
STORYTELLER_UPDATE_PROMPT = '''The robot just took a new action.

Previous summary: {PREVIOUS_SUMMARY}

New action: {NEW_ACTION}
Action result: {ACTION_RESULT}
New observations: {NEW_OBSERVATIONS}

Please update the narrative summary to include this new development. Keep the total summary to 3-6 sentences.'''

# =============================================================================
# STORY SECTION FORMATTING
# =============================================================================

# Header for story section in main prompt
STORY_SECTION_HEADER = "=== EXPLORATION STORY ==="

# Template for story section
STORY_SECTION_TEMPLATE = '''=== EXPLORATION STORY ===
{STORY_CONTENT}

'''

# Empty story placeholder
STORY_EMPTY = "You have just started your exploration."

# =============================================================================
# ACTION HISTORY FORMATTING
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
# STORYTELLER ACTION RECORD FORMATTING
# =============================================================================

# Template for successful action in storyteller
STORY_ACTION_SUCCESS = "- {action}({argument}): successfully"

# Template for failed action in storyteller  
STORY_ACTION_FAILED = "- {action}({argument}): unsuccessfully ({feedback})"

# Template for action with room change
STORY_ACTION_ROOM_CHANGE = " [moved from {room_before} to {room_after}]"

# Template for action with new discoveries
STORY_ACTION_DISCOVERIES = " [discovered: {discoveries}]"

# =============================================================================
# DECISION GUIDANCE TEMPLATES
# =============================================================================

# When target object is found
GUIDANCE_TARGET_FOUND = '''🎯 TARGET FOUND! The target "{target}" is in the list above.
→ Call stop() immediately to complete the task.'''

# When there are repeated failures
GUIDANCE_REPEATED_FAILURES = '''⚠️ Recent actions have been failing repeatedly.
→ Try a DIFFERENT approach: open a door, explore a different room, or use a different target.
→ Review the exploration story above to avoid repeating past mistakes.
→ If you struggle too long and cannot find the target, call stop() to terminate the task.'''

# When all rooms are explored but target not found
GUIDANCE_ALL_EXPLORED = '''All discovered rooms have been fully explored but target not found.
→ Focus on opening closed doors to discover new rooms that may contain the target.
→ Check the exploration story - you may have missed something.'''

# When there are unexplored areas
GUIDANCE_UNEXPLORED = '''There are still unexplored areas in discovered rooms.
→ Consider exploring rooms with unexplored areas OR opening doors to find new rooms.
→ Use the exploration story to prioritize areas you haven't thoroughly checked.'''

# Default guidance
GUIDANCE_DEFAULT = '''Choose the most efficient action to find the target:
- If target is in the list → stop()
- If target might be in unexplored areas → explore(room)
- If target might be behind closed doors → open(door)
- Review the exploration story for context on what you've tried.'''

# =============================================================================
# RETRY/FAILURE PROMPTS
# =============================================================================

# When last action failed - includes specific feedback
RETRIAL_PROMPT = '''The last action failed: {FAILURE_REASON}

Please choose a different action. Consider:
- Trying a different target (door, room, or object)
- Using a different action type
- The target might be inaccessible from your current position
- Review the exploration story for alternative approaches

Remember to include "Command:" before your action.'''

# Generic retry without specific reason
RETRIAL_PROMPT_GENERIC = '''The last action failed. Please try a different approach.

Tips:
- Don't repeat the exact same action that just failed
- Try a different target or action type
- Check if there are alternative paths or objects
- Think about what the exploration story tells you about this area

Remember to include "Command:" before your action.'''

# Format/parsing error
RETRIAL_PROMPT_FORMAT_ERROR = '''Your response could not be parsed due to format errors.

Required format:
Analysis: [your analysis]
Reasoning: [your reasoning]  
Command: function_name(argument)

Use only available functions with exact object/room names from the list above.'''

# =============================================================================
# ROOM CLASSIFICATION PROMPTS
# =============================================================================

ROOM_CLASSIFICATION_SYSTEM_PROMPT = '''You are a helpful assistant identifying room types in an apartment based on their contents.'''

ROOM_CLASSIFICATION_USER_PROMPT = '''You observe {NUM_ROOMS} rooms containing the following objects:
{ROOM_OBJECT_LIST}

{REQUESTS}

Respond with a bullet list in this exact format:
 - room-X: room type

{REMEMBER}
Include ONLY the bullet list, no other text.'''

# =============================================================================
# HELPER FUNCTIONS - MAIN LLM
# =============================================================================

def format_action_history(action_history: list, max_items: int = 5) -> str:
    """
    Format action history with success/failure status and failure reasons.
    
    Args:
        action_history: List of dicts with keys: action, argument, success, failure_reason (optional)
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
        action = item.get('action', 'unknown')
        argument = item.get('argument', '')
        success = item.get('success', False)
        
        if success:
            lines.append(ACTION_SUCCESS_TEMPLATE.format(action=action, argument=argument))
        else:
            reason = item.get('failure_reason', 'unknown error')
            lines.append(ACTION_FAILED_TEMPLATE.format(action=action, argument=argument, reason=reason))
    
    return '\n'.join(lines)


def format_action_history_from_dataclass(action_history: list, llm_to_human_readable, max_items: int = 5) -> str:
    """
    Format action history from ActionHistory dataclass objects.
    
    Args:
        action_history: List of ActionHistory dataclass instances
        llm_to_human_readable: Function to convert graph names to human-readable names
        max_items: Maximum number of items to show
        
    Returns:
        Formatted action history string
    """
    if not action_history:
        return NO_HISTORY_MESSAGE
    
    # Take most recent actions (newest first)
    recent = action_history[-max_items:][::-1]
    
    lines = [ACTION_HISTORY_HEADER]
    for h in recent:
        action = h.action
        
        # Get human-readable argument
        if h.object_name_graph:
            argument = llm_to_human_readable(h.object_name_graph)
        elif h.orig_api_call:
            match = re.search(r'\(([^)]*)\)', h.orig_api_call)
            argument = match.group(1) if match else ""
        else:
            argument = ""
        
        if h.subtask_success:
            lines.append(ACTION_SUCCESS_TEMPLATE.format(action=action, argument=argument))
        else:
            reason = getattr(h, 'feedback', None) or "navigation/execution failed"
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
        target_found: Whether the target object has been found
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
        return RETRIAL_PROMPT.format(FAILURE_REASON=failure_reason)
    return RETRIAL_PROMPT_GENERIC


def count_recent_failures(action_history: list, lookback: int = 5) -> int:
    """
    Count consecutive failures in recent action history.
    
    Args:
        action_history: List of ActionHistory objects
        lookback: Number of recent actions to check
        
    Returns:
        Count of consecutive failures from the most recent action
    """
    if not action_history:
        return 0
    
    count = 0
    for h in reversed(action_history[-lookback:]):
        if not h.subtask_success:
            count += 1
        else:
            break
    return count


def check_target_in_objects(target_name: str, room_dict: dict) -> bool:
    """
    Check if target object is in any discovered room's object list.
    
    Args:
        target_name: Name of target to find (can be partial match)
        room_dict: Dictionary mapping room names to object lists
        
    Returns:
        True if target is found in any room
    """
    target_lower = target_name.lower()
    for room, objects in room_dict.items():
        for obj in objects:
            if target_lower in obj.lower():
                return True
    return False


# =============================================================================
# HELPER FUNCTIONS - STORYTELLER
# =============================================================================

def format_story_section(story_content: str) -> str:
    """
    Format story content for inclusion in the main LLM prompt.
    
    Args:
        story_content: The narrative summary from storyteller
        
    Returns:
        Formatted story section string
    """
    if not story_content or story_content.strip() == "":
        return ""
    
    return STORY_SECTION_TEMPLATE.format(STORY_CONTENT=story_content)


def format_action_for_storyteller(
    action: str,
    argument: str,
    success: bool,
    feedback: str = "",
    room_before: str = "",
    room_after: str = "",
    discoveries: List[str] = None
) -> str:
    """
    Format a single action record for storyteller context.
    
    Args:
        action: Action name
        argument: Action argument
        success: Whether action succeeded
        feedback: Feedback message if failed
        room_before: Room before action
        room_after: Room after action
        discoveries: List of new discoveries
        
    Returns:
        Formatted action string for storyteller
    """
    if success:
        line = STORY_ACTION_SUCCESS.format(action=action, argument=argument)
    else:
        line = STORY_ACTION_FAILED.format(action=action, argument=argument, feedback=feedback or "unknown")
    
    # Add room change info
    if room_before and room_after and room_before != room_after:
        line += STORY_ACTION_ROOM_CHANGE.format(room_before=room_before, room_after=room_after)
    
    # Add discoveries
    if discoveries:
        line += STORY_ACTION_DISCOVERIES.format(discoveries=", ".join(discoveries))
    
    return line


def format_discovered_rooms_for_storyteller(room_dict: Dict[str, List[str]], max_objects: int = 10) -> str:
    """
    Format discovered rooms for storyteller prompt.
    
    Args:
        room_dict: Dict mapping room names to object lists
        max_objects: Maximum objects to show per room
        
    Returns:
        Formatted rooms string
    """
    if not room_dict:
        return "No rooms discovered yet."
    
    lines = []
    for room, objects in room_dict.items():
        obj_list = ", ".join(objects[:max_objects])
        if len(objects) > max_objects:
            obj_list += f" (and {len(objects) - max_objects} more)"
        lines.append(f"- {room}: [{obj_list}]")
    
    return "\n".join(lines)


def format_action_history_for_storyteller(action_records: list) -> str:
    """
    Format action history for storyteller prompt.
    
    Args:
        action_records: List of action record dicts or objects
        
    Returns:
        Formatted action history string
    """
    if not action_records:
        return "No actions taken yet."
    
    lines = []
    for i, record in enumerate(action_records, 1):
        # Handle both dict and object formats
        if isinstance(record, dict):
            action = record.get('action', 'unknown')
            argument = record.get('argument', '')
            success = record.get('success', False)
            feedback = record.get('feedback', '')
            room_before = record.get('room_before', '')
            room_after = record.get('room_after', '')
            discoveries = record.get('new_discoveries', [])
        else:
            # Assume it's an object with attributes
            action = getattr(record, 'action', 'unknown')
            argument = getattr(record, 'argument', '')
            success = getattr(record, 'success', False)
            feedback = getattr(record, 'feedback', '')
            room_before = getattr(record, 'room_before', '')
            room_after = getattr(record, 'room_after', '')
            discoveries = getattr(record, 'new_discoveries', [])
        
        formatted = format_action_for_storyteller(
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
    """
    Generate a simple fallback story without LLM (for error cases).
    
    Args:
        starting_room: Room where exploration started
        current_room: Current room
        action_records: List of action records
        
    Returns:
        Simple narrative summary
    """
    if not action_records:
        return STORY_EMPTY
    
    lines = []
    lines.append(f"You started in {starting_room or 'an unknown room'}.")
    
    # Count action types
    successful_explores = sum(1 for r in action_records 
                             if getattr(r, 'action', r.get('action', '')) == 'explore' 
                             and getattr(r, 'success', r.get('success', False)))
    successful_gotos = [r for r in action_records 
                       if getattr(r, 'action', r.get('action', '')) == 'goto' 
                       and getattr(r, 'success', r.get('success', False))]
    failed_actions = sum(1 for r in action_records 
                        if not getattr(r, 'success', r.get('success', True)))
    
    if successful_explores:
        lines.append(f"You explored {successful_explores} area(s).")
    
    if successful_gotos:
        targets = [getattr(r, 'argument', r.get('argument', '')) for r in successful_gotos[-3:]]
        lines.append(f"You navigated to: {', '.join(targets)}.")
    
    if failed_actions:
        lines.append(f"Some actions failed ({failed_actions} total).")
    
    if current_room:
        lines.append(f"You are currently in {current_room}.")
    
    return " ".join(lines)


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

def build_storyteller_prompt(
    task_description: str,
    starting_room: str,
    current_room: str,
    discovered_rooms: Dict[str, List[str]],
    action_history: list,
    nearby_objects: List[str],
    unexplored_areas: List[str]
) -> tuple:
    """
    Build system and user prompts for storyteller.
    
    Args:
        task_description: The current task
        starting_room: Starting room
        current_room: Current room
        discovered_rooms: Dict of rooms to objects
        action_history: List of action records
        nearby_objects: List of nearby objects
        unexplored_areas: List of rooms with unexplored areas
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    user_prompt = STORYTELLER_USER_PROMPT.format(
        TASK_DESCRIPTION=task_description,
        STARTING_ROOM=starting_room or "unknown",
        CURRENT_ROOM=current_room or "unknown",
        DISCOVERED_ROOMS=format_discovered_rooms_for_storyteller(discovered_rooms),
        ACTION_HISTORY=format_action_history_for_storyteller(action_history),
        NEARBY_OBJECTS=", ".join(nearby_objects) if nearby_objects else "none visible",
        UNEXPLORED_AREAS=", ".join(unexplored_areas) if unexplored_areas else "none"
    )
    
    return STORYTELLER_SYSTEM_PROMPT, user_prompt


def build_storyteller_update_prompt(
    previous_summary: str,
    new_action: str,
    action_result: str,
    new_observations: str
) -> tuple:
    """
    Build prompt for updating an existing story.
    
    Args:
        previous_summary: The existing story summary
        new_action: The new action taken (e.g., "explore(kitchen)")
        action_result: Result of the action
        new_observations: New things observed
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    user_prompt = STORYTELLER_UPDATE_PROMPT.format(
        PREVIOUS_SUMMARY=previous_summary,
        NEW_ACTION=new_action,
        ACTION_RESULT=action_result,
        NEW_OBSERVATIONS=new_observations or "no new observations"
    )
    
    return STORYTELLER_SYSTEM_PROMPT, user_prompt


def build_main_llm_prompt(
    task_description: str,
    tool_descriptions: str,
    current_room: str,
    nearby_objects: str,
    story_content: str,
    found_rooms_and_objects: str,
    action_history_section: str,
    rooms_with_frontier: str,
    fully_explored_rooms: str,
    visited_rooms: str,
    rooms_with_closed_doors: str,
    decision_guidance: str,
    include_story: bool = True
) -> tuple:
    """
    Build system and user prompts for main planning LLM.
    
    Args:
        task_description: The task description
        tool_descriptions: Available actions
        current_room: Current room
        nearby_objects: Nearby objects string
        story_content: Story from storyteller (or empty)
        found_rooms_and_objects: Formatted rooms and objects
        action_history_section: Formatted action history
        rooms_with_frontier: Rooms with unexplored areas
        fully_explored_rooms: Fully explored rooms
        visited_rooms: Visited rooms
        rooms_with_closed_doors: Rooms with closed doors
        decision_guidance: Guidance text
        include_story: Whether to include story section
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = SYSTEM_PROMPT.format(
        TASK_DESCRIPTION=task_description,
        TOOL_DESCRIPTIONS=tool_descriptions
    )
    
    # Format story section
    story_section = ""
    if include_story and story_content:
        story_section = format_story_section(story_content)
    
    user_prompt = USER_PROMPT.format(
        CURRENT_ROOM=current_room,
        LIST_NEARBY_OBJECTS=nearby_objects,
        STORY_SECTION=story_section,
        LIST_FOUND_ROOMS_AND_OBJECTS=found_rooms_and_objects,
        ACTION_HISTORY_SECTION=action_history_section,
        ROOMS_WITH_FRONTIER_DESCRIPTION=rooms_with_frontier,
        FULLY_EXPLORED_ROOMS=fully_explored_rooms,
        VISITED_ROOMS=visited_rooms,
        ROOMS_WITH_CLOSED_DOORS_DESCRIPTION=rooms_with_closed_doors,
        DECISION_GUIDANCE=decision_guidance
    )
    
    return system_prompt, user_prompt


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main LLM prompts
    'SYSTEM_PROMPT',
    'USER_PROMPT',
    'USER_PROMPT_NO_STORY',
    
    # Storyteller prompts
    'STORYTELLER_SYSTEM_PROMPT',
    'STORYTELLER_USER_PROMPT',
    'STORYTELLER_UPDATE_PROMPT',
    
    # Story formatting
    'STORY_SECTION_HEADER',
    'STORY_SECTION_TEMPLATE',
    'STORY_EMPTY',
    
    # Action history formatting
    'ACTION_SUCCESS_TEMPLATE',
    'ACTION_FAILED_TEMPLATE',
    'NO_HISTORY_MESSAGE',
    'ACTION_HISTORY_HEADER',
    
    # Storyteller action formatting
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
    
    # Helper functions - Main LLM
    'format_action_history',
    'format_action_history_from_dataclass',
    'get_decision_guidance',
    'format_retry_prompt',
    'count_recent_failures',
    'check_target_in_objects',
    
    # Helper functions - Storyteller
    'format_story_section',
    'format_action_for_storyteller',
    'format_discovered_rooms_for_storyteller',
    'format_action_history_for_storyteller',
    'generate_fallback_story',
    
    # Prompt builders
    'build_storyteller_prompt',
    'build_storyteller_update_prompt',
    'build_main_llm_prompt',
]

