"""
Simplified Multi-LLM Prompts for Habitat LLM Environment (v4).

This version simplifies v3 by:
- Removing DISCOVERED OBJECTS section (story provides this context)
- Keeping only discovered rooms list
- Retaining door information for action planning
- Relying more on the exploration story for object context

Key principle: The storyteller provides rich context about objects found,
so the main LLM prompt focuses on actionable information (rooms, doors, status).
"""

import re
from typing import List, Dict, Any, Optional

# =============================================================================
# MAIN PLANNING LLM - SYSTEM PROMPT (Simplified)
# =============================================================================

SYSTEM_PROMPT = '''You are a robot navigating an unexplored house. Your task is to {TASK_DESCRIPTION}.

Available actions:
{TOOL_DESCRIPTIONS}

Response format (follow strictly):
Analysis: Brief assessment based on your exploration story.
Reasoning: Why this action is the best next step.
Command: function_name(argument)

Rules:
- If you find the target object, call stop() immediately.
- Use the exploration story to avoid repeating failed actions.
- Open doors to discover new rooms when stuck.
'''

# =============================================================================
# MAIN PLANNING LLM - USER PROMPT (Simplified - no object lists)
# =============================================================================

USER_PROMPT = '''Current location: {CURRENT_ROOM}
Nearby objects: {LIST_NEARBY_OBJECTS}

{STORY_SECTION}=== DISCOVERED ROOMS ===
{DISCOVERED_ROOMS_LIST}

=== EXPLORATION STATUS ===
Unexplored areas: {ROOMS_WITH_FRONTIER_DESCRIPTION}
Fully explored: {FULLY_EXPLORED_ROOMS}
{DOORS_SECTION}
=== ACTION HISTORY ===
{ACTION_HISTORY_SECTION}

=== GUIDANCE ===
{DECISION_GUIDANCE}
'''

# User prompt without story section (fallback/ablation)
USER_PROMPT_NO_STORY = '''Current location: {CURRENT_ROOM}
Nearby objects: {LIST_NEARBY_OBJECTS}

=== DISCOVERED ROOMS ===
{DISCOVERED_ROOMS_LIST}

=== EXPLORATION STATUS ===
Unexplored areas: {ROOMS_WITH_FRONTIER_DESCRIPTION}
Fully explored: {FULLY_EXPLORED_ROOMS}
{DOORS_SECTION}
=== ACTION HISTORY ===
{ACTION_HISTORY_SECTION}

=== GUIDANCE ===
{DECISION_GUIDANCE}
'''

# =============================================================================
# STORYTELLER LLM - SYSTEM PROMPT (Enhanced to carry more object context)
# =============================================================================

STORYTELLER_SYSTEM_PROMPT = '''You are a narrative assistant helping a robot understand its exploration journey.
Summarize the robot's actions, discoveries, and observations in a clear story format.

CRITICAL: Check if the TARGET OBJECT from the task appears in ANY discovered room's object list.
If found, you MUST clearly state: "TARGET FOUND: [object name] was discovered in [room name]!"

Guidelines:
- Write in second person ("You went to X", "You found Y")
- Emphasize objects discovered that might relate to the task
- ALWAYS check if the target object exists in the discovered rooms' objects
- Mention rooms visited and what was found there
- Note failed actions and lessons learned
- Keep it concise: 4-6 sentences max
- Focus on information useful for decision-making'''

# =============================================================================
# STORYTELLER LLM - USER PROMPTS
# =============================================================================

# Initial story generation
STORYTELLER_USER_PROMPT = '''Task: {TASK_DESCRIPTION}

IMPORTANT: Check if the target object from the task exists in ANY room's object list below!

Journey so far:
- Started: {STARTING_ROOM}
- Now: {CURRENT_ROOM}

Rooms explored (check these for the target!):
{DISCOVERED_ROOMS}

Actions taken:
{ACTION_HISTORY}

Current observations: {NEARBY_OBJECTS}
Unexplored areas: {UNEXPLORED_AREAS}

Provide a brief narrative (4-6 sentences). 
FIRST: Check if the target object exists in any room's object list. If found, START with "🎯 TARGET FOUND: [object] in [room]!"
Then summarize the exploration journey.'''

# Story update prompt
STORYTELLER_UPDATE_PROMPT = '''Previous story: {PREVIOUS_SUMMARY}

New action: {NEW_ACTION}
Result: {ACTION_RESULT}
New observations: {NEW_OBSERVATIONS}

Update the story (4-6 sentences). 
CRITICAL: If the target object now appears in discovered rooms, START with "🎯 TARGET FOUND!"
Otherwise, summarize what happened and what to try next.'''

# =============================================================================
# STORY SECTION FORMATTING
# =============================================================================

STORY_SECTION_TEMPLATE = '''=== YOUR EXPLORATION STORY ===
{STORY_CONTENT}

'''

STORY_EMPTY = "You have just started exploring."

# =============================================================================
# ACTION HISTORY FORMATTING (Compact)
# =============================================================================

ACTION_SUCCESS = "✓ {action}({argument})"
ACTION_FAILED = "✗ {action}({argument}) - {reason}"
NO_HISTORY = "No actions yet."

# =============================================================================
# DOORS SECTION
# =============================================================================

DOORS_SECTION_TEMPLATE = '''Closed doors: {CLOSED_DOORS}'''
NO_CLOSED_DOORS = "No closed doors found."

# =============================================================================
# DECISION GUIDANCE (Simplified)
# =============================================================================

GUIDANCE_TARGET_FOUND = '''🎯 TARGET FOUND! Call stop() now.'''

GUIDANCE_REPEATED_FAILURES = '''⚠️ Multiple failures. Try a different approach or open a door. If you have explored all the rooms and cannot find the target, try to call stop() to terminate.'''

GUIDANCE_ALL_EXPLORED = '''All rooms explored. Focus on opening closed doors.'''

GUIDANCE_UNEXPLORED = '''Unexplored areas remain. Continue exploring or open doors.'''

GUIDANCE_DEFAULT = '''Find the target: explore rooms, open doors, or stop() if found.'''

# =============================================================================
# RETRY PROMPTS
# =============================================================================

RETRY_PROMPT = '''Last action failed: {FAILURE_REASON}
Try a different target or action. Include "Command:" in response.'''

RETRY_PROMPT_GENERIC = '''Last action failed. Try something different.'''

RETRY_PROMPT_FORMAT = '''Format error. Use:
Analysis: [analysis]
Reasoning: [reasoning]
Command: function_name(argument)'''

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_action_history(action_history: list, max_items: int = 5) -> str:
    """Format action history compactly."""
    if not action_history:
        return NO_HISTORY
    
    recent = action_history[-max_items:][::-1]
    lines = []
    
    for item in recent:
        action = item.get('action', 'unknown')
        argument = item.get('argument', '')
        success = item.get('success', False)
        
        if success:
            lines.append(ACTION_SUCCESS.format(action=action, argument=argument))
        else:
            reason = item.get('failure_reason', 'failed')
            lines.append(ACTION_FAILED.format(action=action, argument=argument, reason=reason))
    
    return '\n'.join(lines)


def format_action_history_from_dataclass(action_history: list, llm_to_human_readable, max_items: int = 5) -> str:
    """Format action history from ActionHistory dataclass objects."""
    if not action_history:
        return NO_HISTORY
    
    recent = action_history[-max_items:][::-1]
    lines = []
    
    for h in recent:
        action = h.action
        
        if h.object_name_graph:
            argument = llm_to_human_readable(h.object_name_graph)
        elif h.orig_api_call:
            match = re.search(r'\(([^)]*)\)', h.orig_api_call)
            argument = match.group(1) if match else ""
        else:
            argument = ""
        
        if h.subtask_success:
            lines.append(ACTION_SUCCESS.format(action=action, argument=argument))
        else:
            reason = getattr(h, 'feedback', None) or "failed"
            lines.append(ACTION_FAILED.format(action=action, argument=argument, reason=reason))
    
    return '\n'.join(lines)


def format_discovered_rooms(room_dict: Dict[str, List[str]]) -> str:
    """Format discovered rooms (names only, no objects)."""
    if not room_dict:
        return "None yet."
    return ", ".join(room_dict.keys())


def format_doors_section(closed_doors: List[str]) -> str:
    """Format closed doors information."""
    if not closed_doors:
        return ""
    return DOORS_SECTION_TEMPLATE.format(CLOSED_DOORS=", ".join(closed_doors))


def format_story_section(story_content: str) -> str:
    """Format story for main LLM prompt."""
    if not story_content or story_content.strip() == "":
        return ""
    return STORY_SECTION_TEMPLATE.format(STORY_CONTENT=story_content)


def get_decision_guidance(
    target_found: bool = False,
    recent_failure_count: int = 0,
    has_unexplored_areas: bool = True,
    all_rooms_explored: bool = False
) -> str:
    """Generate contextual guidance."""
    if target_found:
        return GUIDANCE_TARGET_FOUND
    if recent_failure_count >= 3:
        return GUIDANCE_REPEATED_FAILURES
    if all_rooms_explored:
        return GUIDANCE_ALL_EXPLORED
    if has_unexplored_areas:
        return GUIDANCE_UNEXPLORED
    return GUIDANCE_DEFAULT


def format_retry_prompt(failure_reason: str = None) -> str:
    """Generate retry prompt."""
    if failure_reason:
        return RETRY_PROMPT.format(FAILURE_REASON=failure_reason)
    return RETRY_PROMPT_GENERIC


def count_recent_failures(action_history: list, lookback: int = 5) -> int:
    """Count consecutive recent failures."""
    if not action_history:
        return 0
    
    count = 0
    for h in reversed(action_history[-lookback:]):
        if not h.subtask_success:
            count += 1
        else:
            break
    return count


# =============================================================================
# STORYTELLER HELPERS
# =============================================================================

def format_rooms_for_storyteller(room_dict: Dict[str, List[str]], max_objects: int = 8) -> str:
    """Format rooms with objects for storyteller (storyteller needs object context)."""
    if not room_dict:
        return "No rooms discovered yet."
    
    lines = []
    for room, objects in room_dict.items():
        obj_list = ", ".join(objects[:max_objects])
        if len(objects) > max_objects:
            obj_list += f" (+{len(objects) - max_objects} more)"
        lines.append(f"- {room}: {obj_list}")
    
    return "\n".join(lines)


def format_actions_for_storyteller(action_records: list) -> str:
    """Format action history for storyteller."""
    if not action_records:
        return "No actions taken yet."
    
    lines = []
    for i, record in enumerate(action_records, 1):
        if isinstance(record, dict):
            action = record.get('action', 'unknown')
            argument = record.get('argument', '')
            success = record.get('success', False)
            feedback = record.get('feedback', '')
        else:
            action = getattr(record, 'action', 'unknown')
            argument = getattr(record, 'argument', '')
            success = getattr(record, 'success', False)
            feedback = getattr(record, 'feedback', '')
        
        status = "success" if success else f"failed ({feedback or 'error'})"
        lines.append(f"{i}. {action}({argument}): {status}")
    
    return "\n".join(lines)


def generate_fallback_story(starting_room: str, current_room: str, action_records: list) -> str:
    """Generate simple fallback story without LLM."""
    if not action_records:
        return STORY_EMPTY
    
    parts = [f"You started in {starting_room or 'an unknown room'}."]
    
    explores = sum(1 for r in action_records 
                   if getattr(r, 'action', r.get('action', '')) == 'explore' 
                   and getattr(r, 'success', r.get('success', False)))
    if explores:
        parts.append(f"You explored {explores} area(s).")
    
    failures = sum(1 for r in action_records 
                   if not getattr(r, 'success', r.get('success', True)))
    if failures:
        parts.append(f"Some actions failed ({failures}).")
    
    if current_room:
        parts.append(f"You are now in {current_room}.")
    
    return " ".join(parts)


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
    """Build storyteller prompts."""
    user_prompt = STORYTELLER_USER_PROMPT.format(
        TASK_DESCRIPTION=task_description,
        STARTING_ROOM=starting_room or "unknown",
        CURRENT_ROOM=current_room or "unknown",
        DISCOVERED_ROOMS=format_rooms_for_storyteller(discovered_rooms),
        ACTION_HISTORY=format_actions_for_storyteller(action_history),
        NEARBY_OBJECTS=", ".join(nearby_objects) if nearby_objects else "none",
        UNEXPLORED_AREAS=", ".join(unexplored_areas) if unexplored_areas else "none"
    )
    return STORYTELLER_SYSTEM_PROMPT, user_prompt


def build_storyteller_update_prompt(
    previous_summary: str,
    new_action: str,
    action_result: str,
    new_observations: str
) -> tuple:
    """Build story update prompts."""
    user_prompt = STORYTELLER_UPDATE_PROMPT.format(
        PREVIOUS_SUMMARY=previous_summary,
        NEW_ACTION=new_action,
        ACTION_RESULT=action_result,
        NEW_OBSERVATIONS=new_observations or "none"
    )
    return STORYTELLER_SYSTEM_PROMPT, user_prompt


def build_main_llm_prompt(
    task_description: str,
    tool_descriptions: str,
    current_room: str,
    nearby_objects: str,
    story_content: str,
    discovered_rooms: Dict[str, List[str]],
    action_history_section: str,
    rooms_with_frontier: str,
    fully_explored_rooms: str,
    closed_doors: List[str],
    decision_guidance: str,
    include_story: bool = True
) -> tuple:
    """
    Build main LLM prompts (simplified - no object lists).
    
    Args:
        task_description: The task
        tool_descriptions: Available actions
        current_room: Current room
        nearby_objects: Nearby objects string
        story_content: Story from storyteller
        discovered_rooms: Dict of rooms (used for room names only)
        action_history_section: Formatted action history
        rooms_with_frontier: Unexplored areas
        fully_explored_rooms: Fully explored rooms
        closed_doors: List of closed door names
        decision_guidance: Guidance text
        include_story: Include story section
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = SYSTEM_PROMPT.format(
        TASK_DESCRIPTION=task_description,
        TOOL_DESCRIPTIONS=tool_descriptions
    )
    
    story_section = ""
    if include_story and story_content:
        story_section = format_story_section(story_content)
    
    doors_section = format_doors_section(closed_doors) if closed_doors else ""
    
    user_prompt = USER_PROMPT.format(
        CURRENT_ROOM=current_room,
        LIST_NEARBY_OBJECTS=nearby_objects,
        STORY_SECTION=story_section,
        DISCOVERED_ROOMS_LIST=format_discovered_rooms(discovered_rooms),
        ROOMS_WITH_FRONTIER_DESCRIPTION=rooms_with_frontier,
        FULLY_EXPLORED_ROOMS=fully_explored_rooms,
        DOORS_SECTION=doors_section,
        ACTION_HISTORY_SECTION=action_history_section,
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
    
    # Formatting
    'STORY_SECTION_TEMPLATE',
    'STORY_EMPTY',
    'ACTION_SUCCESS',
    'ACTION_FAILED',
    'NO_HISTORY',
    'DOORS_SECTION_TEMPLATE',
    'NO_CLOSED_DOORS',
    
    # Guidance
    'GUIDANCE_TARGET_FOUND',
    'GUIDANCE_REPEATED_FAILURES',
    'GUIDANCE_ALL_EXPLORED',
    'GUIDANCE_UNEXPLORED',
    'GUIDANCE_DEFAULT',
    
    # Retry prompts
    'RETRY_PROMPT',
    'RETRY_PROMPT_GENERIC',
    'RETRY_PROMPT_FORMAT',
    
    # Helper functions
    'format_action_history',
    'format_action_history_from_dataclass',
    'format_discovered_rooms',
    'format_doors_section',
    'format_story_section',
    'get_decision_guidance',
    'format_retry_prompt',
    'count_recent_failures',
    
    # Storyteller helpers
    'format_rooms_for_storyteller',
    'format_actions_for_storyteller',
    'generate_fallback_story',
    # Aliases for v3 compatibility
    'format_discovered_rooms_for_storyteller',
    'format_action_history_for_storyteller',
    
    # Prompt builders
    'build_storyteller_prompt',
    'build_storyteller_update_prompt',
    'build_main_llm_prompt',
]

# Aliases for backwards compatibility with v3 naming
format_discovered_rooms_for_storyteller = format_rooms_for_storyteller
format_action_history_for_storyteller = format_actions_for_storyteller

