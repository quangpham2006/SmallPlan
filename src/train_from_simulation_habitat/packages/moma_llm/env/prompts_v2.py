"""
Improved prompts for Habitat LLM environment (v2).

Key improvements over v1:
1. Better organized memory context with clear sections
2. Failed action feedback with specific failure reasons integrated into history
3. More concise format to reduce token usage
4. Clearer guidance on when to stop vs continue exploring
"""

# =============================================================================
# SYSTEM PROMPT
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
'''

# =============================================================================
# USER PROMPT - Main context for each decision
# =============================================================================

USER_PROMPT = '''Current location: {CURRENT_ROOM}
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
# DECISION GUIDANCE TEMPLATES - Dynamic based on state
# =============================================================================

# When target object is found
GUIDANCE_TARGET_FOUND = '''🎯 TARGET FOUND! The target "{target}" is in the list above.
→ Call stop() immediately to complete the task.'''

# When there are repeated failures
GUIDANCE_REPEATED_FAILURES = '''⚠️ Recent actions have been failing repeatedly.
→ Try a DIFFERENT approach: open a door, explore a different room, or use a different target.
→ If you struggle too long and cannot find the target, call stop() to terminate the task.'''

# When all rooms are explored but target not found
GUIDANCE_ALL_EXPLORED = '''All discovered rooms have been fully explored but target not found.
→ Focus on opening closed doors to discover new rooms that may contain the target.'''

# When there are unexplored areas
GUIDANCE_UNEXPLORED = '''There are still unexplored areas in discovered rooms.
→ Consider exploring rooms with unexplored areas OR opening doors to find new rooms.'''

# Default guidance
GUIDANCE_DEFAULT = '''Choose the most efficient action to find the target:
- If target is in the list → stop()
- If target might be in unexplored areas → explore(room)
- If target might be behind closed doors → open(door)'''

# =============================================================================
# RETRY/FAILURE PROMPTS
# =============================================================================

# When last action failed - includes specific feedback
RETRIAL_PROMPT = '''The last action failed: {FAILURE_REASON}

Please choose a different action. Consider:
- Trying a different target (door, room, or object)
- Using a different action type
- The target might be inaccessible from your current position

Remember to include "Command:" before your action.'''

# Generic retry without specific reason
RETRIAL_PROMPT_GENERIC = '''The last action failed. Please try a different approach.

Tips:
- Don't repeat the exact same action that just failed
- Try a different target or action type
- Check if there are alternative paths or objects

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
# HELPER FUNCTIONS
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
        target_found: Whether the target object has been found in discovered objects
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


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def format_action_history_from_dataclass(action_history: list, llm_to_human_readable, max_items: int = 5) -> str:
    """
    Format action history from ActionHistory dataclass objects.
    
    This is designed to work with the existing ActionHistory dataclass in habitat_llm_env.py:
    
    @dataclass
    class ActionHistory:
        action: str
        object_name_graph: str
        position: tuple
        subtask_success: bool
        opendoors_roompos: Any = None
        orig_api_call: str = None
        feedback: str = None  # <-- Add this field to store failure reasons
    
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
            # Parse argument from original call
            import re
            match = re.search(r'\(([^)]*)\)', h.orig_api_call)
            argument = match.group(1) if match else ""
        else:
            argument = ""
        
        if h.subtask_success:
            lines.append(ACTION_SUCCESS_TEMPLATE.format(action=action, argument=argument))
        else:
            # Use feedback if available, otherwise generic message
            reason = getattr(h, 'feedback', None) or "navigation/execution failed"
            lines.append(ACTION_FAILED_TEMPLATE.format(action=action, argument=argument, reason=reason))
    
    return '\n'.join(lines)


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
            break  # Stop at first success
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
# EXAMPLE USAGE (Integration Guide)
# =============================================================================
"""
To integrate prompts_v2 with habitat_llm_env.py:

1. Import the new prompts:
   ```python
   from moma_llm.env.prompts_v2 import (
       SYSTEM_PROMPT, USER_PROMPT, 
       RETRIAL_PROMPT_FORMAT_ERROR,
       format_action_history_from_dataclass,
       get_decision_guidance,
       format_retry_prompt,
       count_recent_failures,
       check_target_in_objects
   )
   ```

2. Modify ActionHistory dataclass to include feedback:
   ```python
   @dataclass
   class ActionHistory:
       action: str
       object_name_graph: str
       position: tuple
       subtask_success: bool
       opendoors_roompos: Any = None
       orig_api_call: str = None
       feedback: str = None  # Add this field
   ```

3. When executing actions, store the feedback:
   ```python
   history = ActionHistory(
       action="open",
       object_name_graph=obj.name if obj else None,
       position=position_2d,
       subtask_success=subtask_success,
       feedback=feedback  # Store the failure reason
   )
   ```

4. In _create_prompt, use the new formatting:
   ```python
   # Format action history with failure reasons
   action_history_section = format_action_history_from_dataclass(
       self.action_history,
       self.llm.to_human_readable_object_name
   )
   
   # Generate contextual guidance
   target_found = check_target_in_objects(target_name, room_dict)
   recent_failures = count_recent_failures(self.action_history)
   has_unexplored = len(rooms_with_frontier_leading_out) > 0
   all_explored = len(fully_explored_rooms) == len(labelled_rooms)
   
   decision_guidance = get_decision_guidance(
       target_found=target_found,
       target_name=target_name,
       recent_failure_count=recent_failures,
       has_unexplored_areas=has_unexplored,
       all_rooms_explored=all_explored
   )
   
   user_prompt = USER_PROMPT.format(
       CURRENT_ROOM=current_room,
       LIST_NEARBY_OBJECTS=list_nearby_objects,
       LIST_FOUND_ROOMS_AND_OBJECTS=list_found_rooms_and_objects,
       ACTION_HISTORY_SECTION=action_history_section,
       ROOMS_WITH_FRONTIER_DESCRIPTION=rooms_with_frontier_descr,
       FULLY_EXPLORED_ROOMS=fully_explored_descr,
       VISITED_ROOMS=visited_rooms_descr,
       ROOMS_WITH_CLOSED_DOORS_DESCRIPTION=rooms_with_closed_doors_descr,
       DECISION_GUIDANCE=decision_guidance
   )
   ```

5. For retry prompts with failure feedback:
   ```python
   # Instead of:
   conversation.add_message({"role": "user", "content": RETRIAL_PROMPT})
   
   # Use:
   retry_msg = format_retry_prompt(failure_reason=last_feedback)
   conversation.add_message({"role": "user", "content": retry_msg})
   ```
"""

