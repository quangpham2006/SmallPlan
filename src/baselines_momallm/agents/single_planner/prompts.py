# =============================================================================
# MoMa-LLM Style Prompts for Language-Grounded Navigation
# Reference: https://github.com/robot-learning-freiburg/MoMa-LLM
# =============================================================================

# =============================================================================
# HIGH-LEVEL ACTION PROMPTS (MoMa-LLM Style with Scene Graph)
# =============================================================================

HIGH_LEVEL_SYSTEM_PROMPT = """You are a mobile robot tasked with interactive object search in an indoor environment. Your task is to {task_description}.

You have access to a dynamic scene graph that represents your understanding of the environment:
- ROOMS: Discovered rooms with their classified types
- OBJECTS: Objects observed in each room
- FRONTIERS: Unexplored areas that can be reached

Available actions:
{action_descriptions}

Response format (follow STRICTLY):
Thought: Analyze the current scene graph and reason about where the target might be located.
Action: function_name(argument)

Important navigation rules:
1. Use semantic reasoning about room types - e.g., a "bed" is likely in a "bedroom"
2. If you see the target object, navigate directly to it
3. Explore rooms that are likely to contain the target based on their type
4. Open closed doors to discover new rooms when current rooms are exhausted
5. Learn from failed actions - do not repeat the same failing action
"""

HIGH_LEVEL_USER_PROMPT = """=== SCENE GRAPH ===

CURRENT LOCATION: {current_room}

VISIBLE OBJECTS (with distances):
{nearby_objects}

DISCOVERED ROOMS AND CONTENTS:
{discovered_rooms}

FRONTIERS (unexplored areas):
{unexplored_info}

=== NAVIGATION STATE ===

ACTION HISTORY:
{action_history}

Last action result: {last_feedback}

{exploration_status}

=== TASK ===

Target object: {target_object}

{decision_guidance}

Based on the scene graph and your reasoning about where a "{target_object}" would likely be found, what action should you take?"""


# =============================================================================
# LOW-LEVEL ACTION PROMPTS
# =============================================================================

LOW_LEVEL_SYSTEM_PROMPT = """You are a mobile robot navigating an indoor environment. Your task is to {task_description}.

Available actions:
{action_descriptions}

Response format (follow strictly):
Thought: Brief analysis of current observation and target location.
Action: ACTION_NAME

Navigation rules:
- Move towards the target object when visible
- Turn to look around when the target is not visible
- Use backward movement to avoid obstacles
- Stop only when close to the target object
"""

LOW_LEVEL_USER_PROMPT = """=== OBSERVATION ===

Current room: {current_room}

Visible objects:
{nearby_objects}

Discovered rooms:
{discovered_rooms}

Unexplored directions: {unexplored_info}
{exploration_status}

=== HISTORY ===
{action_history}
Last result: {last_feedback}

=== TASK ===
Find: {target_object}

What is your next action?"""


# =============================================================================
# ROOM CLASSIFICATION PROMPTS (MoMa-LLM Style)
# Reference: Language-Grounded Dynamic Scene Graphs
# =============================================================================

ROOM_CLASSIFICATION_SYSTEM_PROMPT = """You are a scene understanding system that classifies rooms based on their observed contents.

Given a list of objects observed in a room, classify the room into one of these categories:
bathroom, bedroom, closet, corridor, dining room, entryway, garage, hallway, kitchen, 
laundry room, living room, office, other room, outdoor, stairs

Use semantic reasoning about typical object-room associations:
- Beds, nightstands, wardrobes → bedroom
- Toilets, sinks, bathtubs → bathroom
- Stoves, refrigerators, counters → kitchen
- Sofas, TVs, coffee tables → living room
- Desks, computers, bookshelves → office"""

ROOM_CLASSIFICATION_USER_PROMPT = """Classify the following {num_rooms} room(s) based on their observed objects:

{room_object_list}

{request}

Respond with a classification for each room in this exact format:
 - room-X: room_type

{remember}
Provide ONLY the classifications, no additional explanation."""


# =============================================================================
# SCENE GRAPH FORMATTING (MoMa-LLM Style)
# =============================================================================

def format_scene_graph_rooms(
    discovered_rooms: dict,
    room_exploration_status: dict = None
) -> str:
    """
    Format discovered rooms as a scene graph representation.
    
    MoMa-LLM style: Shows rooms as graph nodes with object contents.
    
    Args:
        discovered_rooms: Dict mapping room display name -> list of object names
        room_exploration_status: Dict mapping room display name -> is_fully_explored
        
    Returns:
        Formatted scene graph string
    """
    if not discovered_rooms:
        return "  No rooms discovered yet. Use explore() to discover rooms."
    
    room_exploration_status = room_exploration_status or {}
    
    lines = []
    for room_name, objects in sorted(discovered_rooms.items()):
        if not objects:
            continue
        
        # Deduplicate and sort objects
        unique_objects = sorted(set(objects))[:12]
        obj_list = ", ".join(unique_objects)
        
        is_explored = room_exploration_status.get(room_name, False)
        status = "[fully explored]" if is_explored else "[has unexplored areas]"
        
        lines.append(f"  • {room_name} {status}")
        lines.append(f"    Objects: {obj_list}")
    
    return "\n".join(lines) if lines else "  No rooms discovered yet."


def format_frontiers(frontier_info: list) -> str:
    """Format frontier information for the prompt."""
    if not frontier_info:
        return "  No unexplored frontiers from current position."
    
    return "  " + ", ".join(frontier_info[:5])


# =============================================================================
# ACTION HISTORY FORMATTING (MoMa-LLM Style)
# =============================================================================

ACTION_SUCCESS_TEMPLATE = "  ✓ {action}({argument}) - success"
ACTION_FAILED_TEMPLATE = "  ✗ {action}({argument}) - failed: {reason}"
NO_HISTORY_MESSAGE = "  No actions taken yet."
ACTION_HISTORY_HEADER = "Recent actions:"


def format_action_history(action_history: list, max_items: int = 5) -> str:
    """
    Format action history with success/failure status.
    
    Args:
        action_history: List of tuples (action_name, argument, success, feedback)
        max_items: Maximum number of history items to show
        
    Returns:
        Formatted string for the prompt
    """
    if not action_history:
        return NO_HISTORY_MESSAGE
    
    recent = action_history[-max_items:][::-1]
    
    lines = [ACTION_HISTORY_HEADER]
    for item in recent:
        if len(item) >= 4:
            action, argument, success, feedback = item[:4]
        elif len(item) >= 3:
            action, argument, success = item[:3]
            feedback = "unknown"
        else:
            continue
        
        if success:
            lines.append(ACTION_SUCCESS_TEMPLATE.format(action=action, argument=argument))
        else:
            reason = feedback if feedback else "unknown"
            lines.append(ACTION_FAILED_TEMPLATE.format(action=action, argument=argument, reason=reason))
    
    return '\n'.join(lines)


# =============================================================================
# DECISION GUIDANCE (MoMa-LLM Semantic Reasoning Style)
# =============================================================================

GUIDANCE_TARGET_FOUND = """TARGET VISIBLE: The "{target}" has been detected in your current view.
→ Use navigate({target}) to approach it and complete the task."""

GUIDANCE_SEMANTIC_SEARCH = """SEMANTIC SEARCH: Consider which room type would typically contain a "{target}".
→ Navigate to rooms of that type, or explore to discover more rooms."""

GUIDANCE_REPEATED_FAILURES = """REPEATED FAILURES: Recent actions have failed multiple times.
→ Try a different strategy: explore new areas, open doors, or navigate to different rooms.
→ If the target cannot be found, use done() to terminate."""

GUIDANCE_ALL_EXPLORED = """ALL ROOMS EXPLORED: All discovered rooms have been fully explored.
→ Look for closed doors to discover new rooms.
→ The target may be in an undiscovered area."""

GUIDANCE_UNEXPLORED_AREAS = """UNEXPLORED AREAS: There are still unexplored regions in some rooms.
→ Use explore(room) to search these areas for the target."""


def get_decision_guidance(
    target_found: bool = False,
    target_name: str = "",
    recent_failure_count: int = 0,
    has_unexplored_areas: bool = True,
    all_rooms_explored: bool = False
) -> str:
    """
    Generate MoMa-LLM style decision guidance based on current state.
    """
    if target_found:
        return GUIDANCE_TARGET_FOUND.format(target=target_name)
    
    if recent_failure_count >= 3:
        return GUIDANCE_REPEATED_FAILURES
    
    if all_rooms_explored:
        return GUIDANCE_ALL_EXPLORED
    
    if has_unexplored_areas:
        return GUIDANCE_UNEXPLORED_AREAS
    
    return GUIDANCE_SEMANTIC_SEARCH.format(target=target_name)


# =============================================================================
# RETRY/FAILURE PROMPTS
# =============================================================================

RETRIAL_PROMPT = """The last action failed: {failure_reason}

Consider alternative approaches:
- Navigate to a different target
- Explore a different room
- Open a door to discover new areas

Respond with: Thought: [reasoning] Action: function_name(argument)"""

RETRIAL_PROMPT_GENERIC = """The previous action failed. Try a different approach.

Do not repeat the same failing action. Consider:
- Navigating to a different object or room
- Exploring unexplored areas
- Opening doors to find new rooms"""

RETRIAL_PROMPT_FORMAT_ERROR = """Response format error. Please use this exact format:

Thought: [Your reasoning about where the target might be]
Action: function_name(argument)

Available actions: navigate(target), explore(room), go_to_and_open(door), done()"""


def format_retry_prompt(failure_reason: str = None) -> str:
    """Generate retry prompt with optional failure reason."""
    if failure_reason:
        return RETRIAL_PROMPT.format(failure_reason=failure_reason)
    return RETRIAL_PROMPT_GENERIC


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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
    """
    Format discovered rooms for the prompt (wrapper for scene graph format).
    """
    return format_scene_graph_rooms(discovered_rooms, room_exploration_status)


# =============================================================================
# ALIASES FOR BACKWARD COMPATIBILITY
# =============================================================================

SYSTEM_PROMPT = HIGH_LEVEL_SYSTEM_PROMPT
USER_PROMPT = HIGH_LEVEL_USER_PROMPT
