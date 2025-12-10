SYSTEM_PROMPT = '''
You are a robot in an unexplored house. Your task is to {TASK_DESCRIPTION}. 
You have the following action functions available to achieve this task: 
{TOOL_DESCRIPTIONS}

You will strictly follow this response format in your output:
 Analysis: Describe where you could find the objects of interest and what actions you need to execute to get there.
 Reasoning: Justify why the next action is important to solve the task.
 Command: function call in the format function_name(argument)
'''

USER_PROMPT = '''
You are currently in the {CURRENT_ROOM}. You are standing next to: {LIST_NEARBY_OBJECTS}.
You have found the following rooms and objects: 
{LIST_FOUND_ROOMS_AND_OBJECTS}
{LIST_PREVIOUS_ACTIONS}

Exploration status:
- Rooms with unexplored areas (can explore): {ROOMS_WITH_FRONTIER_DESCRIPTION}
- Fully explored rooms (nothing left to discover): {FULLY_EXPLORED_ROOMS}
- Rooms you have visited: {VISITED_ROOMS}
{ROOMS_WITH_CLOSED_DOORS_DESCRIPTION}

Choose your next action wisely:
- If the target object is already in the list above, call stop() immediately.
- If you have explored the same room multiple times without finding the target, consider opening a door to discover new rooms.
- Doors lead to NEW rooms that may contain the target - don't ignore them.
- If actions fail repeatedly, try different targets or actions.
- Use exact object/room names as shown above.
- Only explore rooms listed as having unexplored areas. Do NOT explore fully explored rooms.
'''

ROOM_CLASSIFICATION_SYSTEM_PROMPT = '''
You are a helpful assistant, visiting a new apartment.
'''

ROOM_CLASSIFICATION_USER_PROMPT = '''
You observe {NUM_ROOMS} rooms, they contain the following objects:
{ROOM_OBJECT_LIST}

{REQUESTS}

You return a list with bullet points following this output response format:
 - room-X: room type

{REMEMBER}
DO NOT include any other text in your response.
'''
RETRIAL_PROMPT = "The last action failed. Please try another command based on the previous message feedback. Note that you must have 'command:' before action."
# RETRIAL_PROMPT = "The last action {ACTION}({ARGS}) failed. The reason of failure is {LAST_ENV_FEEDBACK}. Follow the analysis, reasoning and response also."
RETRIAL_PROMPT_FORMAT_ERROR = "Feedback: The last action cannot be executed due to logical errors or format errors in your previous response. Remember to strictly follow the response format and use the available functions only."