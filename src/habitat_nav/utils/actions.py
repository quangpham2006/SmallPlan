"""
Action Definitions for Habitat Navigation

Discrete action space:
- Action 0: MOVE_FORWARD (0.25m forward)
- Action 1: MOVE_BACKWARD (0.25m backward)
- Action 2: TURN_LEFT (30 degrees)
- Action 3: TURN_RIGHT (30 degrees)
- Action 4: STOP (terminate episode)
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional


class Action(IntEnum):
    """
    Discrete navigation actions.
    
    These are low-level actions that the agent can execute in the Habitat simulator.
    """
    MOVE_FORWARD = 0   # Move forward 0.25 meters
    MOVE_BACKWARD = 1  # Move backward 0.25 meters
    TURN_LEFT = 2      # Turn left 30 degrees
    TURN_RIGHT = 3     # Turn right 30 degrees  
    STOP = 4           # Terminate and indicate task completion
    
    @classmethod
    def from_name(cls, name: str) -> "Action":
        """Convert action name to Action enum."""
        name_upper = name.upper().replace(" ", "_")
        name_map = {
            "MOVE_FORWARD": cls.MOVE_FORWARD,
            "FORWARD": cls.MOVE_FORWARD,
            "MOVE_BACKWARD": cls.MOVE_BACKWARD,
            "BACKWARD": cls.MOVE_BACKWARD,
            "BACK": cls.MOVE_BACKWARD,
            "TURN_LEFT": cls.TURN_LEFT,
            "LEFT": cls.TURN_LEFT,
            "TURN_RIGHT": cls.TURN_RIGHT,
            "RIGHT": cls.TURN_RIGHT,
            "STOP": cls.STOP,
            "DONE": cls.STOP,
        }
        return name_map.get(name_upper, cls.STOP)
    
    def to_habitat_action(self) -> str:
        """Convert to Habitat-Sim action string."""
        action_map = {
            Action.MOVE_FORWARD: "move_forward",
            Action.MOVE_BACKWARD: "move_backward",
            Action.TURN_LEFT: "turn_left",
            Action.TURN_RIGHT: "turn_right",
            Action.STOP: "stop",
        }
        return action_map[self]


@dataclass
class ActionSpec:
    """Specification for an action."""
    name: str
    action_id: int
    description: str
    habitat_action: str


# Define the complete action space
DISCRETE_ACTIONS: Dict[int, ActionSpec] = {
    0: ActionSpec(
        name="MOVE_FORWARD",
        action_id=0,
        description="Move forward 0.25 meters",
        habitat_action="move_forward"
    ),
    1: ActionSpec(
        name="MOVE_BACKWARD",
        action_id=1,
        description="Move backward 0.25 meters",
        habitat_action="move_backward"
    ),
    2: ActionSpec(
        name="TURN_LEFT", 
        action_id=2,
        description="Turn left 30 degrees",
        habitat_action="turn_left"
    ),
    3: ActionSpec(
        name="TURN_RIGHT",
        action_id=3, 
        description="Turn right 30 degrees",
        habitat_action="turn_right"
    ),
    4: ActionSpec(
        name="STOP",
        action_id=4,
        description="Stop and indicate task completion",
        habitat_action="stop"
    ),
}


class ActionSpace:
    """
    Action space manager for Habitat navigation.
    
    Provides mapping between different action representations:
    - Integer IDs (0-4)
    - Action enums
    - Habitat-Sim action strings
    """
    
    def __init__(self, 
                 forward_step: float = 0.25,
                 turn_angle: float = 30.0):
        """
        Initialize action space.
        
        Args:
            forward_step: Distance to move forward/backward in meters
            turn_angle: Angle to turn in degrees
        """
        self.forward_step = forward_step
        self.turn_angle = turn_angle
        self.n_actions = len(DISCRETE_ACTIONS)
        
    def get_action(self, action_id: int) -> Action:
        """Get Action enum from action ID."""
        if 0 <= action_id < self.n_actions:
            return Action(action_id)
        raise ValueError(f"Invalid action ID: {action_id}. Must be 0-{self.n_actions-1}")
    
    def get_habitat_action(self, action: Action | int) -> str:
        """Convert action to Habitat-Sim action string."""
        if isinstance(action, int):
            action = self.get_action(action)
        return action.to_habitat_action()
    
    @property
    def action_names(self) -> list:
        """Get list of action names."""
        return [spec.name for spec in DISCRETE_ACTIONS.values()]
    
    def sample(self) -> Action:
        """Sample a random action (excluding STOP)."""
        import random
        return Action(random.randint(0, 3))  # Exclude STOP from random sampling
    
    def __len__(self) -> int:
        return self.n_actions
    
    @staticmethod
    def get_action_descriptions() -> str:
        """Get formatted action descriptions for LLM prompts."""
        lines = []
        for action_id, spec in DISCRETE_ACTIONS.items():
            lines.append(f"- {spec.name}: {spec.description}")
        return "\n".join(lines)
    
    @staticmethod
    def parse_action(action_str: str) -> Optional[Action]:
        """
        Parse action string from LLM response into Action.
        
        Args:
            action_str: Action string like "MOVE_FORWARD" or "move_forward"
            
        Returns:
            Action or None if parsing fails
        """
        action_str = action_str.strip().upper().replace(" ", "_")
        
        # Try direct enum name match
        try:
            return Action[action_str]
        except KeyError:
            pass
        
        # Try from_name mapping
        return Action.from_name(action_str)


# =============================================================================
# High-Level Actions for LLM Agent
# =============================================================================

@dataclass
class HighLevelAction:
    """
    High-level action that LLM agents output.
    
    These get decomposed into sequences of discrete actions.
    """
    name: str
    argument: str
    description: str = ""


class HighLevelActionSpace:
    """
    High-level action space for LLM-based navigation.
    
    Aligned with train_from_simulation_habitat action space.
    Maps semantic actions to low-level navigation commands.
    """
    
    ACTIONS = {
        "goto": ("target", "Navigate to a target object or room. Example: goto(sofa), goto(kitchen)"),
        "explore": ("room_name", "Explore unexplored areas in a room. Example: explore(bedroom), explore(kitchen)"),
        "open": ("object_name", "Navigate to and open a door or container. Example: open(door), open(cabinet)"),
        "stop": ("", "Call when the task is completed or no further actions possible. Example: stop()"),
    }
    
    @classmethod
    def parse(cls, action_str: str) -> Optional[HighLevelAction]:
        """
        Parse action string into HighLevelAction.
        
        Expected format: action_name(argument)
        Valid actions: goto, open, explore, stop
        
        Args:
            action_str: Action string like "goto(kitchen)" or "stop()"
            
        Returns:
            HighLevelAction or None if parsing fails
        """
        import re
        
        # Clean up the action string
        action_str = action_str.strip()
        
        # Try to match action(argument) pattern
        match = re.match(r"(\w+)\(([^)]*)\)", action_str)
        if not match:
            # Try without parentheses for stop
            if action_str.lower() in ("stop", "done"):
                return HighLevelAction(name="stop", argument="", description="Task completed")
            return None
        
        action_name, argument = match.groups()
        action_name = action_name.lower()
        
        # Handle legacy action names (backward compatibility with moma_llm)
        action_mapping = {
            "navigate": "goto",
            "go_to_and_open": "open",
            "done": "stop"
        }
        action_name = action_mapping.get(action_name, action_name)
        
        if action_name not in cls.ACTIONS:
            return None
        
        arg_name, description = cls.ACTIONS[action_name]
        return HighLevelAction(
            name=action_name,
            argument=argument.strip().strip("'\""),
            description=description
        )
    
    @classmethod
    def get_action_descriptions(cls) -> str:
        """Get formatted action descriptions for prompts."""
        lines = []
        for i, (name, (arg, desc)) in enumerate(cls.ACTIONS.items(), 1):
            if arg:
                lines.append(f"{i}. {name}({arg}): {desc}")
            else:
                lines.append(f"{i}. {name}(): {desc}")
        return "\n".join(lines)
