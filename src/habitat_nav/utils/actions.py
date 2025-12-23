"""
Action Definitions for Habitat Navigation

Follows Nav-R1 discrete action space convention:
- Action 0: MOVE_FORWARD (0.25m forward)
- Action 1: TURN_LEFT (30 degrees)
- Action 2: TURN_RIGHT (30 degrees)
- Action 3: STOP (terminate episode)

Reference: https://github.com/AIGeeksGroup/Nav-R1
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional


class Action(IntEnum):
    """
    Discrete navigation actions following Nav-R1 convention.
    
    These are low-level actions that the agent can execute in the Habitat simulator.
    """
    MOVE_FORWARD = 0   # Move forward 0.25 meters
    TURN_LEFT = 1      # Turn left 30 degrees
    TURN_RIGHT = 2     # Turn right 30 degrees  
    STOP = 3           # Terminate and indicate task completion
    
    @classmethod
    def from_name(cls, name: str) -> "Action":
        """Convert action name to Action enum."""
        name_upper = name.upper().replace(" ", "_")
        name_map = {
            "MOVE_FORWARD": cls.MOVE_FORWARD,
            "FORWARD": cls.MOVE_FORWARD,
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
        name="TURN_LEFT", 
        action_id=1,
        description="Turn left 30 degrees",
        habitat_action="turn_left"
    ),
    2: ActionSpec(
        name="TURN_RIGHT",
        action_id=2, 
        description="Turn right 30 degrees",
        habitat_action="turn_right"
    ),
    3: ActionSpec(
        name="STOP",
        action_id=3,
        description="Stop and indicate task completion",
        habitat_action="stop"
    ),
}


class ActionSpace:
    """
    Action space manager for Habitat navigation.
    
    Provides mapping between different action representations:
    - Integer IDs (0-3)
    - Action enums
    - Habitat-Sim action strings
    """
    
    def __init__(self, 
                 forward_step: float = 0.25,
                 turn_angle: float = 30.0):
        """
        Initialize action space.
        
        Args:
            forward_step: Distance to move forward in meters
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
        return Action(random.randint(0, 2))  # Exclude STOP from random sampling
    
    def __len__(self) -> int:
        return self.n_actions


# High-level actions for LLM agent
@dataclass
class HighLevelAction:
    """
    High-level action that LLM agents output.
    
    These get decomposed into sequences of discrete actions.
    """
    name: str
    argument: str
    description: str


class HighLevelActionSpace:
    """
    High-level action space for LLM-based navigation.
    
    Maps semantic actions to low-level navigation commands.
    """
    
    ACTIONS = {
        "goto": ("target", "Navigate to a target object or room. Example: goto(sofa), goto(kitchen)"),
        "explore": ("room", "Explore unexplored areas in a room. Example: explore(bedroom)"),
        "open": ("object", "Open a door or container. Example: open(door), open(cabinet)"),
        "stop": ("", "Terminate task when goal is reached or no further progress possible."),
    }
    
    @classmethod
    def parse(cls, action_str: str) -> Optional[HighLevelAction]:
        """
        Parse action string into HighLevelAction.
        
        Args:
            action_str: Action string like "goto(kitchen)" or "stop()"
            
        Returns:
            HighLevelAction or None if parsing fails
        """
        import re
        match = re.match(r"(\w+)\(([^)]*)\)", action_str.strip())
        if not match:
            return None
        
        action_name, argument = match.groups()
        action_name = action_name.lower()
        
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

