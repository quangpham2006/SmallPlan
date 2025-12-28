"""
Action Definitions for MoMa-LLM Baseline on Habitat

Low-level discrete action space:
- Action 0: MOVE_FORWARD (0.25m forward)
- Action 1: MOVE_BACKWARD (0.25m backward)
- Action 2: TURN_LEFT (30 degrees)
- Action 3: TURN_RIGHT (30 degrees)
- Action 4: STOP (terminate episode)

High-level actions (MoMa-LLM style):
- navigate(target): Navigate to object or room
- go_to_and_open(target): Navigate to and open a door
- explore(room): Explore a room
- done(): Task completion

Reference: https://github.com/robot-learning-freiburg/MoMa-LLM
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional
import re


class Action(IntEnum):
    """
    Discrete navigation actions for Habitat-Sim.
    """
    MOVE_FORWARD = 0
    MOVE_BACKWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    STOP = 4
    
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


DISCRETE_ACTIONS: Dict[int, ActionSpec] = {
    0: ActionSpec("MOVE_FORWARD", 0, "Move forward 0.25 meters", "move_forward"),
    1: ActionSpec("MOVE_BACKWARD", 1, "Move backward 0.25 meters", "move_backward"),
    2: ActionSpec("TURN_LEFT", 2, "Turn left 30 degrees", "turn_left"),
    3: ActionSpec("TURN_RIGHT", 3, "Turn right 30 degrees", "turn_right"),
    4: ActionSpec("STOP", 4, "Stop and indicate task completion", "stop"),
}


class ActionSpace:
    """
    Action space manager for Habitat navigation.
    """
    
    def __init__(self, forward_step: float = 0.25, turn_angle: float = 30.0):
        self.forward_step = forward_step
        self.turn_angle = turn_angle
        self.n_actions = len(DISCRETE_ACTIONS)
        
    def get_action(self, action_id: int) -> Action:
        if 0 <= action_id < self.n_actions:
            return Action(action_id)
        raise ValueError(f"Invalid action ID: {action_id}")
    
    def get_habitat_action(self, action: Action | int) -> str:
        if isinstance(action, int):
            action = self.get_action(action)
        return action.to_habitat_action()
    
    @property
    def action_names(self) -> list:
        return [spec.name for spec in DISCRETE_ACTIONS.values()]
    
    def sample(self) -> Action:
        import random
        return Action(random.randint(0, 3))
    
    def __len__(self) -> int:
        return self.n_actions
    
    @staticmethod
    def get_action_descriptions() -> str:
        lines = []
        for action_id, spec in DISCRETE_ACTIONS.items():
            lines.append(f"- {spec.name}: {spec.description}")
        return "\n".join(lines)
    
    @staticmethod
    def parse_action(action_str: str) -> Optional[Action]:
        action_str = action_str.strip().upper().replace(" ", "_")
        try:
            return Action[action_str]
        except KeyError:
            return Action.from_name(action_str)


# =============================================================================
# High-Level Actions (MoMa-LLM Style)
# Reference: https://github.com/robot-learning-freiburg/MoMa-LLM
# =============================================================================

@dataclass
class HighLevelAction:
    """
    High-level action for MoMa-LLM style navigation.
    
    Uses MoMa-LLM action naming conventions:
    - navigate(target): Navigate to object or room
    - go_to_and_open(target): Navigate to and open a door
    - explore(room): Explore a room  
    - done(): Task completion
    """
    name: str
    argument: str
    description: str = ""


class HighLevelActionSpace:
    """
    High-level action space aligned with MoMa-LLM.
    
    Action naming follows MoMa-LLM conventions:
    - navigate() instead of goto()
    - go_to_and_open() instead of open()
    - done() instead of stop()
    
    Reference: https://github.com/robot-learning-freiburg/MoMa-LLM
    """
    
    # MoMa-LLM style actions
    ACTIONS = {
        "navigate": ("target", "Navigate to a target object or room. Example: navigate(sofa), navigate(kitchen)"),
        "go_to_and_open": ("object_name", "Navigate to and open a door or container. Example: go_to_and_open(door)"),
        "explore": ("room_name", "Explore unexplored areas in a room. Example: explore(bedroom)"),
        "done": ("", "Call when the task is completed or cannot be completed. Example: done()"),
    }
    
    @classmethod
    def parse(cls, action_str: str) -> Optional[HighLevelAction]:
        """
        Parse action string into HighLevelAction.
        
        Supports MoMa-LLM style: navigate(kitchen), go_to_and_open(door), done()
        Also supports legacy names for compatibility: goto(), open(), stop()
        """
        action_str = action_str.strip()
        
        # Match action(argument) pattern
        match = re.match(r"(\w+)\(([^)]*)\)", action_str)
        if not match:
            if action_str.lower() in ("done", "stop"):
                return HighLevelAction(name="done", argument="", description="Task completed")
            return None
        
        action_name, argument = match.groups()
        action_name = action_name.lower()
        
        # Map legacy action names to MoMa-LLM style
        action_mapping = {
            "goto": "navigate",
            "open": "go_to_and_open",
            "stop": "done"
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
        """Get formatted action descriptions for prompts (MoMa-LLM style)."""
        lines = []
        for i, (name, (arg, desc)) in enumerate(cls.ACTIONS.items(), 1):
            if arg:
                lines.append(f"{i}. {name}({arg}): {desc}")
            else:
                lines.append(f"{i}. {name}(): {desc}")
        return "\n".join(lines)
