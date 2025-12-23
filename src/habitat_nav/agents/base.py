"""
Base Agent for Habitat Navigation

Abstract base class for navigation agents.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from ..utils.actions import Action, HighLevelAction
from ..core.observations import ProcessedObservation

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """
    Agent internal state tracking.
    """
    # Action history
    action_history: List[Tuple[str, str, bool]] = field(default_factory=list)
    
    # Visited locations
    visited_positions: List[np.ndarray] = field(default_factory=list)
    
    # Observed objects
    seen_objects: set = field(default_factory=set)
    
    # Task info
    target_category: str = ""
    current_room: str = ""
    
    # Metrics
    total_steps: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    
    def reset(self, target_category: str = ""):
        """Reset agent state for new episode."""
        self.action_history = []
        self.visited_positions = []
        self.seen_objects = set()
        self.target_category = target_category
        self.current_room = ""
        self.total_steps = 0
        self.successful_actions = 0
        self.failed_actions = 0
    
    def add_action(self, action: str, argument: str, success: bool):
        """Record an action in history."""
        self.action_history.append((action, argument, success))
        self.total_steps += 1
        if success:
            self.successful_actions += 1
        else:
            self.failed_actions += 1
    
    def add_observation(self, obs: ProcessedObservation):
        """Update state with new observation."""
        if obs.visible_objects:
            self.seen_objects.update(obs.visible_objects)
        self.visited_positions.append(obs.position.copy())
    
    def get_action_summary(self, max_actions: int = 5) -> str:
        """Get summary of recent actions."""
        recent = self.action_history[-max_actions:]
        if not recent:
            return "No actions taken yet."
        
        lines = []
        for action, arg, success in recent:
            status = "success" if success else "failed"
            if arg:
                lines.append(f"{action}({arg}) - {status}")
            else:
                lines.append(f"{action}() - {status}")
        
        return ", ".join(lines)


class BaseAgent(ABC):
    """
    Abstract base class for navigation agents.
    
    Subclasses must implement:
    - act(): Select action given observation
    - reset(): Reset agent state
    """
    
    def __init__(self, name: str = "base_agent"):
        """
        Initialize base agent.
        
        Args:
            name: Agent name for logging
        """
        self.name = name
        self.state = AgentState()
        
    @abstractmethod
    def act(self,
            observation: ProcessedObservation,
            task_description: str,
            info: Dict[str, Any]) -> Action | HighLevelAction:
        """
        Select an action given the current observation.
        
        Args:
            observation: Current processed observation
            task_description: Natural language task description
            info: Additional info from environment
            
        Returns:
            Action to execute (low-level Action or HighLevelAction)
        """
        raise NotImplementedError
    
    @abstractmethod
    def reset(self, target_category: str = ""):
        """
        Reset agent state for a new episode.
        
        Args:
            target_category: Target object category for the episode
        """
        raise NotImplementedError
    
    def update(self, 
               action: Action | HighLevelAction,
               observation: ProcessedObservation,
               reward: float,
               info: Dict[str, Any]):
        """
        Update agent after taking an action.
        
        Args:
            action: Action that was taken
            observation: Resulting observation
            reward: Reward received
            info: Environment info
        """
        self.state.add_observation(observation)
    
    def is_goal_reached(self, observation: ProcessedObservation) -> bool:
        """
        Check if the agent believes the goal is reached.
        
        Args:
            observation: Current observation
            
        Returns:
            True if agent believes goal is reached
        """
        if not self.state.target_category:
            return False
        
        if observation.visible_objects:
            return self.state.target_category in observation.visible_objects
        
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return {
            "total_steps": self.state.total_steps,
            "successful_actions": self.state.successful_actions,
            "failed_actions": self.state.failed_actions,
            "objects_seen": len(self.state.seen_objects),
        }

