"""
Random Navigation Agent

Baseline agent that selects random actions.
"""

import logging
from typing import Dict, Any

import numpy as np

from .base import BaseAgent, AgentState
from ..utils.actions import Action, HighLevelAction
from ..core.observations import ProcessedObservation

logger = logging.getLogger(__name__)


class RandomAgent(BaseAgent):
    """
    Random baseline agent.
    
    Selects actions uniformly at random (excluding STOP unless explicitly allowed).
    Useful as a baseline comparison.
    """
    
    def __init__(self,
                 seed: int = 42,
                 stop_probability: float = 0.0,
                 name: str = "random_agent"):
        """
        Initialize random agent.
        
        Args:
            seed: Random seed
            stop_probability: Probability of selecting STOP action
            name: Agent name
        """
        super().__init__(name=name)
        self.rng = np.random.RandomState(seed)
        self.stop_probability = stop_probability
    
    def act(self,
            observation: ProcessedObservation,
            task_description: str,
            info: Dict[str, Any]) -> Action:
        """
        Select a random action.
        
        Args:
            observation: Current observation (ignored)
            task_description: Task description (ignored)
            info: Environment info (ignored)
            
        Returns:
            Random action
        """
        # Check for goal
        if self.is_goal_reached(observation):
            return Action.STOP
        
        # Maybe stop randomly
        if self.rng.random() < self.stop_probability:
            return Action.STOP
        
        # Random navigation action
        return Action(self.rng.randint(0, 3))
    
    def reset(self, target_category: str = ""):
        """Reset agent state."""
        self.state.reset(target_category)
        logger.debug(f"RandomAgent reset for target: {target_category}")

