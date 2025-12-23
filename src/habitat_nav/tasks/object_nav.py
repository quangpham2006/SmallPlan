"""
Object Navigation Task

Defines the ObjectNav task where an agent must find an instance of a target object category.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

from ..utils.constants import EXCLUDED_OBJECT_CATEGORIES

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of a navigation task."""
    success: bool
    distance_to_target: float
    steps_taken: int
    distance_travelled: float
    shortest_path_distance: Optional[float]
    spl: float  # Success weighted by Path Length
    failure_reason: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "distance_to_target": self.distance_to_target,
            "steps_taken": self.steps_taken,
            "distance_travelled": self.distance_travelled,
            "shortest_path_distance": self.shortest_path_distance,
            "spl": self.spl,
            "failure_reason": self.failure_reason,
        }


@dataclass
class ObjectNavTask:
    """
    Object Goal Navigation Task.
    
    The agent must navigate to find an instance of a target object category.
    The task is successful when the agent is within a distance threshold of
    the target and the target is visible.
    """
    target_category: str
    success_distance: float = 1.5
    max_steps: int = 500
    require_visibility: bool = True
    
    # Task state
    start_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    shortest_path_distance: Optional[float] = None
    target_positions: List[np.ndarray] = field(default_factory=list)
    
    @property
    def description(self) -> str:
        """Human-readable task description."""
        return f"Find a {self.target_category}"
    
    @classmethod
    def sample_task(cls,
                    scene,
                    agent_position: np.ndarray,
                    rng: np.random.RandomState,
                    success_distance: float = 1.5,
                    max_steps: int = 500) -> "ObjectNavTask":
        """
        Sample a random ObjectNav task for the scene.
        
        Args:
            scene: Scene wrapper with object information
            agent_position: Current agent position
            rng: Random number generator
            success_distance: Distance threshold for success
            max_steps: Maximum episode steps
            
        Returns:
            ObjectNavTask instance
        """
        # Get valid categories
        valid_categories = cls.get_valid_categories(scene)
        
        if not valid_categories:
            logger.warning("No valid target categories found, using 'chair'")
            return cls(
                target_category="chair",
                success_distance=success_distance,
                max_steps=max_steps,
                start_position=agent_position.copy(),
            )
        
        # Sample target category
        target_category = rng.choice(list(valid_categories))
        
        # Get target positions
        target_objects = scene.get_objects_by_category(target_category)
        target_positions = [obj.get_position() for obj in target_objects]
        
        # Compute shortest path distance
        if target_positions:
            distances = [
                np.linalg.norm(agent_position - pos)
                for pos in target_positions
            ]
            shortest_dist = min(distances)
        else:
            shortest_dist = None
        
        return cls(
            target_category=target_category,
            success_distance=success_distance,
            max_steps=max_steps,
            start_position=agent_position.copy(),
            shortest_path_distance=shortest_dist,
            target_positions=target_positions,
        )
    
    @staticmethod
    def get_valid_categories(scene) -> Set[str]:
        """
        Get valid object categories for navigation targets.
        
        Filters out structural elements and invalid categories.
        """
        if scene is None or not hasattr(scene, 'category_ids'):
            return set()
        
        valid = set()
        for category in scene.category_ids:
            cat_lower = category.lower()
            
            # Check exclusions
            if cat_lower in EXCLUDED_OBJECT_CATEGORIES:
                continue
            
            # Check for partial matches
            excluded = False
            for ex in ["wall", "floor", "ceiling", "void", "unknown"]:
                if ex in cat_lower:
                    excluded = True
                    break
            
            if not excluded:
                valid.add(category)
        
        return valid
    
    def check_success(self,
                      agent_position: np.ndarray,
                      visible_objects: Optional[Set[str]] = None) -> bool:
        """
        Check if the task is successfully completed.
        
        Args:
            agent_position: Current agent position
            visible_objects: Set of visible object categories
            
        Returns:
            True if task is successful
        """
        # Check distance to any target position
        for target_pos in self.target_positions:
            distance = np.linalg.norm(agent_position - target_pos)
            if distance <= self.success_distance:
                # Check visibility if required
                if self.require_visibility:
                    if visible_objects and self.target_category in visible_objects:
                        return True
                else:
                    return True
        
        return False
    
    def get_distance_to_target(self, agent_position: np.ndarray) -> float:
        """Get distance to nearest target instance."""
        if not self.target_positions:
            return float('inf')
        
        distances = [
            np.linalg.norm(agent_position - pos)
            for pos in self.target_positions
        ]
        return min(distances)
    
    def compute_spl(self, 
                    distance_travelled: float,
                    success: bool) -> float:
        """
        Compute Success weighted by Path Length.
        
        SPL = success * (shortest / max(shortest, actual))
        """
        if not success:
            return 0.0
        
        if self.shortest_path_distance is None or self.shortest_path_distance <= 0:
            return 0.0
        
        return self.shortest_path_distance / max(
            self.shortest_path_distance, distance_travelled
        )
    
    def get_result(self,
                   agent_position: np.ndarray,
                   steps_taken: int,
                   distance_travelled: float,
                   success: bool,
                   failure_reason: Optional[str] = None) -> TaskResult:
        """
        Get final task result.
        
        Args:
            agent_position: Final agent position
            steps_taken: Number of steps taken
            distance_travelled: Total distance travelled
            success: Whether task was successful
            failure_reason: Reason for failure if applicable
            
        Returns:
            TaskResult with all metrics
        """
        distance_to_target = self.get_distance_to_target(agent_position)
        spl = self.compute_spl(distance_travelled, success)
        
        return TaskResult(
            success=success,
            distance_to_target=distance_to_target,
            steps_taken=steps_taken,
            distance_travelled=distance_travelled,
            shortest_path_distance=self.shortest_path_distance,
            spl=spl,
            failure_reason=failure_reason,
        )

