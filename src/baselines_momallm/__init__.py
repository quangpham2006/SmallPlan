"""
MoMa-LLM Baseline for Habitat ObjectNav

A baseline implementation following the MoMa-LLM paper's approach
for language-grounded navigation using dynamic scene graphs.

Reference: https://github.com/robot-learning-freiburg/MoMa-LLM

This implementation adapts MoMa-LLM's concepts to run on Habitat-Sim:
- Dynamic scene graph representation (rooms, objects, frontiers)
- LLM-based room classification
- Semantic action space: navigate(), explore(), go_to_and_open(), done()
- Thought/Action response format

Directory Structure:
    - core/: Core environment and simulator wrappers
    - agents/: MoMa-LLM style LLM navigation agent
    - tasks/: Task definitions (ObjectNav)
    - utils/: Utility functions and constants
"""

__version__ = "0.1.0"
