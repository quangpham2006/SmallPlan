# MoMa-LLM Baseline for Habitat

A baseline implementation following the [MoMa-LLM paper](https://github.com/robot-learning-freiburg/MoMa-LLM) approach for language-grounded navigation using dynamic scene graphs, adapted to run on Habitat-Sim.

## Reference

**MoMa-LLM: Language-Grounded Dynamic Scene Graphs for Interactive Object Search with Mobile Manipulation**

- Original Repository: https://github.com/robot-learning-freiburg/MoMa-LLM
- Original Environment: iGibson

## Key Differences from Original MoMa-LLM

| Aspect | Original MoMa-LLM | This Baseline |
|--------|------------------|---------------|
| Simulator | iGibson | Habitat-Sim |
| Scene Format | iGibson URDF | HM3D/MP3D GLB |
| Navigation | iGibson actions | Habitat waypoint teleportation |
| Scene Graph | Full DSG with regions | Simplified room-object graph |
| Mobile Manipulation | Supported | Navigation only |

## Action Space (MoMa-LLM Style)

This baseline uses MoMa-LLM's semantic action naming:

```python
# Navigate to a target object or room
navigate(target)    # e.g., navigate(sofa), navigate(kitchen)

# Explore unexplored areas in a room
explore(room)       # e.g., explore(bedroom)

# Navigate to and open a door
go_to_and_open(door)  # e.g., go_to_and_open(door)

# Complete the task
done()
```

## Prompt Format

The prompts follow MoMa-LLM's Thought/Action format:

```
Response format:
Thought: Analyze the current scene graph and reason about where the target might be located.
Action: function_name(argument)
```

## Room Classification

Uses LLM-based room classification similar to MoMa-LLM:
- Rooms are classified based on observed objects
- Categories: bathroom, bedroom, closet, corridor, dining room, entryway, garage, hallway, kitchen, laundry room, living room, office, other room, outdoor, stairs

## Usage

```bash
# Run with default settings (high-level actions, GPT-4o)
python -m src.baselines_momallm.inference --scene 00800-TEEsavR23oF --episodes 5

# Run with OpenAI API
python -m src.baselines_momallm.inference --agent llm-openai --model gpt-4o

# Custom output directory
python -m src.baselines_momallm.inference --output ./outputs/moma_llm
```

## Directory Structure

```
src/baselines_momallm/
├── agents/
│   ├── __init__.py
│   ├── base.py           # Base agent class
│   └── single_planner/
│       ├── llm_agent.py  # MoMa-LLM style LLM agent
│       └── prompts.py    # MoMa-LLM style prompts
├── core/
│   ├── simulator.py      # Habitat-Sim wrapper
│   ├── environment.py    # ObjectNav environment
│   ├── observations.py   # Observation processing
│   └── action_executor.py # High-level action executor
├── tasks/
│   └── object_nav.py     # ObjectNav task definition
├── utils/
│   ├── actions.py        # MoMa-LLM action definitions
│   └── constants.py      # Constants and configurations
├── inference.py          # Main inference runner
└── README.md
```

## Key Components

### Dynamic Scene Graph (Simplified)

The agent maintains a scene graph representation:
- **Rooms**: Discovered rooms with classified types
- **Objects**: Objects observed in each room with distances
- **Frontiers**: Unexplored areas that can be reached

### LLM Prompting

System prompt provides:
- Task description
- Available actions with descriptions
- Response format instructions
- Navigation rules

User prompt provides:
- Current location (classified room name)
- Visible objects with distances
- Discovered rooms and their contents
- Frontier information
- Action history with success/failure
- Decision guidance

### Room Classification

When new rooms are discovered:
1. Objects observed in the room are collected
2. LLM classifies the room based on object contents
3. Unique display names are assigned (e.g., "bedroom", "bedroom 2")

## Metrics

Standard ObjectNav metrics:
- **Success Rate (SR)**: Percentage of episodes where target was found
- **SPL**: Success weighted by Path Length
- **Episode Time**: Average time per episode
- **Steps**: Average action steps per episode
