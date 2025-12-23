# Habitat Object Navigation Module

A clean, modular implementation for LLM-based object navigation using Habitat-Sim.
Follows [Nav-R1](https://github.com/AIGeeksGroup/Nav-R1) style action space and simulator conventions.

## Overview

This module provides:
- **ObjectNav Environment**: Episode-based navigation to find target objects
- **Nav-R1 Compatible Actions**: Discrete action space (MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP)
- **LLM Agent**: High-level navigation using language models
- **Baseline Agents**: Random agent for comparison

## Directory Structure

```
habitat_nav/
├── core/               # Core environment and simulator
│   ├── simulator.py    # Habitat-Sim wrapper
│   ├── environment.py  # ObjectNav environment
│   └── observations.py # Observation processing
├── agents/             # Navigation agents
│   ├── base.py         # Base agent class
│   ├── llm_agent.py    # LLM-based agent
│   └── random_agent.py # Random baseline
├── tasks/              # Task definitions
│   └── object_nav.py   # ObjectNav task
├── utils/              # Utilities
│   ├── actions.py      # Action definitions
│   └── constants.py    # Constants and scene lists
└── inference.py        # Main inference runner
```

## Quick Start

### Prerequisites

1. Install Habitat-Sim and Habitat-Lab:
```bash
# Follow official instructions for your OS/CUDA version
# https://github.com/facebookresearch/habitat-sim
```

2. Set up environment variables:
```bash
export HABITAT_DATA_PATH=/path/to/habitat/data
```

3. Download HM3D dataset (or other supported datasets)

### Running Inference

```bash
# Run with custom LLM API
python -m src.habitat_nav.inference --scene 00800-TEEsavR23oF --episodes 5

# Run with OpenAI GPT-4o
python -m src.habitat_nav.inference --agent llm-openai --model gpt-4o

# Run with OpenAI GPT-3.5-turbo (cheaper)
python -m src.habitat_nav.inference --agent llm-openai --model gpt-3.5-turbo

# Run with custom OpenAI API key
python -m src.habitat_nav.inference --agent llm-openai --openai-key sk-...

# Run random baseline
python -m src.habitat_nav.inference --agent random --episodes 10

# Use custom config
python -m src.habitat_nav.inference --config configs/habitat_nav.yaml

# Verbose output without video saving
python -m src.habitat_nav.inference --verbose --no-video
```

### LLM API Options

The LLM agent supports two API backends:

**OpenAI API** (recommended for best performance):
```bash
# Set API key via environment variable
export OPENAI_API_KEY=sk-your-key-here

# Or pass directly
python -m src.habitat_nav.inference --agent llm-openai --openai-key sk-...
```

Supported OpenAI models:
- `gpt-4o` (recommended - best quality)
- `gpt-4-turbo`
- `gpt-3.5-turbo` (faster, cheaper)

**Custom API** (for local LLM servers):
```bash
python -m src.habitat_nav.inference --agent llm-custom --api-url http://localhost:8000/generate
```

## Action Space

Following Nav-R1 conventions:

| Action | ID | Description |
|--------|-----|-------------|
| MOVE_FORWARD | 0 | Move forward 0.25 meters |
| TURN_LEFT | 1 | Turn left 30 degrees |
| TURN_RIGHT | 2 | Turn right 30 degrees |
| STOP | 3 | Terminate episode |

## High-Level Actions

For LLM agents, we provide semantic high-level actions:

| Action | Example | Description |
|--------|---------|-------------|
| `goto(target)` | `goto(kitchen)` | Navigate to object/room |
| `explore(room)` | `explore(bedroom)` | Explore unexplored areas |
| `open(object)` | `open(door)` | Open door or container |
| `stop()` | `stop()` | Terminate task |

## API Usage

```python
from src.habitat_nav.core import ObjectNavEnv
from src.habitat_nav.agents import LLMAgent, create_openai_agent, create_custom_api_agent

# Create environment
env = ObjectNavEnv(scene_id="00800-TEEsavR23oF")

# Option 1: Create agent with OpenAI API
agent = create_openai_agent(model="gpt-4o")

# Option 2: Create agent with custom API
agent = create_custom_api_agent(api_url="http://localhost:8000/generate")

# Option 3: Manual configuration
agent = LLMAgent(
    api_type="openai",  # or "custom"
    model_name="gpt-4o",
    temperature=0.7
)

# Run episode
obs, info = env.reset()
agent.reset(target_category=info["target"])

while not info["done"]:
    action = agent.act(obs, env.get_task_description(), info)
    obs, reward, info = env.step(action)

print(f"Success: {info['success']}, SPL: {env.compute_spl()}")

# Get token usage (for OpenAI)
metrics = agent.get_metrics()
print(f"Tokens used: {metrics['total_tokens']}")

env.close()
```

## Configuration

See `configs/habitat_nav.yaml` for all options:

```yaml
# Key settings
dataset: hm3d
agent_type: llm           # llm, llm-openai, llm-custom, random
llm_api_type: openai      # openai or custom
llm_model: gpt-4o         # Model name
llm_temperature: 0.7      # Sampling temperature

# Nav-R1 compatible settings
forward_step_size: 0.25   # Nav-R1 default (meters)
turn_angle: 30.0          # Nav-R1 default (degrees)
success_distance: 1.5     # meters
max_episode_steps: 500
```

## Metrics

- **Success Rate**: Percentage of episodes where target was found
- **SPL**: Success weighted by Path Length (efficiency metric)
- **Steps**: Number of actions taken

## References

- [Nav-R1](https://github.com/AIGeeksGroup/Nav-R1) - Navigation with R1 reasoning
- [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) - Embodied AI platform
- [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) - High-performance simulator

