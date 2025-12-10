# Baseline Methods for Habitat

This module implements baseline action selection strategies for object search in Habitat environments. These serve as comparison baselines for the LLM-based navigation approach in SmallPlan.

## Overview

### 1. Random Search Baseline (`random_inference.py`)

The random search algorithm:
1. At each step, collects all available actions: frontier points, closed objects, visible objects, and rooms
2. **Randomly selects** one of these options
3. Executes the selected action (explore, open, goto)
4. Continues until the target object is found or the step limit is reached

### 2. Greedy Frontier-Based Exploration (`greedy_inference.py`)

The greedy search algorithm:
1. At each step, collects all available actions: frontier points, closed objects, visible objects, and rooms
2. Computes the path cost (distance) to each target
3. **Selects the closest target** based on Euclidean distance
4. Executes the selected action (explore, open, goto)
5. Continues until the target object is found or the step limit is reached

The greedy baseline provides a stronger comparison than random search, as it uses spatial reasoning to prioritize nearby exploration targets.

## Usage

### Random Search

#### Basic Run (Headless)
```bash
cd /media/khointn/SmallPlan
python -m src.random_search_habitat.random_inference
```

#### With Video Recording
```bash
python -m src.random_search_habitat.random_inference --save-video
```

#### With GUI Visualization (requires display)
```bash
python -m src.random_search_habitat.random_inference --gui
```

#### Custom Seed (for reproducibility testing)
```bash
python -m src.random_search_habitat.random_inference --seed 123
```

#### Full Options
```bash
python -m src.random_search_habitat.random_inference \
    --mode headless \
    --save-video \
    --video-dir ./videos/random_search \
    --seed 42 \
    --verbose
```

### Greedy Frontier-Based Exploration

#### Basic Run (Headless)
```bash
cd /media/khointn/SmallPlan
python -m src.random_search_habitat.greedy_inference
```

#### With Video Recording
```bash
python -m src.random_search_habitat.greedy_inference --save-video
```

#### With GUI Visualization (requires display)
```bash
python -m src.random_search_habitat.greedy_inference --gui
```

#### Custom Seed (for reproducibility testing)
```bash
python -m src.random_search_habitat.greedy_inference --seed 123
```

#### Full Options
```bash
python -m src.random_search_habitat.greedy_inference \
    --mode headless \
    --save-video \
    --video-dir ./videos/greedy_search \
    --seed 42 \
    --verbose
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--gui` | Enable GUI mode for visualization | False |
| `--mode` | Rendering mode: 'headless' or 'gui' | 'headless' |
| `--save-video` | Save RGB frames as video files | False |
| `--video-dir` | Directory to save videos | './videos/random_search' |
| `--seed` | Override random seed from config | Config value (42) |
| `--verbose`, `-v` | Enable verbose DEBUG output | False |

## Configuration

| Baseline | Config File |
|----------|-------------|
| Random Search | `configs/random_search_habitat.yaml` |
| Greedy Search | `configs/greedy_search_habitat.yaml` |
| LLM Inference | `configs/moma_llm_habitat.yaml` |

All configs mirror the same settings for fair comparison:
- Same `seed: 42` (or override via `--seed`)
- Same `max_high_level_steps: 50`
- Same scene configuration and dataset
- Same robot configuration

## Comparison with LLM Inference

To compare all methods on the same scenarios:

1. Run random search:
```bash
python -m src.random_search_habitat.random_inference --save-video --seed 42
```

2. Run greedy search:
```bash
python -m src.random_search_habitat.greedy_inference --save-video --seed 42
```

3. Run LLM inference (same scenes, same seed):
```bash
python -m src.train_from_simulation_habitat.habitat_inference --save-video
```

All runs will log metrics to WandB for comparison, including:
- Success rate
- Steps to completion (SPL)
- Navigation efficiency curves
- Episode videos (if `--save-video` enabled)

## Output

Videos are saved to timestamped subdirectories:
```
./videos/random_search/
└── 20231215_143022/
    ├── success/     # Successful episodes
    ├── failed/      # Failed episodes
    └── error/       # Interrupted episodes
```

## Key Differences Between Methods

| Aspect | Random Search | Greedy Search | LLM Inference |
|--------|---------------|---------------|---------------|
| Action Selection | Random from available options | Closest target (by distance) | LLM reasoning based on context |
| Token Usage | 0 | 0 | Varies per episode |
| Determinism | Deterministic with same seed | Deterministic with same seed | May vary due to API randomness |
| Speed | Fastest (no computation) | Fast (distance computation) | Slowest (LLM API latency) |
| Spatial Reasoning | None | Distance-based | Semantic understanding |

## Files in This Module

| File | Description |
|------|-------------|
| `random_inference.py` | Random search inference script |
| `greedy_inference.py` | Greedy frontier-based exploration script |
| `fixed_baseline.py` | Fixed random baseline with comprehensive action types |
| `fixed_greedy_baseline.py` | Fixed greedy baseline with comprehensive action types |
| `__init__.py` | Module initialization |
| `README.md` | This documentation |

