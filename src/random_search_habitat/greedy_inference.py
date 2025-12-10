# Greedy Frontier-Based Exploration Inference Script for SmallPlan (Habitat)
# Implements greedy action selection baseline (closest target) for comparison with LLM and random

# Set environment variables to suppress Habitat-Sim C++ warnings before imports
import os
os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'

import argparse
import atexit
import signal
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Any, Optional
import yaml

import numpy as np
from sklearn.metrics import auc
import wandb

import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress Habitat-Sim warnings about missing scene instance files
habitat_sim_logger = logging.getLogger("habitat_sim")
habitat_sim_logger.setLevel(logging.ERROR)

# Also suppress habitat logging warnings
habitat_logger = logging.getLogger("habitat")
habitat_logger.setLevel(logging.ERROR)

# Global state for cleanup on interrupt
_current_env = None
_current_video_path = None
_video_saved = False
_run_video_dir = None  # Run-specific video directory with datetime
_inference_logger = None  # Real-time inference statistics logger


def _save_video_on_exit():
    """Save video when script exits (cleanup handler) - saves to 'error' subdirectory."""
    global _current_env, _current_video_path, _video_saved, _run_video_dir, _inference_logger
    
    if _video_saved:
        return
        
    if _current_env is not None and _current_video_path is not None:
        try:
            # On interrupt/error, save to "error" subdirectory within the run directory
            video_filename = os.path.basename(_current_video_path)
            if _run_video_dir:
                error_dir = os.path.join(_run_video_dir, "error")
            else:
                # Fallback to parent directory of current video path
                video_dir = os.path.dirname(_current_video_path)
                error_dir = os.path.join(video_dir, "error")
            os.makedirs(error_dir, exist_ok=True)
            error_video_path = os.path.join(error_dir, video_filename)
            
            logger.info(f"Saving video on exit (interrupted/error) to {error_video_path}")
            if hasattr(_current_env, 'env') and hasattr(_current_env.env, 'save_video'):
                _current_env.env.save_video(error_video_path, fps=10)
            elif hasattr(_current_env, 'save_video'):
                _current_env.save_video(error_video_path, fps=10)
            _video_saved = True
            logger.info("Video saved successfully on exit to error directory")
            
            # Record error in inference logger
            if _inference_logger is not None:
                _inference_logger.record_error("Episode interrupted/crashed")
        except Exception as e:
            logger.error(f"Failed to save video on exit: {e}")


def _signal_handler(signum, frame):
    """Handle interrupt signals (Ctrl+C)."""
    logger.info(f"\nReceived signal {signum}. Saving video before exit...")
    _save_video_on_exit()
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
atexit.register(_save_video_on_exit)

# Import Habitat-compatible modules from the existing habitat training package
from src.train_from_simulation_habitat.packages.moma_llm.env.habitat_env import (
    OurHabitatEnv,
    create_habitat_env
)
from src.train_from_simulation_habitat.packages.moma_llm.env.habitat_llm_env import (
    HabitatHighLevelEnv,
    HabitatLLMEnv
)
from src.train_from_simulation_habitat.packages.moma_llm.env.habitat_baselines import (
    HabitatGreedyBaseline,
    HabitatRandomBaseline
)
# Use fixed greedy baseline with comprehensive action types
from src.random_search_habitat.fixed_greedy_baseline import FixedHabitatGreedyBaseline
from src.train_from_simulation_habitat.packages.moma_llm.tasks.habitat_object_search_task import HabitatObjectSearchTask
from src.train_from_simulation_habitat.packages.moma_llm.utils.habitat_constants import (
    TRAINING_SCENES,
    TEST_SCENES,
    get_scenes_for_dataset,
    NODETYPE,
    POSSIBLE_ROOMS
)
from src.train_from_simulation_habitat.packages.moma_llm.llm.habitat_llm import LLM_hugging

# Import shared utility functions from habitat_train
from src.train_from_simulation_habitat.habitat_train import (
    load_config,
    calc_area_under_curve,
    plot_efficiency_curves,
    calculate_metric_means,
    log_summary_table
)

# Import inference logger for real-time statistics
from src.train_from_simulation_habitat.inference_logger import (
    InferenceLogger,
    create_inference_logger
)


def create_greedy_env(cfg: Dict,
                      config_file: str,
                      scene_id: str,
                      control_freq: float,
                      seed: int,
                      mode: str = "headless") -> FixedHabitatGreedyBaseline:
    """
    Create greedy baseline environment for Habitat.
    
    Args:
        cfg: Configuration dictionary
        config_file: Path to config file
        scene_id: Scene identifier
        control_freq: Control frequency
        seed: Random seed
        mode: Rendering mode ("headless" or "gui")
        
    Returns:
        FixedHabitatGreedyBaseline environment
    """
    # Create a minimal LLM instance (needed for room classification and object name formatting)
    # even though we don't use it for action decisions
    llm = LLM_hugging(
        debug=True,
        room_classification_model="gpt-4o",
        open_set_rooms=cfg.get("open_set_room_categories", True),
        slm_api_url=""  # Not used for greedy baseline
    )
    
    low_level_env = create_habitat_env(
        config_file=config_file,
        scene_id=scene_id,
        control_freq=control_freq,
        seed=seed,
        mode=mode
    )
    
    # Attach task
    low_level_env.task = HabitatObjectSearchTask(low_level_env)
    
    # Use FixedHabitatGreedyBaseline for greedy action selection
    high_level_env = FixedHabitatGreedyBaseline(env=low_level_env, llm=llm, seed=seed)
    return high_level_env


def evaluate_scene_greedy(config_file: str, 
                          cfg: Dict, 
                          scene_id: str, 
                          tot_ep: int, 
                          mode: str = "headless",
                          save_video: bool = False,
                          video_dir: str = None) -> tuple:
    """
    Evaluate greedy baseline on a single scene.
    
    Args:
        config_file: Path to config file
        cfg: Configuration dictionary
        scene_id: Scene identifier
        tot_ep: Total episode count
        mode: Rendering mode ("headless" or "gui")
        save_video: Whether to save RGB frames as video
        video_dir: Run-specific directory to save videos (with datetime subfolder)
        
    Returns:
        Tuple of (episode_infos, tot_ep)
    """
    global _current_env, _current_video_path, _video_saved, _run_video_dir, _inference_logger
    
    # Use the global run video directory if not provided
    if video_dir is None:
        video_dir = _run_video_dir
    
    logger.info(f"evaluate_scene_greedy called for {scene_id}")
    
    if save_video and video_dir:
        os.makedirs(video_dir, exist_ok=True)
    
    episode_infos = []
    
    high_level_env = create_greedy_env(
        cfg=cfg,
        config_file=config_file,
        scene_id=scene_id,
        control_freq=cfg.get("control_freq", 10.0),
        seed=cfg.get("seed", 42),
        mode=mode
    )
    
    for i in range(cfg.get("num_episodes_per_scene", 2)):
        done = False
        obs = high_level_env.reset(config_file=config_file, scene_id=scene_id, episode_num=i)
        
        # Set up global state for interrupt handling
        if save_video:
            _current_env = high_level_env
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            _current_video_path = os.path.join(video_dir, f"{timestamp}_{scene_id}_episode_{i}_tot_{tot_ep}.mp4")
            _video_saved = False
        
        print("########################################")
        print(f"{scene_id} - Starting episode {i + 1} (GREEDY) in scene {scene_id}, "
              f"{tot_ep + 1} overall. Task: {high_level_env.env.task.task_description}")
        print("########################################")
        
        # Log episode start
        if _inference_logger is not None:
            _inference_logger.start_episode(
                scene_id=scene_id,
                episode_num=i,
                task_description=high_level_env.env.task.task_description
            )
        
        step_count = 0
        max_steps = cfg.get("max_high_level_steps", 50)
        
        while not done and step_count < max_steps:
            high_level_env.visualize(obs)
            
            # Greedy baseline uses take_action instead of take_action_inference
            # It greedily selects the closest target from available frontier points and closed objects
            done, task_success, episode_info = high_level_env.take_action(
                obs=obs,
                task_description=high_level_env.env.task.task_description
            )
            
            wandb.log({"bev_maps": high_level_env.env.f})
            obs = high_level_env.get_state(compute_scene_graph=True)
            pprint(episode_info)
            
            step_count += 1
            
            # Check for step limit
            if step_count >= max_steps and not done:
                episode_info["failure_reason"] = "max_high_level_steps timeout"
                episode_info["task_success"] = False
                task_success = False  # Explicitly set to False on timeout
                done = True
            
        high_level_env.visualize(obs)
        
        # Save video if requested - to "success" or "failed" subdirectory based on outcome
        if save_video and hasattr(high_level_env.env, 'save_video'):
            # Determine subdirectory based on task success
            outcome_subdir = "success" if task_success else "failed"
            outcome_dir = os.path.join(video_dir, outcome_subdir)
            os.makedirs(outcome_dir, exist_ok=True)
            
            # Build video path with outcome subdirectory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"{timestamp}_{scene_id}_episode_{i}_tot_{tot_ep}.mp4"
            video_path = os.path.join(outcome_dir, video_filename)
            
            # Pass task description for video overlay
            task_desc = high_level_env.env.task.task_description if high_level_env.env.task else None
            high_level_env.env.save_video(video_path, fps=10, task_text=task_desc)
            _video_saved = True  # Mark as saved so signal handler doesn't save again
            print(f"Saved episode video to {video_path} ({outcome_subdir})")
        
        if "failure_reason" in episode_info:
            high_level_env.env.f.suptitle(
                f"{high_level_env.env.f._suptitle.get_text()}, {episode_info['failure_reason']}"
            )
            
        episode_info["bev_maps"] = high_level_env.env.f
        episode_info["num_low_level_steps_with_open_cost"] = (
            episode_info.get("num_low_level_steps", 0) +
            high_level_env.env.config.get("magic_open_cost", 30) *
            episode_info.get("magic_open_actions", 0)
        )
        
        if episode_info.get("num_low_level_steps_gtDone", None) is not None:
            episode_info["num_low_level_steps_with_open_cost_gtDone"] = (
                episode_info["num_low_level_steps_gtDone"] +
                high_level_env.env.config.get("magic_open_cost", 30) *
                episode_info.get("magic_open_actions_gtDone", 0)
            )
            episode_info["task_success_gtDone"] = True
        else:
            episode_info["num_low_level_steps_with_open_cost_gtDone"] = \
                episode_info["num_low_level_steps_with_open_cost"]
            episode_info["task_success_gtDone"] = task_success
            
        episode_info["episode_step"] = tot_ep
        episode_info["num_high_level_steps"] = step_count
        episode_info["task_success"] = task_success  # Ensure task_success is always set
        
        # Handle None values for shortest_dist and dist_travelled
        shortest_dist = episode_info.get("shortest_dist") or 1
        dist_travelled = episode_info.get("dist_travelled") or 1
        episode_info["spl"] = episode_info.get("task_success", False) * (
            shortest_dist / max(shortest_dist, dist_travelled)
        )
        
        # Add greedy baseline specific metrics (no LLM tokens used)
        episode_info["episode_input_tokens"] = 0
        episode_info["episode_output_tokens"] = 0
        episode_info["episode_total_tokens"] = 0
        episode_info["episode_llm_queries"] = 0
        episode_info["avg_tokens_per_query"] = 0
        
        pprint(episode_info)
        wandb.log({k: float(v) if isinstance(v, bool) else v 
                   for k, v in episode_info.items()})
        
        episode_infos.append(episode_info)
        successes = [e.get("task_success", False) for e in episode_infos]
        tot_ep += 1
        
        print(f"Task success: {task_success} (wandb_step: {wandb.run.step}). "
              f"Current successes: {sum(successes)}/{len(successes)}")
        
        # Log episode end
        if _inference_logger is not None:
            _inference_logger.end_episode(
                task_success=task_success,
                episode_info=episode_info,
                failure_reason=episode_info.get("failure_reason")
            )
    
    scene_logs = calculate_metric_means({scene_id: episode_infos})
    wandb.log({f"{scene_id}_{k}": v for k, v in scene_logs[scene_id].items()})
    
    # Debug: Print episode_info keys
    if episode_infos:
        print(f"Episode info keys for {scene_id}: {list(episode_infos[0].keys())}")
    
    high_level_env.close()
    return episode_infos, tot_ep


def setup_cfgs():
    """Setup configurations for greedy search inference."""
    # Use dedicated greedy search config (ensures same settings as LLM/random inference)
    config_file = "./configs/greedy_search_habitat.yaml"
    cfg = load_config(config_file)
    wandb_cfg = load_config("./configs/wandb.yaml")
    
    # Ensure agent is set to greedy
    cfg["agent"] = "greedy"
    
    return config_file, cfg, wandb_cfg


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Greedy Frontier-Based Exploration for SmallPlan (Habitat)")
    parser.add_argument(
        "--gui", 
        action="store_true",
        help="Enable GUI mode for visualization (requires display)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="headless",
        choices=["headless", "gui"],
        help="Rendering mode: 'headless' (default) or 'gui'"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save RGB frames as video files for each episode"
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="./videos/greedy_search",
        help="Directory to save videos (default: ./videos/greedy_search)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose DEBUG output for navigation and other components"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed from config (for reproducibility tests)"
    )
    return parser.parse_args()


def main():
    """Main greedy search inference function."""
    global _run_video_dir, _inference_logger
    
    args = parse_args()
    
    # Determine rendering mode
    mode = "gui" if args.gui else args.mode
    logger.info(f"Running greedy search in {mode} mode")
    
    # Set verbose mode
    verbose = args.verbose
    if verbose:
        logger.info("Verbose mode enabled - DEBUG messages will be printed")
    
    # Create run-specific video directory with datetime
    if args.save_video:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _run_video_dir = os.path.join(args.video_dir, f"{run_timestamp}")
        os.makedirs(_run_video_dir, exist_ok=True)
        # Pre-create subdirectories
        os.makedirs(os.path.join(_run_video_dir, "success"), exist_ok=True)
        os.makedirs(os.path.join(_run_video_dir, "failed"), exist_ok=True)
        os.makedirs(os.path.join(_run_video_dir, "error"), exist_ok=True)
        logger.info(f"Videos will be saved to: {_run_video_dir}")
        
        # Create inference logger for real-time statistics
        _inference_logger = create_inference_logger(
            video_dir=_run_video_dir,
            run_name=f"greedy-search-{run_timestamp}",
            agent_type="greedy",
            enabled=True
        )
    
    np.set_printoptions(precision=3, suppress=True)
    
    config_file, cfg, wandb_cfg = setup_cfgs()
    
    # Override seed if provided via command line
    if args.seed is not None:
        cfg["seed"] = args.seed
        logger.info(f"Using command line seed: {args.seed}")
    
    # Add verbose flag to config so it's accessible throughout the codebase
    cfg["verbose"] = verbose
    
    logger.info("=" * 50)
    logger.info("GREEDY FRONTIER-BASED EXPLORATION BASELINE")
    logger.info("Action selection: Greedy (closest target from frontiers and closed objects)")
    logger.info(f"Seed: {cfg.get('seed', 42)}")
    logger.info("=" * 50)
    
    run_name = f"habitat-greedy-search-seed{cfg.get('seed', 42)}"
    
    if cfg.get("seed", 0) > 0:
        np.random.seed(cfg["seed"])
    
    # Get scene IDs based on dataset and split
    # Default to hm3d dataset
    dataset = cfg.get("habitat", {}).get("dataset", "hm3d")
    if cfg.get("datasplit") == "train":
        scene_ids = get_scenes_for_dataset(dataset, "train")
    elif cfg.get("datasplit") == "test":
        scene_ids = get_scenes_for_dataset(dataset, "test")
    else:
        raise ValueError(f"Unknown datasplit {cfg.get('datasplit')}")
    
    cfg.update({"scene_ids": scene_ids, "agent": "greedy"})
    
    wandb.init(
        project=wandb_cfg.get("project_inference", "smallplan-habitat-inference"),
        entity=wandb_cfg.get("entity"),
        config=cfg,
        mode=wandb_cfg.get("mode", "online") if cfg.get("wandb", True) else "disabled",
        name=run_name,
        tags=["greedy-baseline", "habitat"]
    )
    
    # Copy config to wandb directory
    new_config_file = Path(wandb.run.dir) / Path(config_file).name
    shutil.copy(config_file, new_config_file)
    config_file = str(new_config_file)
    
    episode_infos = defaultdict(list)
    tot_ep = 0
    
    if isinstance(scene_ids, str):
        scene_ids = [scene_ids]
        
    for scene_id in scene_ids:
        infos, tot_ep = evaluate_scene_greedy(
            config_file=config_file,
            cfg=cfg,
            scene_id=scene_id,
            tot_ep=tot_ep,
            mode=mode,
            save_video=args.save_video,
            video_dir=_run_video_dir  # Use run-specific directory with datetime
        )
        episode_infos[scene_id] = infos
        
    log_summary_table(episode_infos=episode_infos)
    plot_efficiency_curves(episode_infos=episode_infos,
                          max_hl_steps=cfg.get("max_high_level_steps", 50))
    
    # Log final summary
    all_successes = []
    for scene_id, infos in episode_infos.items():
        all_successes.extend([e.get("task_success", False) for e in infos])
    
    success_rate = sum(all_successes) / len(all_successes) if all_successes else 0
    logger.info("=" * 50)
    logger.info("GREEDY SEARCH RESULTS")
    logger.info(f"Total episodes: {len(all_successes)}")
    logger.info(f"Successful episodes: {sum(all_successes)}")
    logger.info(f"Success rate: {success_rate:.2%}")
    logger.info("=" * 50)
    
    wandb.log({
        "final_success_rate": success_rate,
        "total_episodes": len(all_successes),
        "successful_episodes": sum(all_successes)
    })
    
    # Finalize inference logger
    if _inference_logger is not None:
        _inference_logger.finalize()
        _inference_logger.print_current_stats()
    
    wandb.run.finish()
    logger.info("Greedy search inference completed successfully.")


if __name__ == "__main__":
    main()


