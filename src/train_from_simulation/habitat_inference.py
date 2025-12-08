# Habitat Inference Script for SmallPlan
# Replaces iGibson-based inference.py for Habitat-Lab/Habitat-Sim

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


def _save_video_on_exit():
    """Save video when script exits (cleanup handler) - saves to 'error' subdirectory."""
    global _current_env, _current_video_path, _video_saved
    
    if _video_saved:
        return
        
    if _current_env is not None and _current_video_path is not None:
        try:
            # On interrupt/error, save to "error" subdirectory
            video_dir = os.path.dirname(_current_video_path)
            video_filename = os.path.basename(_current_video_path)
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

# Import Habitat-compatible modules
from src.train_from_simulation.packages.moma_llm.env.habitat_env import (
    OurHabitatEnv,
    create_habitat_env
)
from src.train_from_simulation.packages.moma_llm.env.habitat_llm_env import (
    HabitatHighLevelEnv,
    HabitatLLMEnv
)
from src.train_from_simulation.packages.moma_llm.env.habitat_baselines import (
    HabitatGreedyBaseline,
    HabitatRandomBaseline
)
from src.train_from_simulation.packages.moma_llm.tasks.habitat_object_search_task import HabitatObjectSearchTask
from src.train_from_simulation.packages.moma_llm.utils.habitat_constants import (
    TRAINING_SCENES,
    TEST_SCENES,
    get_scenes_for_dataset,
    NODETYPE,
    POSSIBLE_ROOMS
)

# Import from habitat_train for shared functionality
from src.train_from_simulation.habitat_train import (
    load_config,
    create_env,
    calc_area_under_curve,
    plot_efficiency_curves,
    calculate_metric_means,
    log_summary_table
)


def log_scene_token_metrics(scene_id: str, episode_infos: List[Dict]):
    """Log token usage metrics for a specific scene to WandB."""
    
    input_tokens = [e.get('episode_input_tokens', 0) for e in episode_infos 
                    if 'episode_input_tokens' in e]
    output_tokens = [e.get('episode_output_tokens', 0) for e in episode_infos 
                     if 'episode_output_tokens' in e]
    total_tokens = [e.get('episode_total_tokens', 0) for e in episode_infos 
                    if 'episode_total_tokens' in e]
    queries = [e.get('episode_llm_queries', 0) for e in episode_infos 
               if 'episode_llm_queries' in e]
    tokens_per_query = [e.get('avg_tokens_per_query', 0) for e in episode_infos 
                        if 'avg_tokens_per_query' in e]
    
    if not total_tokens:
        print(f"No token metrics found for scene {scene_id}")
        return
    
    scene_token_summary = {
        f'{scene_id}_avg_input_tokens_per_episode': np.mean(input_tokens) if input_tokens else 0,
        f'{scene_id}_avg_output_tokens_per_episode': np.mean(output_tokens) if output_tokens else 0,
        f'{scene_id}_avg_total_tokens_per_episode': np.mean(total_tokens) if total_tokens else 0,
        f'{scene_id}_avg_llm_queries_per_episode': np.mean(queries) if queries else 0,
        f'{scene_id}_avg_tokens_per_query': np.mean(tokens_per_query) if tokens_per_query else 0,
        f'{scene_id}_total_input_tokens': sum(input_tokens) if input_tokens else 0,
        f'{scene_id}_total_output_tokens': sum(output_tokens) if output_tokens else 0,
        f'{scene_id}_total_tokens': sum(total_tokens) if total_tokens else 0,
        f'{scene_id}_total_queries': sum(queries) if queries else 0,
    }
    
    wandb.log(scene_token_summary)
    
    print(f"=== Token Usage Summary for {scene_id} ===")
    for key, value in scene_token_summary.items():
        clean_key = key.replace(f'{scene_id}_', '')
        print(f"{clean_key}: {value:.2f}" if isinstance(value, float) else f"{clean_key}: {value}")
    print("=" * (30 + len(scene_id)))


def log_token_metrics(episode_infos: Dict[str, List[Dict]]):
    """Log overall average token usage metrics to WandB."""
    
    all_episodes = []
    for scene_id in episode_infos:
        all_episodes.extend(episode_infos[scene_id])
    
    input_tokens = [e.get('episode_input_tokens', 0) for e in all_episodes 
                    if 'episode_input_tokens' in e]
    output_tokens = [e.get('episode_output_tokens', 0) for e in all_episodes 
                     if 'episode_output_tokens' in e]
    total_tokens = [e.get('episode_total_tokens', 0) for e in all_episodes 
                    if 'episode_total_tokens' in e]
    queries = [e.get('episode_llm_queries', 0) for e in all_episodes 
               if 'episode_llm_queries' in e]
    tokens_per_query = [e.get('avg_tokens_per_query', 0) for e in all_episodes 
                        if 'avg_tokens_per_query' in e]
    
    if not total_tokens:
        print("No token metrics found in episodes")
        return
    
    overall_token_summary = {
        'overall_avg_input_tokens_per_episode': np.mean(input_tokens) if input_tokens else 0,
        'overall_avg_output_tokens_per_episode': np.mean(output_tokens) if output_tokens else 0,
        'overall_avg_total_tokens_per_episode': np.mean(total_tokens) if total_tokens else 0,
        'overall_avg_llm_queries_per_episode': np.mean(queries) if queries else 0,
        'overall_avg_tokens_per_query': np.mean(tokens_per_query) if tokens_per_query else 0,
        'overall_total_input_tokens': sum(input_tokens) if input_tokens else 0,
        'overall_total_output_tokens': sum(output_tokens) if output_tokens else 0,
        'overall_total_tokens': sum(total_tokens) if total_tokens else 0,
        'overall_total_queries': sum(queries) if queries else 0,
    }
    
    wandb.log(overall_token_summary)
    
    print("=== Overall Token Usage Summary ===")
    for key, value in overall_token_summary.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    print("===================================")


def evaluate_scene(config_file: str, 
                   cfg: Dict, 
                   scene_id: str, 
                   tot_ep: int, 
                   slm_api_url: str,
                   mode: str = "headless",
                   save_video: bool = False,
                   video_dir: str = "./videos") -> tuple:
    """
    Evaluate on a single scene.
    
    Args:
        config_file: Path to config file
        cfg: Configuration dictionary
        scene_id: Scene identifier
        tot_ep: Total episode count
        slm_api_url: SLM API URL
        mode: Rendering mode ("headless" or "gui")
        save_video: Whether to save RGB frames as video
        video_dir: Directory to save videos
        
    Returns:
        Tuple of (episode_infos, tot_ep)
    """
    global _current_env, _current_video_path, _video_saved
    
    # DEBUG: Log config value at the start of evaluate_scene
    logger.info(f"evaluate_scene called for {scene_id}")
    logger.info(f"  cfg['open_set_room_categories'] = {cfg.get('open_set_room_categories', 'KEY NOT FOUND')}")
    
    if save_video:
        os.makedirs(video_dir, exist_ok=True)
    
    episode_infos = []
    
    high_level_env = create_env(
        cfg=cfg,
        agent=cfg.get("agent", "moma_llm"),
        config_file=config_file,
        scene_id=scene_id,
        control_freq=cfg.get("control_freq", 10.0),
        cheap=cfg.get("cheap", False),
        seed=cfg.get("seed", 42),
        slm_api_url=slm_api_url,
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
        print(f"{scene_id} - Starting episode {i + 1} in scene {scene_id}, "
              f"{tot_ep + 1} overall. Task: {high_level_env.env.task.task_description}")
        print("########################################")
        
        while not done:
            high_level_env.visualize(obs)
            done, task_success, episode_info = high_level_env.take_action_inference(
                obs=obs,
                task_description=high_level_env.env.task.task_description
            )
            
            wandb.log({"bev_maps": high_level_env.env.f})
            obs = high_level_env.get_state(compute_scene_graph=True)
            pprint(episode_info)
            
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
            
            high_level_env.env.save_video(video_path, fps=10)
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
        # Handle None values for shortest_dist and dist_travelled
        shortest_dist = episode_info.get("shortest_dist") or 1
        dist_travelled = episode_info.get("dist_travelled") or 1
        episode_info["spl"] = episode_info.get("task_success", False) * (
            shortest_dist / max(shortest_dist, dist_travelled)
        )
        
        pprint(episode_info)
        wandb.log({k: float(v) if isinstance(v, bool) else v 
                   for k, v in episode_info.items()})
        
        episode_infos.append(episode_info)
        successes = [e.get("task_success", False) for e in episode_infos]
        tot_ep += 1
        
        print(f"Task success: {task_success} (wandb_step: {wandb.run.step}). "
              f"Current successes: {sum(successes)}/{len(successes)}")
    
    scene_logs = calculate_metric_means({scene_id: episode_infos})
    wandb.log({f"{scene_id}_{k}": v for k, v in scene_logs[scene_id].items()})
    
    # Debug: Print episode_info keys
    if episode_infos:
        print(f"Episode info keys for {scene_id}: {list(episode_infos[0].keys())}")
    
    # Log token metrics for this scene
    log_scene_token_metrics(scene_id, episode_infos)
    
    high_level_env.close()
    return episode_infos, tot_ep


def setup_cfgs():
    """Setup configurations for inference."""
    config_file = "./configs/moma_llm_habitat.yaml"
    cfg = load_config(config_file)
    wandb_cfg = load_config("./configs/wandb.yaml")
    slm_training_cfg = load_config("./configs/slm_training.yaml")
    
    return config_file, cfg, wandb_cfg, slm_training_cfg


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Habitat Inference for SmallPlan")
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
        default="./videos",
        help="Directory to save videos (default: ./videos)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose DEBUG output for navigation and other components"
    )
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()
    
    # Determine rendering mode
    mode = "gui" if args.gui else args.mode
    logger.info(f"Running in {mode} mode")
    
    # Set verbose mode
    verbose = args.verbose
    if verbose:
        logger.info("Verbose mode enabled - DEBUG messages will be printed")
    
    np.set_printoptions(precision=3, suppress=True)
    
    config_file, cfg, wandb_cfg, slm_training_cfg = setup_cfgs()
    
    # DEBUG: Print the actual config value to verify it's being loaded correctly
    logger.info(f"Config loaded from {config_file}")
    logger.info(f"open_set_room_categories from config: {cfg.get('open_set_room_categories', 'KEY NOT FOUND')}")
    
    # Add verbose flag to config so it's accessible throughout the codebase
    cfg["verbose"] = verbose
    run_name = f"habitat-{slm_training_cfg.get('slm_api_model', 'default')}"
    slm_api_url = f"http://{slm_training_cfg['slm_api_host']}:{slm_training_cfg['slm_api_port']}"
    
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
    
    cfg.update({"scene_ids": scene_ids, "agent": cfg.get("agent", "moma_llm")})
    
    wandb.init(
        project=wandb_cfg.get("project_inference", "smallplan-habitat-inference"),
        entity=wandb_cfg.get("entity"),
        config=cfg,
        mode=wandb_cfg.get("mode", "online") if cfg.get("wandb", True) else "disabled",
        name=run_name
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
        infos, tot_ep = evaluate_scene(
            config_file=config_file,
            cfg=cfg,
            scene_id=scene_id,
            tot_ep=tot_ep,
            slm_api_url=slm_api_url,
            mode=mode,
            save_video=args.save_video,
            video_dir=args.video_dir
        )
        episode_infos[scene_id] = infos
        
    log_summary_table(episode_infos=episode_infos)
    plot_efficiency_curves(episode_infos=episode_infos,
                          max_hl_steps=cfg.get("max_high_level_steps", 50))
    
    # Log overall token metrics
    log_token_metrics(episode_infos)
    
    wandb.run.finish()
    logger.info("Inference completed successfully.")


if __name__ == "__main__":
    main()
