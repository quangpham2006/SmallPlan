# Habitat Multi-LLM Inference Script for SmallPlan
# Dual-LLM system with a planning LLM and a storyteller LLM
#
# This script provides an alternative to habitat_inference.py that uses
# two LLMs working together:
# 1. Main Planning LLM: Decides actions (same as single-LLM version)
# 2. Storyteller LLM: Generates narrative summaries of exploration
#
# The storyteller provides temporal context to help the planning LLM
# understand what has been tried and make better decisions.

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
_run_video_dir = None
_inference_logger = None  # Real-time inference statistics logger


def _save_video_on_exit():
    """Save video when script exits (cleanup handler) - saves to 'error' subdirectory."""
    global _current_env, _current_video_path, _video_saved, _run_video_dir, _inference_logger
    
    if _video_saved:
        return
        
    if _current_env is not None and _current_video_path is not None:
        try:
            video_filename = os.path.basename(_current_video_path)
            if _run_video_dir:
                error_dir = os.path.join(_run_video_dir, "error")
            else:
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

# Import Habitat-compatible modules
from src.train_from_simulation_habitat.packages.moma_llm.env.habitat_env import (
    OurHabitatEnv,
    create_habitat_env
)
from src.train_from_simulation_habitat.packages.moma_llm.env.habitat_multi_llm_env import (
    HabitatMultiLLMEnv
)
from src.train_from_simulation_habitat.packages.moma_llm.llm.habitat_llm import LLM_hugging
from src.train_from_simulation_habitat.packages.moma_llm.tasks.habitat_object_search_task import HabitatObjectSearchTask
from src.train_from_simulation_habitat.packages.moma_llm.utils.habitat_constants import (
    TRAINING_SCENES,
    TEST_SCENES,
    get_scenes_for_dataset,
    NODETYPE,
    POSSIBLE_ROOMS
)

# Import from habitat_train for shared functionality
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


def create_multi_llm_env(cfg: Dict,
                         config_file: str,
                         scene_id: str,
                         control_freq: float,
                         seed: int,
                         slm_api_url: str,
                         mode: str = "headless",
                         storyteller_model: str = "gpt-4o-mini",
                         storyteller_temperature: float = 0.3,
                         enable_storyteller: bool = True,
                         debug_storyteller: bool = False,
                         prompt_version: int = 2):
    """
    Create Multi-LLM environment.
    
    Args:
        cfg: Configuration dictionary
        config_file: Path to config file
        scene_id: Scene identifier
        control_freq: Control frequency
        seed: Random seed
        slm_api_url: SLM API URL
        mode: Rendering mode ("headless" or "gui")
        storyteller_model: Model for storyteller LLM
        storyteller_temperature: Temperature for storyteller
        enable_storyteller: Whether to enable storyteller
        debug_storyteller: Whether to debug storyteller
        prompt_version: Prompt version to use (1, 2, or 4)
        
    Returns:
        HabitatMultiLLMEnv instance
    """
    open_set_rooms_value = cfg.get("open_set_room_categories", True)
    logger.info(f"Creating LLM with open_set_rooms={open_set_rooms_value}")
    
    llm = LLM_hugging(
        debug=True,
        room_classification_model="gpt-4o",
        open_set_rooms=open_set_rooms_value,
        slm_api_url=slm_api_url
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
    
    # Create multi-LLM environment
    high_level_env = HabitatMultiLLMEnv(
        env=low_level_env, 
        llm=llm, 
        seed=seed,
        storyteller_model=storyteller_model,
        storyteller_temperature=storyteller_temperature,
        enable_storyteller=enable_storyteller,
        debug_storyteller=debug_storyteller,
        prompt_version=prompt_version
    )
    
    return high_level_env


def log_scene_token_metrics(scene_id: str, episode_infos: List[Dict]):
    """Log token usage metrics for a specific scene to WandB (including storyteller)."""
    
    # Main LLM metrics
    input_tokens = [e.get('episode_input_tokens', 0) for e in episode_infos 
                    if 'episode_input_tokens' in e]
    output_tokens = [e.get('episode_output_tokens', 0) for e in episode_infos 
                     if 'episode_output_tokens' in e]
    total_tokens = [e.get('episode_total_tokens', 0) for e in episode_infos 
                    if 'episode_total_tokens' in e]
    queries = [e.get('episode_llm_queries', 0) for e in episode_infos 
               if 'episode_llm_queries' in e]
    
    # Storyteller metrics
    storyteller_input = [e.get('storyteller_input_tokens', 0) for e in episode_infos 
                         if 'storyteller_input_tokens' in e]
    storyteller_output = [e.get('storyteller_output_tokens', 0) for e in episode_infos 
                          if 'storyteller_output_tokens' in e]
    storyteller_queries = [e.get('storyteller_queries', 0) for e in episode_infos 
                           if 'storyteller_queries' in e]
    
    # Combined metrics
    combined_total = [e.get('combined_total_tokens', 0) for e in episode_infos 
                      if 'combined_total_tokens' in e]
    
    if not total_tokens:
        print(f"No token metrics found for scene {scene_id}")
        return
    
    scene_token_summary = {
        # Main LLM
        f'{scene_id}_avg_planner_input_tokens': np.mean(input_tokens) if input_tokens else 0,
        f'{scene_id}_avg_planner_output_tokens': np.mean(output_tokens) if output_tokens else 0,
        f'{scene_id}_avg_planner_total_tokens': np.mean(total_tokens) if total_tokens else 0,
        f'{scene_id}_avg_planner_queries': np.mean(queries) if queries else 0,
        # Storyteller
        f'{scene_id}_avg_storyteller_input_tokens': np.mean(storyteller_input) if storyteller_input else 0,
        f'{scene_id}_avg_storyteller_output_tokens': np.mean(storyteller_output) if storyteller_output else 0,
        f'{scene_id}_avg_storyteller_queries': np.mean(storyteller_queries) if storyteller_queries else 0,
        # Combined
        f'{scene_id}_avg_combined_total_tokens': np.mean(combined_total) if combined_total else 0,
        f'{scene_id}_total_combined_tokens': sum(combined_total) if combined_total else 0,
    }
    
    wandb.log(scene_token_summary)
    
    print(f"=== Token Usage Summary for {scene_id} (Multi-LLM) ===")
    for key, value in scene_token_summary.items():
        clean_key = key.replace(f'{scene_id}_', '')
        print(f"{clean_key}: {value:.2f}" if isinstance(value, float) else f"{clean_key}: {value}")
    print("=" * (40 + len(scene_id)))


def log_token_metrics(episode_infos: Dict[str, List[Dict]]):
    """Log overall average token usage metrics to WandB (including storyteller)."""
    
    all_episodes = []
    for scene_id in episode_infos:
        all_episodes.extend(episode_infos[scene_id])
    
    # Main LLM metrics
    input_tokens = [e.get('episode_input_tokens', 0) for e in all_episodes 
                    if 'episode_input_tokens' in e]
    output_tokens = [e.get('episode_output_tokens', 0) for e in all_episodes 
                     if 'episode_output_tokens' in e]
    
    # Storyteller metrics
    storyteller_input = [e.get('storyteller_input_tokens', 0) for e in all_episodes 
                         if 'storyteller_input_tokens' in e]
    storyteller_output = [e.get('storyteller_output_tokens', 0) for e in all_episodes 
                          if 'storyteller_output_tokens' in e]
    storyteller_queries = [e.get('storyteller_queries', 0) for e in all_episodes 
                           if 'storyteller_queries' in e]
    
    # Combined metrics
    combined_total = [e.get('combined_total_tokens', 0) for e in all_episodes 
                      if 'combined_total_tokens' in e]
    
    overall_token_summary = {
        # Main LLM
        'overall_avg_planner_input_tokens': np.mean(input_tokens) if input_tokens else 0,
        'overall_avg_planner_output_tokens': np.mean(output_tokens) if output_tokens else 0,
        'overall_total_planner_tokens': sum(input_tokens) + sum(output_tokens) if input_tokens else 0,
        # Storyteller
        'overall_avg_storyteller_input_tokens': np.mean(storyteller_input) if storyteller_input else 0,
        'overall_avg_storyteller_output_tokens': np.mean(storyteller_output) if storyteller_output else 0,
        'overall_total_storyteller_tokens': sum(storyteller_input) + sum(storyteller_output) if storyteller_input else 0,
        'overall_total_storyteller_queries': sum(storyteller_queries) if storyteller_queries else 0,
        # Combined
        'overall_avg_combined_total_tokens': np.mean(combined_total) if combined_total else 0,
        'overall_total_combined_tokens': sum(combined_total) if combined_total else 0,
    }
    
    wandb.log(overall_token_summary)
    
    print("=== Overall Token Usage Summary (Multi-LLM) ===")
    for key, value in overall_token_summary.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    print("=" * 50)


def evaluate_scene(config_file: str, 
                   cfg: Dict, 
                   scene_id: str, 
                   tot_ep: int, 
                   slm_api_url: str,
                   mode: str = "headless",
                   save_video: bool = False,
                   video_dir: str = None,
                   storyteller_model: str = "gpt-4o-mini",
                   storyteller_temperature: float = 0.3,
                   enable_storyteller: bool = True,
                   debug_storyteller: bool = False,
                   prompt_version: int = 2) -> tuple:
    """
    Evaluate on a single scene using Multi-LLM system.
    
    Args:
        config_file: Path to config file
        cfg: Configuration dictionary
        scene_id: Scene identifier
        tot_ep: Total episode count
        slm_api_url: SLM API URL
        mode: Rendering mode
        save_video: Whether to save videos
        video_dir: Directory for videos
        storyteller_model: Model for storyteller
        storyteller_temperature: Temperature for storyteller
        enable_storyteller: Whether to enable storyteller
        debug_storyteller: Whether to debug storyteller
        prompt_version: Prompt version to use (1, 2, or 4)
        
    Returns:
        Tuple of (episode_infos, tot_ep)
    """
    global _current_env, _current_video_path, _video_saved, _run_video_dir, _inference_logger
    
    if video_dir is None:
        video_dir = _run_video_dir
    
    logger.info(f"evaluate_scene (Multi-LLM) called for {scene_id}")
    logger.info(f"  Storyteller enabled: {enable_storyteller}")
    logger.info(f"  Storyteller model: {storyteller_model}")
    logger.info(f"  Prompt version: {prompt_version}")
    
    if save_video and video_dir:
        os.makedirs(video_dir, exist_ok=True)
    
    episode_infos = []
    
    high_level_env = create_multi_llm_env(
        cfg=cfg,
        config_file=config_file,
        scene_id=scene_id,
        control_freq=cfg.get("control_freq", 10.0),
        seed=cfg.get("seed", 42),
        slm_api_url=slm_api_url,
        mode=mode,
        storyteller_model=storyteller_model,
        storyteller_temperature=storyteller_temperature,
        enable_storyteller=enable_storyteller,
        debug_storyteller=debug_storyteller,
        prompt_version=prompt_version
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
        print(f"[Multi-LLM] {scene_id} - Starting episode {i + 1} in scene {scene_id}, "
              f"{tot_ep + 1} overall. Task: {high_level_env.env.task.task_description}")
        print(f"[Multi-LLM] Storyteller: {'ENABLED' if enable_storyteller else 'DISABLED'}")
        print("########################################")
        
        # Log episode start
        if _inference_logger is not None:
            _inference_logger.start_episode(
                scene_id=scene_id,
                episode_num=i,
                task_description=high_level_env.env.task.task_description
            )
        
        while not done:
            high_level_env.visualize(obs)
            done, task_success, episode_info = high_level_env.take_action_inference(
                obs=obs,
                task_description=high_level_env.env.task.task_description
            )
            
            wandb.log({"bev_maps": high_level_env.env.f})
            obs = high_level_env.get_state(compute_scene_graph=True)
            
            # Log current story (if enabled)
            if enable_storyteller:
                current_story = high_level_env.get_current_story()
                if current_story:
                    logger.info(f"[Storyteller] Current narrative:\n{current_story}")
            
            pprint(episode_info)
            
        high_level_env.visualize(obs)
        
        # Save video if requested
        if save_video and hasattr(high_level_env.env, 'save_video'):
            outcome_subdir = "success" if task_success else "failed"
            outcome_dir = os.path.join(video_dir, outcome_subdir)
            os.makedirs(outcome_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"{timestamp}_{scene_id}_episode_{i}_tot_{tot_ep}.mp4"
            video_path = os.path.join(outcome_dir, video_filename)
            
            # Pass task description for video overlay
            task_desc = high_level_env.env.task.task_description if high_level_env.env.task else None
            high_level_env.env.save_video(video_path, fps=10, task_text=task_desc)
            _video_saved = True
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
        episode_info["multi_llm_enabled"] = enable_storyteller
        
        # Handle None values for shortest_dist and dist_travelled
        shortest_dist = episode_info.get("shortest_dist") or 1
        dist_travelled = episode_info.get("dist_travelled") or 1
        episode_info["spl"] = episode_info.get("task_success", False) * (
            shortest_dist / max(shortest_dist, dist_travelled)
        )
        
        # Log final story
        if enable_storyteller:
            final_story = high_level_env.get_current_story()
            episode_info["final_story"] = final_story
            logger.info(f"[Storyteller] Final narrative for episode:\n{final_story}")
        
        pprint(episode_info)
        wandb.log({k: float(v) if isinstance(v, bool) else v 
                   for k, v in episode_info.items() if k != "final_story"})
        
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
    parser = argparse.ArgumentParser(description="Habitat Multi-LLM Inference for SmallPlan")
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
        default="./videos/multi_llm",
        help="Directory to save videos (default: ./videos/multi_llm)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose DEBUG output for navigation and other components"
    )
    # Storyteller-specific arguments
    parser.add_argument(
        "--no-storyteller",
        action="store_true",
        help="Disable the storyteller LLM (for ablation/comparison)"
    )
    parser.add_argument(
        "--storyteller-model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for storyteller LLM (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--storyteller-temperature",
        type=float,
        default=0.3,
        help="Temperature for storyteller LLM (default: 0.3)"
    )
    parser.add_argument(
        "--debug-storyteller",
        action="store_true",
        help="Enable debug output for storyteller LLM"
    )
    parser.add_argument(
        "--prompt-version",
        type=int,
        default=None,
        choices=[1, 2, 4],
        help="Prompt version to use: 1 (v1 basic), 2 (v3 with objects), 4 (v4 simplified). Default: from config"
    )
    return parser.parse_args()


def main():
    """Main inference function for Multi-LLM system."""
    global _run_video_dir, _inference_logger
    
    args = parse_args()
    
    # Determine rendering mode
    mode = "gui" if args.gui else args.mode
    logger.info(f"Running in {mode} mode")
    
    # Set verbose mode
    verbose = args.verbose
    if verbose:
        logger.info("Verbose mode enabled - DEBUG messages will be printed")
    
    # Storyteller settings
    enable_storyteller = not args.no_storyteller
    storyteller_model = args.storyteller_model
    storyteller_temperature = args.storyteller_temperature
    debug_storyteller = args.debug_storyteller
    
    logger.info("=" * 60)
    logger.info("MULTI-LLM INFERENCE MODE")
    logger.info(f"  Storyteller enabled: {enable_storyteller}")
    if enable_storyteller:
        logger.info(f"  Storyteller model: {storyteller_model}")
        logger.info(f"  Storyteller temperature: {storyteller_temperature}")
        logger.info(f"  Debug storyteller: {debug_storyteller}")
    logger.info(f"  Prompt version: {args.prompt_version or 'from config'}")
    logger.info("=" * 60)
    
    # Create run-specific video directory
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.save_video:
        _run_video_dir = os.path.join(args.video_dir, run_timestamp)
        os.makedirs(_run_video_dir, exist_ok=True)
        os.makedirs(os.path.join(_run_video_dir, "success"), exist_ok=True)
        os.makedirs(os.path.join(_run_video_dir, "failed"), exist_ok=True)
        os.makedirs(os.path.join(_run_video_dir, "error"), exist_ok=True)
        logger.info(f"Videos will be saved to: {_run_video_dir}")
        
        # Create inference logger for real-time statistics
        agent_type = "multi-llm" if enable_storyteller else "multi-llm-no-storyteller"
        _inference_logger = create_inference_logger(
            video_dir=_run_video_dir,
            run_name=f"multi-llm-{run_timestamp}",
            agent_type=agent_type,
            enabled=True
        )
    
    np.set_printoptions(precision=3, suppress=True)
    
    config_file, cfg, wandb_cfg, slm_training_cfg = setup_cfgs()
    
    logger.info(f"Config loaded from {config_file}")
    
    # Override prompt version from command line if specified
    if args.prompt_version is not None:
        cfg['prompt_version'] = args.prompt_version
        logger.info(f"Prompt version overridden from command line: {args.prompt_version}")
    
    # Print prompt version
    prompt_version = cfg.get('prompt_version', 1)
    logger.info(f"PROMPT VERSION: {prompt_version}")
    
    # Add verbose flag to config
    cfg["verbose"] = verbose
    
    # Create descriptive run name
    run_name = f"multi-llm-v{prompt_version}-{slm_training_cfg.get('slm_api_model', 'default')}"
    if not enable_storyteller:
        run_name += "-no-storyteller"
    
    slm_api_url = f"http://{slm_training_cfg['slm_api_host']}:{slm_training_cfg['slm_api_port']}"
    
    if cfg.get("seed", 0) > 0:
        np.random.seed(cfg["seed"])
    
    # Get scene IDs
    dataset = cfg.get("habitat", {}).get("dataset", "hm3d")
    if cfg.get("datasplit") == "train":
        scene_ids = get_scenes_for_dataset(dataset, "train")
    elif cfg.get("datasplit") == "test":
        scene_ids = get_scenes_for_dataset(dataset, "test")
    else:
        raise ValueError(f"Unknown datasplit {cfg.get('datasplit')}")
    
    cfg.update({"scene_ids": scene_ids, "agent": "multi_llm"})
    
    # Add multi-LLM specific config to wandb
    cfg["multi_llm"] = {
        "enabled": enable_storyteller,
        "storyteller_model": storyteller_model,
        "storyteller_temperature": storyteller_temperature,
        "prompt_version": prompt_version,
    }
    
    wandb.init(
        project=wandb_cfg.get("project_inference", "smallplan-habitat-inference"),
        entity=wandb_cfg.get("entity"),
        config=cfg,
        mode=wandb_cfg.get("mode", "online") if cfg.get("wandb", True) else "disabled",
        name=run_name,
        tags=["multi-llm"] if enable_storyteller else ["multi-llm", "ablation-no-storyteller"]
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
            video_dir=_run_video_dir,
            storyteller_model=storyteller_model,
            storyteller_temperature=storyteller_temperature,
            enable_storyteller=enable_storyteller,
            debug_storyteller=debug_storyteller,
            prompt_version=prompt_version
        )
        episode_infos[scene_id] = infos
        
    log_summary_table(episode_infos=episode_infos)
    plot_efficiency_curves(episode_infos=episode_infos,
                          max_hl_steps=cfg.get("max_high_level_steps", 50))
    
    # Log overall token metrics
    log_token_metrics(episode_infos)
    
    # Finalize inference logger
    if _inference_logger is not None:
        _inference_logger.finalize()
        _inference_logger.print_current_stats()
    
    wandb.run.finish()
    logger.info("Multi-LLM Inference completed successfully.")


if __name__ == "__main__":
    main()

