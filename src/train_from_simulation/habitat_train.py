# Habitat Training Script for SmallPlan
# Replaces iGibson-based train.py for Habitat-Lab/Habitat-Sim

import shutil
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Literal, Dict, List, Any, Optional
import yaml

import numpy as np
import pandas as pd
from sklearn.metrics import auc
import wandb

import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Import Habitat-compatible modules
from src.train_from_simulation.packages.moma_llm.llm.habitat_llm import (
    LLM_hugging, 
    Conversation,
    object_states
)
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


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def create_env(cfg: Dict,
               agent: str,
               config_file: str,
               scene_id: str,
               control_freq: float,
               cheap: bool,
               seed: int,
               slm_api_url: str,
               mode: str = "headless"):
    """
    Create environment for given agent type.
    
    Args:
        cfg: Configuration dictionary
        agent: Agent type (moma_llm, json_llm, greedy, random)
        config_file: Path to config file
        scene_id: Scene identifier
        control_freq: Control frequency
        cheap: Whether to use cheap mode
        seed: Random seed
        slm_api_url: SLM API URL
        mode: Rendering mode ("headless" or "gui")
        
    Returns:
        High-level environment
    """
    # Select environment class based on agent type
    if agent == "moma_llm" or agent == "json_llm":
        env_fn = HabitatLLMEnv
    elif agent == "greedy":
        env_fn = HabitatGreedyBaseline
    elif agent == "random":
        env_fn = HabitatRandomBaseline
    else:
        raise ValueError(f"Unknown agent type: {agent}")
    
    llm = LLM_hugging(
        debug=True,
        room_classification_model="gpt-4o",
        open_set_rooms=cfg.get("open_set_room_categories", True),
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
    
    high_level_env = env_fn(env=low_level_env, llm=llm, seed=seed)
    return high_level_env


def calc_area_under_curve(x: np.ndarray, y: np.ndarray, max_x: float) -> float:
    """Calculate area under curve."""
    if max(x) > max_x:
        idx = (x <= max_x)
        x = x[idx]
        y = y[idx]
    if max(x) < max_x:
        x = np.concatenate([x, [max_x]])
        y = np.concatenate([y, [y[-1]]])
    
    x = np.concatenate([[0], x])
    y = np.concatenate([[0], y])
    return auc(x, y) / max_x


def plot_efficiency_curves(episode_infos: Dict, max_hl_steps: int):
    """Plot efficiency curves to wandb."""
    ll_steps = []
    ll_steps_gtDone = []
    hl_steps = []
    task_success = []
    task_success_gtDone = []
    
    for scene_id in sorted(episode_infos.keys()):
        for e in episode_infos[scene_id]:
            ll_steps.append(e.get("num_low_level_steps_with_open_cost", 0))
            ll_steps_gtDone.append(e.get("num_low_level_steps_with_open_cost_gtDone", 0))
            hl_steps.append(e.get("num_high_level_steps", 0))
            task_success.append(e.get("task_success", False))
            task_success_gtDone.append(e.get("task_success_gtDone", False))
            
    task_success = np.array(task_success)
    task_success_gtDone = np.array(task_success_gtDone)
    ll_steps = np.array(ll_steps)
    hl_steps = np.array(hl_steps)
    
    def _plot(steps, task_success):
        df = pd.DataFrame({"steps": steps, "task_success": task_success})
        df = df.sort_values("steps")
        values = [np.logical_and(df["task_success"].values, df["steps"].values <= max_steps).mean() 
                  for max_steps in df["steps"]]
        df2 = pd.DataFrame({"steps": df["steps"].values, "success": values})
        return wandb.Table(dataframe=df2)
    
    def _get_auc(steps, task_success, max_x: int, title: str):
        table = _plot(steps=steps, task_success=task_success)
        auc_val = calc_area_under_curve(
            table.get_dataframe()["steps"].values,
            table.get_dataframe()["success"].values,
            max_x=max_x
        )
        wandb_plot = wandb.plot_table(
            "wandb/area-under-curve/v0",
            table,
            {"x": "steps", "y": "success"},
            {"title": title, "x-axis-title": "Steps", "y-axis-title": "Success rate"}
        )
        return auc_val, wandb_plot
    
    hl_auc, hl_plot = _get_auc(steps=hl_steps, task_success=task_success, 
                               max_x=max_hl_steps, title="High-level-step-curve")
    wandb.log({"efficiency_curve_high_level_steps": hl_plot, "hl_auc": hl_auc})
    
    ll_auc_max_steps = 5000
    ll_auc, ll_plot = _get_auc(steps=ll_steps, task_success=task_success,
                               max_x=ll_auc_max_steps, title="Low-level-step-curve")
    wandb.log({"efficiency_curve_low_level_steps": ll_plot, "ll_auc": ll_auc})
    
    ll_auc_gtDone, ll_plot_gtDone = _get_auc(
        steps=ll_steps_gtDone, task_success=task_success_gtDone,
        max_x=ll_auc_max_steps, title="Low-level-step-curve-gtDone"
    )
    wandb.log({"efficiency_curve_low_level_steps_gtDone": ll_plot_gtDone, 
               "ll_auc_gtDone": ll_auc_gtDone})


def calculate_metric_means(episode_infos: Dict) -> Dict:
    """Calculate mean metrics per scene."""
    columns = sorted(list(episode_infos.values())[0][0].keys())
    scene_logs = defaultdict(dict)
    
    for scene_id in sorted(episode_infos.keys()):
        for column in columns:
            if isinstance(episode_infos[scene_id][0].get(column, None), str):
                continue
            elif isinstance(episode_infos[scene_id][0].get(column, None), (np.ScalarType)):
                d = np.nanmean([e.get(column, np.nan) for e in episode_infos[scene_id]])
                scene_logs[scene_id][column] = d
            else:
                continue
            print(column, d)
    return scene_logs


def log_summary_table(episode_infos: Dict):
    """Log summary table to wandb."""
    def _check_float(v):
        try:
            float(v)
            return True
        except:
            return False
    
    scene_logs = calculate_metric_means(episode_infos)
    data = []
    scenes = list(scene_logs.keys())
    columns = ["scene_id"]
    
    for k in scene_logs[scenes[0]].keys():
        if all([k in scene_logs[s] for s in scenes]):
            columns.append(k)
    
    for scene_id in sorted(scenes):
        data.append([scene_id] + [scene_logs[scene_id][c] for c in columns[1:]])
        
    avg_row = ["Overall avg"]
    avg_dict = {}
    for i, column_values in enumerate(np.array(data).T):
        if _check_float(column_values[0]):
            d = np.mean(column_values.astype(float))
            avg_dict[f"avg_{columns[i]}"] = d
            avg_row.append(str(d))
    data.append(avg_row)
    avg_dict["overview_table"] = wandb.Table(columns=columns, data=np.array(data).astype(str))
    wandb.log(avg_dict)


def train_scene(config_file: str,
                cfg: Dict,
                scene_id: str,
                tot_ep: int,
                save_dir: str,
                slm_api_url: str,
                strategy: Literal["RL-SFT", "SFT", "SFT-RL"]) -> tuple:
    """
    Train on a single scene.
    
    Args:
        config_file: Path to config file
        cfg: Configuration dictionary
        scene_id: Scene identifier
        tot_ep: Total episode count
        save_dir: Directory to save checkpoints
        slm_api_url: SLM API URL
        strategy: Training strategy
        
    Returns:
        Tuple of (episode_infos, tot_ep)
    """
    episode_infos = []
    
    high_level_env = create_env(
        cfg=cfg,
        agent=cfg.get("agent", "moma_llm"),
        config_file=config_file,
        scene_id=scene_id,
        control_freq=cfg.get("control_freq", 10.0),
        cheap=cfg.get("cheap", False),
        seed=cfg.get("seed", 42),
        slm_api_url=slm_api_url
    )
    
    for i in range(cfg.get("num_episodes_per_scene", 2)):
        done = False
        obs = high_level_env.reset(config_file=config_file, scene_id=scene_id, episode_num=i)
        
        print("########################################")
        logger.info(f"{scene_id} - Starting episode {i + 1} in scene {scene_id}, "
                   f"{tot_ep + 1} overall. Task: {high_level_env.env.task.task_description}")
        print("########################################")
        
        while not done:
            high_level_env.visualize(obs)
            done, task_success, episode_info = high_level_env.take_action(
                obs=obs,
                task_description=high_level_env.env.task.task_description,
                strategy=strategy
            )
            
            wandb.log({"bev_maps": high_level_env.env.f})
            obs = high_level_env.get_state(compute_scene_graph=True)
            pprint(episode_info)
            
        high_level_env.visualize(obs)
        
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
        episode_info["spl"] = episode_info.get("task_success", False) * (
            episode_info.get("shortest_dist", 1) / 
            max(episode_info.get("shortest_dist", 1), episode_info.get("dist_travelled", 1))
        )
        
        pprint(episode_info)
        wandb.log({k: float(v) if isinstance(v, bool) else v 
                   for k, v in episode_info.items()})
        
        episode_infos.append(episode_info)
        successes = [e.get("task_success", False) for e in episode_infos]
        tot_ep += 1
        
        logger.info(f"Task success: {task_success} (wandb_step: {wandb.run.step}). "
                   f"Current successes: {sum(successes)}/{len(successes)}")
        
        high_level_env.llm.save_checkpoint(
            save_name=f"{save_dir}/checkpoint_{scene_id}_eps_{i}"
        )
        
    scene_logs = calculate_metric_means({scene_id: episode_infos})
    wandb.log({f"{scene_id}_{k}": v for k, v in scene_logs[scene_id].items()})
    
    high_level_env.close()
    return episode_infos, tot_ep


def setup_cfgs():
    """Setup configurations."""
    config_file = "./configs/moma_llm_habitat.yaml"
    cfg = load_config(config_file)
    wandb_cfg = load_config("./configs/wandb.yaml")
    slm_training_cfg = load_config("./configs/slm_training.yaml")
    
    return config_file, cfg, wandb_cfg, slm_training_cfg


def main():
    """Main training function."""
    np.set_printoptions(precision=3, suppress=True)
    
    config_file, cfg, wandb_cfg, slm_training_cfg = setup_cfgs()
    save_dir = (f"{slm_training_cfg['smallplan_outputs_path']}/"
                f"{slm_training_cfg['strategy']}-{slm_training_cfg['model_tag']}-"
                f"{slm_training_cfg['slm_api_model']}")
    slm_api_url = f"http://{slm_training_cfg['slm_api_host']}:{slm_training_cfg['slm_api_port']}"
    
    if cfg.get("seed", 0) > 0:
        np.random.seed(cfg["seed"])
    
    # Get scene IDs based on dataset and split
    dataset = cfg.get("habitat", {}).get("dataset", "hm3d")
    if cfg.get("datasplit") == "train":
        scene_ids = get_scenes_for_dataset(dataset, "train")
    elif cfg.get("datasplit") == "test":
        scene_ids = get_scenes_for_dataset(dataset, "test")
    else:
        raise ValueError(f"Unknown datasplit {cfg.get('datasplit')}")
    
    cfg.update({"scene_ids": scene_ids, "agent": cfg.get("agent", "moma_llm")})
    
    wandb.init(
        project=wandb_cfg.get("project", "smallplan-habitat"),
        entity=wandb_cfg.get("entity"),
        config=cfg,
        mode=wandb_cfg.get("mode", "online") if cfg.get("wandb", True) else "disabled"
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
        infos, tot_ep = train_scene(
            config_file=config_file,
            cfg=cfg,
            scene_id=scene_id,
            tot_ep=tot_ep,
            save_dir=save_dir,
            slm_api_url=slm_api_url,
            strategy=slm_training_cfg.get("strategy", "SFT")
        )
        episode_infos[scene_id] = infos
        
    log_summary_table(episode_infos=episode_infos)
    plot_efficiency_curves(episode_infos=episode_infos, 
                          max_hl_steps=cfg.get("max_high_level_steps", 50))
    
    wandb.run.finish()
    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()
