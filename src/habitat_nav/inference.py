"""
Habitat ObjectNav Inference Runner

Main entry point for running object navigation inference with LLM or baseline agents.

Usage:
    python -m src.habitat_nav.inference --scene 00800-TEEsavR23oF --episodes 5
    python -m src.habitat_nav.inference --agent random --episodes 10
    python -m src.habitat_nav.inference --config configs/habitat_nav.yaml

Reference: https://github.com/AIGeeksGroup/Nav-R1
"""

import argparse
import atexit
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

# Suppress Habitat logging noise
os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'

# Global state for cleanup on interrupt
_current_env = None
_current_video_path = None
_video_saved = False
_run_dir = None


def _save_video_on_exit():
    """Save video when script exits (cleanup handler) - saves to 'error' subdirectory."""
    global _current_env, _current_video_path, _video_saved, _run_dir
    
    if _video_saved:
        return
        
    if _current_env is not None and _current_video_path is not None:
        try:
            video_filename = os.path.basename(_current_video_path)
            if _run_dir:
                error_dir = os.path.join(_run_dir, "videos", "error")
            else:
                video_dir = os.path.dirname(_current_video_path)
                error_dir = os.path.join(os.path.dirname(video_dir), "error")
            os.makedirs(error_dir, exist_ok=True)
            error_video_path = os.path.join(error_dir, video_filename)
            
            logging.info(f"Saving video on exit (interrupted/error) to {error_video_path}")
            if hasattr(_current_env, 'save_video'):
                _current_env.save_video(error_video_path, fps=10)
            _video_saved = True
            logging.info("Video saved successfully to error directory")
        except Exception as e:
            logging.error(f"Failed to save video on exit: {e}")


def _signal_handler(signum, frame):
    """Handle interrupt signals (Ctrl+C)."""
    logging.info(f"\nReceived signal {signum}. Saving video before exit...")
    _save_video_on_exit()
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
atexit.register(_save_video_on_exit)

from .core.simulator import HabitatSimulator, SimulatorConfig
from .core.environment import ObjectNavEnv
from .core.observations import ProcessedObservation
from .core.action_executor import ActionExecutor, NavigationConfig
from .agents import BaseAgent, LLMAgent, RandomAgent, TwoPlannerAgent, ImagePlannerAgent
from .tasks.object_nav import ObjectNavTask, TaskResult
from .utils.actions import Action, HighLevelAction
from .utils.constants import get_scenes_for_dataset, DEFAULT_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference run."""
    # Scene settings
    scene_ids: List[str] = field(default_factory=list)
    dataset: str = "hm3d"
    split: str = "test"
    
    # Episode settings
    num_episodes_per_scene: int = 3
    max_episode_steps: int = 500  # Max low-level steps (for low-level action mode)
    max_llm_queries: int = 100  # Max LLM queries (for high-level action mode)
    success_distance: float = 1.5
    
    # Agent settings
    agent_type: str = "llm"  # "llm", "llm-openai", "random"
    planner_type: str = "single"  # "single", "two", or "image"
    llm_api_url: str = "http://localhost:8000/generate"
    llm_api_type: str = "custom"  # "custom" or "openai"
    llm_model: str = "gpt-4o"  # Model name for main LLM
    narrator_model: str = "gpt-4o-mini"  # Model for narrator (two-planner only)
    openai_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")
    llm_temperature: float = 0.7
    narrator_temperature: float = 0.7  # Narrator temperature (two-planner only)
    narrator_update_frequency: int = 3  # Generate story every N actions (two-planner only)
    action_level: str = "low"  # "low" for low-level actions, "high" for high-level actions
    
    # Image planner settings
    max_images: int = 3  # Number of images to send to LLM (image planner only)
    image_detail: str = "low"  # Image detail level: "low", "high", "auto" (image planner only)
    
    # Output settings
    output_dir: str = "./outputs/habitat_nav"
    save_videos: bool = True
    video_fps: int = 10
    
    # Navigation settings
    collect_action_images: bool = False  # Collect images during actions for LLM context
    collision_distance: float = 0.4      # Safe distance from obstacles (meters)
    
    # Video settings
    annotate_videos: bool = True         # Annotate video frames with object bounding boxes
    
    # Misc
    seed: int = 42
    verbose: bool = False
    
    @classmethod
    def from_yaml(cls, path: str) -> "InferenceConfig":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "InferenceConfig":
        """Create config from command line args."""
        config = cls()
        
        if args.scene:
            config.scene_ids = [args.scene]
        elif args.scenes:
            config.scene_ids = args.scenes.split(",")
        else:
            config.scene_ids = get_scenes_for_dataset(config.dataset, config.split)
        
        if args.episodes:
            config.num_episodes_per_scene = args.episodes
        if args.agent:
            config.agent_type = args.agent
            # Auto-detect OpenAI agent type
            if args.agent == "llm-openai":
                config.llm_api_type = "openai"
        if args.api_url:
            config.llm_api_url = args.api_url
        if args.api_type:
            config.llm_api_type = args.api_type
        if args.model:
            config.llm_model = args.model
        if args.openai_key:
            config.openai_api_key = args.openai_key
        if args.temperature:
            config.llm_temperature = args.temperature
        if hasattr(args, 'action_level') and args.action_level:
            config.action_level = args.action_level
        if hasattr(args, 'planner') and args.planner:
            config.planner_type = args.planner
        if hasattr(args, 'narrator_model') and args.narrator_model:
            config.narrator_model = args.narrator_model
        if hasattr(args, 'narrator_temperature') and args.narrator_temperature:
            config.narrator_temperature = args.narrator_temperature
        if hasattr(args, 'narrator_update_freq') and args.narrator_update_freq:
            config.narrator_update_frequency = args.narrator_update_freq
        if hasattr(args, 'max_images') and args.max_images:
            config.max_images = args.max_images
        if hasattr(args, 'image_detail') and args.image_detail:
            config.image_detail = args.image_detail
        if args.output:
            config.output_dir = args.output
        if args.no_video:
            config.save_videos = False
        if hasattr(args, 'no_annotations') and args.no_annotations:
            config.annotate_videos = False
        if args.verbose:
            config.verbose = True
        if args.seed:
            config.seed = args.seed
            
        return config


@dataclass
class EpisodeResult:
    """Result of a single episode."""
    scene_id: str
    episode_id: int
    target_category: str
    success: bool
    spl: float
    steps: int
    distance_travelled: float
    episode_time: float = 0.0  # Time in seconds
    failure_reason: Optional[str] = None
    video_path: Optional[str] = None
    final_interaction: Optional[Dict[str, Any]] = None  # Final LLM interaction
    
    def to_dict(self) -> Dict:
        return {
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "target_category": self.target_category,
            "success": self.success,
            "spl": self.spl,
            "steps": self.steps,
            "distance_travelled": self.distance_travelled,
            "episode_time": self.episode_time,
            "failure_reason": self.failure_reason,
        }


class InferenceRunner:
    """
    Runs ObjectNav inference across scenes and episodes.
    
    Manages environment setup, agent interaction, and result logging.
    """
    
    def __init__(self, config: InferenceConfig):
        """
        Initialize inference runner.
        
        Args:
            config: Inference configuration
        """
        self.config = config
        self.results: List[EpisodeResult] = []
        self.rng = np.random.RandomState(config.seed)
        
        # Create output directory
        self.run_dir = self._create_run_dir()
        
        # Agent (created per scene or shared)
        self.agent: Optional[BaseAgent] = None
        
        logger.info(f"Initialized InferenceRunner")
        logger.info(f"  Scenes: {len(config.scene_ids)}")
        logger.info(f"  Episodes per scene: {config.num_episodes_per_scene}")
        logger.info(f"  Agent type: {config.agent_type}")
        logger.info(f"  Output: {self.run_dir}")
    
    def _create_run_dir(self) -> Path:
        """Create timestamped run directory with planner type prefix."""
        global _run_dir
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        planner_type = self.config.planner_type  # "single", "two", or "image"
        run_name = f"{planner_type}_{timestamp}"
        run_dir = Path(self.config.output_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories (including error for interrupt/crash videos)
        (run_dir / "videos" / "success").mkdir(parents=True, exist_ok=True)
        (run_dir / "videos" / "failed").mkdir(parents=True, exist_ok=True)
        (run_dir / "videos" / "error").mkdir(parents=True, exist_ok=True)
        
        # Set global run_dir for interrupt handler
        _run_dir = str(run_dir)
        
        # Save config
        config_dict = {k: v for k, v in self.config.__dict__.items()}
        with open(run_dir / "config.yaml", 'w') as f:
            yaml.dump(config_dict, f)
        
        # Setup file logging
        self._setup_file_logging(run_dir)
        
        return run_dir
    
    def _setup_file_logging(self, run_dir: Path):
        """Setup logging - console only, log.txt written at end with clean format."""
        # We don't add a file handler here anymore
        # Instead, we write a clean log.txt at the end with _save_results()
        # This avoids verbose logging in the output file
        logger.info(f"Run directory: {run_dir}")
    
    def _create_agent(self) -> BaseAgent:
        """Create agent based on config."""
        if self.config.agent_type in ("llm", "llm-openai", "llm-custom"):
            # Determine API type
            if self.config.agent_type == "llm-openai":
                api_type = "openai"
            elif self.config.agent_type == "llm-custom":
                api_type = "custom"
            else:
                api_type = self.config.llm_api_type
            
            # Check planner type
            if self.config.planner_type == "two":
                logger.info(f"Creating Two-Planner agent: api_type={api_type}, "
                           f"main_model={self.config.llm_model}, "
                           f"narrator_model={self.config.narrator_model}, "
                           f"action_level={self.config.action_level}")
                
                return TwoPlannerAgent(
                    api_type=api_type,
                    api_url=self.config.llm_api_url,
                    main_model=self.config.llm_model,
                    narrator_model=self.config.narrator_model,
                    openai_api_key=self.config.openai_api_key,
                    temperature=self.config.llm_temperature,
                    narrator_temperature=self.config.narrator_temperature,
                    action_level=self.config.action_level,
                    narrator_update_frequency=self.config.narrator_update_frequency,
                    include_story=True,
                    name="two_planner_agent"
                )
            elif self.config.planner_type == "image":
                logger.info(f"Creating Image-Planner agent: "
                           f"model={self.config.llm_model}, "
                           f"max_images={self.config.max_images}, "
                           f"action_level={self.config.action_level}")
                
                return ImagePlannerAgent(
                    model_name=self.config.llm_model,
                    openai_api_key=self.config.openai_api_key,
                    temperature=self.config.llm_temperature,
                    action_level=self.config.action_level,
                    max_images=self.config.max_images,
                    image_detail=self.config.image_detail,
                    name="image_planner_agent"
                )
            else:
                # Single planner (default)
                logger.info(f"Creating Single-Planner LLM agent: api_type={api_type}, "
                           f"model={self.config.llm_model}, action_level={self.config.action_level}")
                
                return LLMAgent(
                    api_type=api_type,
                    api_url=self.config.llm_api_url,
                    model_name=self.config.llm_model,
                    openai_api_key=self.config.openai_api_key,
                    temperature=self.config.llm_temperature,
                    action_level=self.config.action_level,
                    name="llm_agent"
                )
        elif self.config.agent_type == "random":
            return RandomAgent(seed=self.config.seed, name="random_agent")
        else:
            raise ValueError(f"Unknown agent type: {self.config.agent_type}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run inference across all scenes and episodes.
        
        Returns:
            Summary statistics
        """
        logger.info("Starting inference run...")
        
        total_episodes = len(self.config.scene_ids) * self.config.num_episodes_per_scene
        completed = 0
        
        for scene_id in self.config.scene_ids:
            logger.info(f"Processing scene: {scene_id}")
            
            try:
                scene_results = self._run_scene(scene_id)
                self.results.extend(scene_results)
                completed += len(scene_results)
                
                # Log progress
                success_rate = sum(r.success for r in scene_results) / len(scene_results)
                logger.info(f"  Scene {scene_id}: {success_rate:.1%} success rate")
                
            except Exception as e:
                logger.error(f"Error processing scene {scene_id}: {e}")
                if self.config.verbose:
                    import traceback
                    traceback.print_exc()
        
        # Compute and save summary
        summary = self._compute_summary()
        self._save_results()
        
        logger.info("=" * 50)
        logger.info("INFERENCE COMPLETE")
        logger.info(f"  Total episodes: {len(self.results)}")
        logger.info(f"  Success rate: {summary['success_rate']:.1%}")
        logger.info(f"  Average SPL: {summary['avg_spl']:.3f}")
        logger.info(f"  Results saved to: {self.run_dir}")
        logger.info("=" * 50)
        
        return summary
    
    def _run_scene(self, scene_id: str) -> List[EpisodeResult]:
        """Run episodes for a single scene."""
        results = []
        
        # Create environment for this scene
        env = ObjectNavEnv(
            scene_id=scene_id,
            config=DEFAULT_CONFIG,
            max_episode_steps=self.config.max_episode_steps,
            success_distance=self.config.success_distance,
            seed=self.config.seed,
            annotate_videos=self.config.annotate_videos
        )
        
        # Create agent
        agent = self._create_agent()
        
        # Create action executor for high-level actions
        action_executor = None
        if self.config.action_level == "high":
            nav_config = NavigationConfig(
                collision_distance=self.config.collision_distance,
                success_distance=self.config.success_distance
            )
            action_executor = ActionExecutor(env, nav_config)
        
        try:
            for episode_idx in range(self.config.num_episodes_per_scene):
                result = self._run_episode(env, agent, episode_idx, action_executor)
                results.append(result)
                
                # Progress logging
                status = "✓" if result.success else "✗"
                logger.info(f"    Episode {episode_idx + 1}: {status} "
                           f"(target: {result.target_category}, steps: {result.steps})")
        finally:
            env.close()
        
        return results
    
    def _run_episode(self, 
                     env: ObjectNavEnv, 
                     agent: BaseAgent,
                     episode_idx: int,
                     action_executor: Optional[ActionExecutor] = None) -> EpisodeResult:
        """Run a single episode."""
        global _current_env, _current_video_path, _video_saved
        
        # Start timing
        episode_start_time = time.time()
        
        # Reset environment and agent
        obs, info = env.reset(episode_id=episode_idx)
        target = info["target"]
        agent.reset(target_category=target)
        
        if action_executor:
            action_executor.reset()
        
        # Set up global state for interrupt handling
        if self.config.save_videos:
            _current_env = env
            _current_video_path = str(
                self.run_dir / "videos" / "error" / 
                f"{env.scene_id}_ep{episode_idx}_{target}.mp4"
            )
            _video_saved = False
        
        done = False
        failure_reason = None
        llm_query_count = 0  # Track LLM queries for high-level timeout
        
        while not done:
            # Get action from agent
            action = agent.act(obs, env.get_task_description(), info)
            llm_query_count += 1  # Each act() call = one LLM query
            
            if self.config.action_level == "high" and action_executor and isinstance(action, HighLevelAction):
                # Execute high-level action
                if action.name == "stop":
                    obs, _, info = env.step(Action.STOP)
                    action_feedback = "STOP action executed"
                    result = None
                else:
                    result = action_executor.execute(action, obs)
                    action_feedback = result.feedback
                    # Get updated observation from result
                    if result.observation is not None:
                        obs = result.observation
                    # Always refresh info from environment to get current state
                    info = env.get_info()
                    
                    # Mark target as reached if goto succeeded to the target object
                    # This ensures success even if target isn't visible when stop() is called
                    if (action.name == "goto" and result.success and 
                        hasattr(action, 'argument') and action.argument):
                        target = info.get("target", "").lower()
                        goto_target = action.argument.lower()
                        # Check if goto target matches episode target (partial match)
                        if target in goto_target or goto_target in target:
                            env.mark_target_reached()
                
                # Add action result to info for agent's update method
                if result is not None:
                    info["action_result"] = result
                
                # Update high-level step count in episode info
                env.episode_info.high_level_step_count = llm_query_count
            else:
                # Execute low-level action
                original_action = action
                action = env.simulator.get_safe_action(action)
                
                obs, _, info = env.step(action)
                
                # Provide feedback about collision avoidance
                if action != original_action:
                    action_feedback = f"Collision avoided: {original_action.name} -> {action.name}"
                else:
                    action_feedback = f"Executed {action.name}"
            
            done = info["done"]
            
            # Add max_steps to info for agent's update method
            info["max_steps"] = self.config.max_episode_steps
            info["llm_query_count"] = llm_query_count
            info["action_feedback"] = action_feedback
            
            # Update agent with full info
            agent.update(action, obs, info)
            
            # Check for timeout based on action level
            if self.config.action_level == "high":
                # High-level mode: timeout based on LLM queries
                if llm_query_count >= self.config.max_llm_queries:
                    failure_reason = "timeout"
                    done = True
                    logger.info(f"Episode timed out after {llm_query_count} LLM queries")
            else:
                # Low-level mode: timeout based on action steps
                if info["step_count"] >= self.config.max_episode_steps:
                    failure_reason = "timeout"
                    done = True
        
        # Compute metrics
        episode_time = time.time() - episode_start_time
        success = info["success"]
        spl = env.compute_spl()
        
        if not success and not failure_reason:
            failure_reason = "target_not_found"
        
        # Get final LLM interaction for logging
        final_interaction = None
        if hasattr(agent, 'get_last_interaction'):
            final_interaction = agent.get_last_interaction()
            # Add success status to the interaction
            final_interaction["success"] = success
        
        # Log episode summary to console only
        self._log_episode_summary(
            env.scene_id, episode_idx, target, success, 
            info["step_count"], spl, episode_time, failure_reason, agent
        )
        
        # Save video
        video_path = None
        if self.config.save_videos:
            subdir = "success" if success else "failed"
            video_name = f"{env.scene_id}_ep{episode_idx}_{target}.mp4"
            video_path = str(self.run_dir / "videos" / subdir / video_name)
            env.save_video(video_path, fps=self.config.video_fps)
            _video_saved = True  # Mark as saved so interrupt handler doesn't save again
        
        return EpisodeResult(
            scene_id=env.scene_id,
            episode_id=episode_idx,
            target_category=target,
            success=success,
            spl=spl,
            steps=info["step_count"],
            distance_travelled=info["distance_travelled"],
            episode_time=episode_time,
            failure_reason=failure_reason,
            video_path=video_path,
            final_interaction=final_interaction,
        )
    
    def _log_episode_summary(self,
                             scene_id: str,
                             episode_idx: int,
                             target: str,
                             success: bool,
                             steps: int,
                             spl: float,
                             episode_time: float,
                             failure_reason: Optional[str],
                             agent: BaseAgent):
        """Log a brief summary of the episode to console."""
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"Episode {episode_idx} [{status}] - Target: {target}, "
                   f"Steps: {steps}, SPL: {spl:.3f}, Time: {episode_time:.1f}s")
    
    def _compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        if not self.results:
            return {"success_rate": 0.0, "avg_spl": 0.0, "avg_time": 0.0, "avg_steps": 0.0}
        
        successes = [r.success for r in self.results]
        spls = [r.spl for r in self.results]
        steps = [r.steps for r in self.results]
        times = [r.episode_time for r in self.results]
        
        return {
            "total_episodes": len(self.results),
            "success_rate": np.mean(successes),
            "avg_spl": np.mean(spls),
            "avg_steps": np.mean(steps),
            "avg_time": np.mean(times),
            "std_steps": np.std(steps),
            "std_time": np.std(times),
            "num_scenes": len(set(r.scene_id for r in self.results)),
        }
    
    def _save_results(self):
        """Save results to files."""
        # Save detailed results as YAML
        results_data = [r.to_dict() for r in self.results]
        with open(self.run_dir / "results.yaml", 'w') as f:
            yaml.dump(results_data, f)
        
        # Save summary
        summary = self._compute_summary()
        with open(self.run_dir / "summary.yaml", 'w') as f:
            yaml.dump(summary, f)
        
        # Save clean log.txt with only final interactions and summary
        self._write_clean_log(summary)
    
    def _write_clean_log(self, summary: Dict[str, Any]):
        """
        Write a clean log.txt with only summary metrics and final LLM interactions.
        
        Format:
        - Summary section with SUCCESS RATE, SPL, AVG TIME, AVG STEPS
        - Per-episode section with final system prompt, user prompt, response, action, success
        """
        with open(self.run_dir / "log.txt", 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("HABITAT OBJECTNAV INFERENCE RESULTS\n")
            f.write(f"Run: {self.run_dir.name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary Section
            f.write("=" * 80 + "\n")
            f.write("SUMMARY METRICS\n")
            f.write("=" * 80 + "\n\n")
            
            success_rate = summary.get('success_rate', 0.0) * 100
            avg_spl = summary.get('avg_spl', 0.0)
            avg_time = summary.get('avg_time', 0.0)
            avg_steps = summary.get('avg_steps', 0.0)
            
            f.write(f"  SUCCESS RATE:      {success_rate:.1f}%\n")
            f.write(f"  SPL:               {avg_spl:.4f}\n")
            f.write(f"  AVG TIME:          {avg_time:.2f} seconds\n")
            f.write(f"  AVG STEPS:         {avg_steps:.1f}\n")
            f.write(f"\n")
            f.write(f"  Total Episodes:    {summary.get('total_episodes', 0)}\n")
            f.write(f"  Num Scenes:        {summary.get('num_scenes', 0)}\n")
            f.write(f"  Std Steps:         {summary.get('std_steps', 0.0):.2f}\n")
            f.write(f"  Std Time:          {summary.get('std_time', 0.0):.2f} seconds\n")
            f.write("\n")
            
            # Per-Episode Final Interactions
            f.write("=" * 80 + "\n")
            f.write("PER-EPISODE FINAL LLM INTERACTIONS\n")
            f.write("=" * 80 + "\n\n")
            
            for i, r in enumerate(self.results):
                status = "SUCCESS" if r.success else "FAILED"
                
                f.write("-" * 80 + "\n")
                f.write(f"EPISODE {r.episode_id} | {r.scene_id} | Target: {r.target_category}\n")
                f.write(f"Result: {status} | Steps: {r.steps} | SPL: {r.spl:.4f} | Time: {r.episode_time:.1f}s\n")
                if r.failure_reason:
                    f.write(f"Failure Reason: {r.failure_reason}\n")
                f.write("-" * 80 + "\n\n")
                
                # Write final LLM interaction
                if r.final_interaction:
                    interaction = r.final_interaction
                    
                    # System Prompt
                    f.write(">>> SYSTEM PROMPT:\n")
                    f.write("-" * 40 + "\n")
                    system_prompt = interaction.get("system_prompt", "N/A")
                    f.write(f"{system_prompt}\n\n")
                    
                    # User Prompt
                    f.write(">>> USER PROMPT:\n")
                    f.write("-" * 40 + "\n")
                    user_prompt = interaction.get("user_prompt", "N/A")
                    f.write(f"{user_prompt}\n\n")
                    
                    # LLM Response
                    f.write(">>> LLM RESPONSE:\n")
                    f.write("-" * 40 + "\n")
                    response = interaction.get("response", "N/A")
                    f.write(f"{response}\n\n")
                    
                    # Parsed Action
                    f.write(">>> PARSED ACTION:\n")
                    f.write("-" * 40 + "\n")
                    action = interaction.get("action", "N/A")
                    feedback = interaction.get("feedback", "")
                    f.write(f"Action: {action}\n")
                    if feedback:
                        f.write(f"Feedback: {feedback}\n")
                    f.write(f"Episode Success: {status}\n")
                    f.write("\n")
                else:
                    f.write("(No LLM interaction recorded - possibly random agent)\n\n")
            
            # Footer
            f.write("=" * 80 + "\n")
            f.write("END OF LOG\n")
            f.write("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Habitat ObjectNav Inference Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with low-level actions (default, single planner)
    python -m src.habitat_nav.inference --scene 00800-TEEsavR23oF --episodes 5
    
    # Run with high-level actions (goto, explore, etc.)
    python -m src.habitat_nav.inference --action-level high --scene 00800-TEEsavR23oF
    
    # Run with OpenAI GPT-4o
    python -m src.habitat_nav.inference --agent llm-openai --model gpt-4o
    
    # Run with two-planner (main planner + narrator)
    python -m src.habitat_nav.inference --planner two --action-level high
    
    # Two-planner with custom narrator model
    python -m src.habitat_nav.inference --planner two --narrator-model gpt-4o-mini --action-level high
    
    # Run with image planner (sends images to GPT-4o Vision)
    python -m src.habitat_nav.inference --planner image --action-level high
    
    # Image planner with custom settings
    python -m src.habitat_nav.inference --planner image --max-images 3 --image-detail low --action-level high
    
    # Run random baseline
    python -m src.habitat_nav.inference --agent random --episodes 10
    
    # Use custom config
    python -m src.habitat_nav.inference --config configs/habitat_nav.yaml
        """
    )
    
    # Scene settings
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--scene", type=str, help="Single scene ID to run")
    parser.add_argument("--scenes", type=str, help="Comma-separated scene IDs")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per scene")
    
    # Agent settings
    parser.add_argument(
        "--agent", 
        choices=["llm", "llm-openai", "llm-custom", "random"], 
        default="llm", 
        help="Agent type: llm (auto), llm-openai (OpenAI API), llm-custom (custom API), random"
    )
    parser.add_argument(
        "--api-type",
        choices=["openai", "custom"],
        help="LLM API type (overrides --agent detection)"
    )
    parser.add_argument("--api-url", type=str, help="Custom LLM API URL")
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4o",
        help="LLM model name (e.g., gpt-4o, gpt-4-turbo, gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--openai-key",
        type=str,
        help="OpenAI API key (defaults to OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM sampling temperature (0.0-2.0)"
    )
    parser.add_argument(
        "--action-level",
        choices=["low", "high"],
        default="low",
        help="Action level: 'low' for discrete actions (MOVE_FORWARD, etc.), 'high' for semantic actions (goto, explore, etc.)"
    )
    parser.add_argument(
        "--planner",
        choices=["single", "two", "image"],
        default="single",
        help="Planner type: 'single' for one LLM, 'two' for main planner + narrator, 'image' for vision-based"
    )
    parser.add_argument(
        "--narrator-model",
        type=str,
        default="gpt-4o-mini",
        help="Model for narrator LLM (two-planner only, default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--narrator-temperature",
        type=float,
        default=0.7,
        help="Narrator LLM temperature (two-planner only)"
    )
    parser.add_argument(
        "--narrator-update-freq",
        type=int,
        default=3,
        help="Generate narrator story every N actions (two-planner only)"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=3,
        help="Maximum images to send to LLM (image planner only, default: 3)"
    )
    parser.add_argument(
        "--image-detail",
        choices=["low", "high", "auto"],
        default="low",
        help="Image detail level for OpenAI Vision (image planner only, default: low)"
    )
    
    # Output settings
    parser.add_argument("--output", type=str, default="./outputs/habitat_nav", help="Output directory")
    parser.add_argument("--no-video", action="store_true", help="Disable video saving")
    parser.add_argument("--no-annotations", action="store_true", 
                       help="Disable video annotations (bounding boxes, labels)")
    
    # Navigation settings
    parser.add_argument(
        "--collect-images",
        action="store_true",
        help="Collect images during actions for LLM context"
    )
    parser.add_argument(
        "--collision-distance",
        type=float,
        default=0.4,
        help="Safe distance from obstacles in meters (default: 0.4)"
    )
    
    # Misc
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config
    if args.config:
        config = InferenceConfig.from_yaml(args.config)
    else:
        config = InferenceConfig.from_args(args)
    
    if config.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run inference
    runner = InferenceRunner(config)
    summary = runner.run()
    
    return 0 if summary["success_rate"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

