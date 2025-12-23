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
import logging
import os
import sys
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

from .core.simulator import HabitatSimulator, SimulatorConfig
from .core.environment import ObjectNavEnv
from .core.observations import ProcessedObservation
from .agents import BaseAgent, LLMAgent, RandomAgent
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
    num_episodes_per_scene: int = 5
    max_episode_steps: int = 500
    success_distance: float = 1.5
    
    # Agent settings
    agent_type: str = "llm"  # "llm", "llm-openai", "random"
    llm_api_url: str = "http://localhost:8000/generate"
    llm_api_type: str = "custom"  # "custom" or "openai"
    llm_model: str = "gpt-4o"  # Model name for LLM
    openai_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")
    llm_temperature: float = 0.7
    
    # Output settings
    output_dir: str = "./outputs/habitat_nav"
    save_videos: bool = True
    video_fps: int = 10
    
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
        if args.output:
            config.output_dir = args.output
        if args.no_video:
            config.save_videos = False
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
    failure_reason: Optional[str] = None
    video_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "target_category": self.target_category,
            "success": self.success,
            "spl": self.spl,
            "steps": self.steps,
            "distance_travelled": self.distance_travelled,
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
        """Create timestamped run directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(self.config.output_dir) / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (run_dir / "videos" / "success").mkdir(parents=True, exist_ok=True)
        (run_dir / "videos" / "failed").mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_dict = {k: v for k, v in self.config.__dict__.items()}
        with open(run_dir / "config.yaml", 'w') as f:
            yaml.dump(config_dict, f)
        
        return run_dir
    
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
            
            logger.info(f"Creating LLM agent: api_type={api_type}, model={self.config.llm_model}")
            
            return LLMAgent(
                api_type=api_type,
                api_url=self.config.llm_api_url,
                model_name=self.config.llm_model,
                openai_api_key=self.config.openai_api_key,
                temperature=self.config.llm_temperature,
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
            seed=self.config.seed
        )
        
        # Create agent
        agent = self._create_agent()
        
        try:
            for episode_idx in range(self.config.num_episodes_per_scene):
                result = self._run_episode(env, agent, episode_idx)
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
                     episode_idx: int) -> EpisodeResult:
        """Run a single episode."""
        # Reset environment and agent
        obs, info = env.reset(episode_id=episode_idx)
        target = info["target"]
        agent.reset(target_category=target)
        
        done = False
        failure_reason = None
        
        while not done:
            # Get action from agent
            action = agent.act(obs, env.get_task_description(), info)
            
            # Convert high-level action to low-level if needed
            if isinstance(action, HighLevelAction):
                low_level_action = self._execute_high_level_action(
                    env, action, obs
                )
            else:
                low_level_action = action
            
            # Step environment
            obs, reward, info = env.step(low_level_action)
            done = info["done"]
            
            # Update agent
            agent.update(action, obs, reward, info)
            
            # Check for timeout
            if info["step_count"] >= self.config.max_episode_steps:
                failure_reason = "timeout"
        
        # Compute metrics
        success = info["success"]
        spl = env.compute_spl()
        
        if not success and not failure_reason:
            failure_reason = "target_not_found"
        
        # Save video
        video_path = None
        if self.config.save_videos:
            subdir = "success" if success else "failed"
            video_name = f"{env.scene_id}_ep{episode_idx}_{target}.mp4"
            video_path = str(self.run_dir / "videos" / subdir / video_name)
            env.save_video(video_path, fps=self.config.video_fps)
        
        return EpisodeResult(
            scene_id=env.scene_id,
            episode_id=episode_idx,
            target_category=target,
            success=success,
            spl=spl,
            steps=info["step_count"],
            distance_travelled=info["distance_travelled"],
            failure_reason=failure_reason,
            video_path=video_path,
        )
    
    def _execute_high_level_action(self,
                                   env: ObjectNavEnv,
                                   action: HighLevelAction,
                                   obs: ProcessedObservation) -> Action:
        """
        Convert high-level action to low-level action.
        
        This is a simplified mapping - a full implementation would include
        path planning and navigation controllers.
        """
        if action.name == "stop":
            return Action.STOP
        
        # For simplicity, just move forward or turn randomly
        # A full implementation would do proper path planning
        if action.name in ("goto", "explore"):
            # Check if obstacle ahead
            if obs.depth is not None:
                center_depth = obs.depth[
                    obs.depth.shape[0] // 2,
                    obs.depth.shape[1] // 2
                ]
                if center_depth < 0.5:  # Obstacle close
                    return Action.TURN_LEFT if self.rng.random() > 0.5 else Action.TURN_RIGHT
            return Action.MOVE_FORWARD
        
        if action.name == "open":
            # Move towards the door/object
            return Action.MOVE_FORWARD
        
        return Action.MOVE_FORWARD
    
    def _compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        if not self.results:
            return {"success_rate": 0.0, "avg_spl": 0.0}
        
        successes = [r.success for r in self.results]
        spls = [r.spl for r in self.results]
        steps = [r.steps for r in self.results]
        
        return {
            "total_episodes": len(self.results),
            "success_rate": np.mean(successes),
            "avg_spl": np.mean(spls),
            "avg_steps": np.mean(steps),
            "std_steps": np.std(steps),
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
        
        # Save human-readable log
        with open(self.run_dir / "log.txt", 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"Habitat ObjectNav Inference Results\n")
            f.write(f"Run: {self.run_dir.name}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            for key, value in summary.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            
            f.write("\n\nPER-EPISODE RESULTS\n")
            f.write("-" * 40 + "\n")
            for r in self.results:
                status = "SUCCESS" if r.success else "FAILED"
                f.write(f"  {r.scene_id} ep{r.episode_id}: {status} | "
                       f"target={r.target_category} | steps={r.steps} | "
                       f"SPL={r.spl:.3f}\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Habitat ObjectNav Inference Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with custom LLM API
    python -m src.habitat_nav.inference --scene 00800-TEEsavR23oF --episodes 5
    
    # Run with OpenAI GPT-4o
    python -m src.habitat_nav.inference --agent llm-openai --model gpt-4o
    
    # Run with OpenAI GPT-3.5-turbo (cheaper)
    python -m src.habitat_nav.inference --agent llm-openai --model gpt-3.5-turbo
    
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
    
    # Output settings
    parser.add_argument("--output", type=str, default="./outputs/habitat_nav", help="Output directory")
    parser.add_argument("--no-video", action="store_true", help="Disable video saving")
    
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

