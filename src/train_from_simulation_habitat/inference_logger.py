"""
Inference Logger for SmallPlan

This module provides real-time logging capabilities for inference runs.
It tracks and logs:
1. Current success rate (from video directory counts)
2. Average run time per task
3. Average run time for successful tasks

Usage:
    from src.train_from_simulation_habitat.inference_logger import InferenceLogger
    
    logger = InferenceLogger(video_dir="/path/to/videos", run_name="my_run")
    
    # At start of episode
    logger.start_episode(scene_id, episode_num, task_description)
    
    # At end of episode
    logger.end_episode(task_success=True, episode_info={...})
    
    # At end of run
    logger.finalize()
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class EpisodeRecord:
    """Record for a single episode."""
    scene_id: str
    episode_num: int
    task_description: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    task_success: Optional[bool] = None
    failure_reason: Optional[str] = None
    extra_info: Dict[str, Any] = field(default_factory=dict)


class InferenceLogger:
    """
    Logger for tracking inference run statistics in real-time.
    
    Creates and maintains a log.txt file with:
    - Success/failure/error counts
    - Success rate
    - Average task duration
    - Average successful task duration
    - Per-episode details
    """
    
    def __init__(self, 
                 video_dir: str,
                 run_name: str = "inference",
                 log_filename: str = "log.txt",
                 agent_type: str = "unknown"):
        """
        Initialize the inference logger.
        
        Args:
            video_dir: Directory where videos are saved (with success/failed/error subdirs)
            run_name: Name of this inference run
            log_filename: Name of the log file
            agent_type: Type of agent (e.g., "llm", "multi-llm", "random", "greedy")
        """
        self.video_dir = video_dir
        self.run_name = run_name
        self.agent_type = agent_type
        self.log_filepath = os.path.join(video_dir, log_filename) if video_dir else None
        
        # Episode tracking
        self.episodes: List[EpisodeRecord] = []
        self.current_episode: Optional[EpisodeRecord] = None
        
        # Timing
        self.run_start_time = time.time()
        
        # Counters (in addition to video directory counts)
        self.success_count = 0
        self.failed_count = 0
        self.error_count = 0
        
        # Initialize log file
        if self.log_filepath:
            self._init_log_file()
    
    def _init_log_file(self):
        """Initialize the log file with header information."""
        os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)
        
        with open(self.log_filepath, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(f"INFERENCE LOG - {self.run_name}\n")
            f.write("=" * 70 + "\n")
            f.write(f"Agent Type: {self.agent_type}\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Video Directory: {self.video_dir}\n")
            f.write("=" * 70 + "\n\n")
    
    def _count_videos_in_dir(self, subdir: str) -> int:
        """Count video files in a subdirectory."""
        dir_path = os.path.join(self.video_dir, subdir)
        if not os.path.exists(dir_path):
            return 0
        
        count = 0
        for f in os.listdir(dir_path):
            if f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                count += 1
        return count
    
    def get_video_counts(self) -> Dict[str, int]:
        """Get current video counts from directory structure."""
        if not self.video_dir or not os.path.exists(self.video_dir):
            return {
                'success': self.success_count,
                'failed': self.failed_count,
                'error': self.error_count,
                'total': self.success_count + self.failed_count + self.error_count
            }
        
        success = self._count_videos_in_dir('success')
        failed = self._count_videos_in_dir('failed')
        error = self._count_videos_in_dir('error')
        
        return {
            'success': success,
            'failed': failed,
            'error': error,
            'total': success + failed + error
        }
    
    def get_success_rate(self) -> float:
        """Calculate current success rate."""
        counts = self.get_video_counts()
        completed = counts['success'] + counts['failed']
        if completed == 0:
            return 0.0
        return counts['success'] / completed
    
    def get_average_duration(self) -> float:
        """Calculate average duration for all completed episodes."""
        completed_episodes = [e for e in self.episodes if e.duration is not None]
        if not completed_episodes:
            return 0.0
        return sum(e.duration for e in completed_episodes) / len(completed_episodes)
    
    def get_average_success_duration(self) -> float:
        """Calculate average duration for successful episodes."""
        successful_episodes = [e for e in self.episodes 
                              if e.duration is not None and e.task_success]
        if not successful_episodes:
            return 0.0
        return sum(e.duration for e in successful_episodes) / len(successful_episodes)
    
    def get_average_failed_duration(self) -> float:
        """Calculate average duration for failed episodes."""
        failed_episodes = [e for e in self.episodes 
                         if e.duration is not None and not e.task_success]
        if not failed_episodes:
            return 0.0
        return sum(e.duration for e in failed_episodes) / len(failed_episodes)
    
    def start_episode(self, 
                      scene_id: str, 
                      episode_num: int, 
                      task_description: str = ""):
        """
        Mark the start of a new episode.
        
        Args:
            scene_id: Scene identifier
            episode_num: Episode number within the scene
            task_description: Description of the task
        """
        self.current_episode = EpisodeRecord(
            scene_id=scene_id,
            episode_num=episode_num,
            task_description=task_description,
            start_time=time.time()
        )
        
        # Log episode start
        if self.log_filepath:
            self._append_to_log(
                f"\n[EPISODE START] Scene: {scene_id}, Episode: {episode_num}\n"
                f"  Task: {task_description}\n"
                f"  Time: {datetime.now().strftime('%H:%M:%S')}\n"
            )
    
    def end_episode(self, 
                    task_success: bool, 
                    episode_info: Optional[Dict] = None,
                    failure_reason: Optional[str] = None):
        """
        Mark the end of the current episode and update statistics.
        
        Args:
            task_success: Whether the task was successful
            episode_info: Additional episode information
            failure_reason: Reason for failure (if applicable)
        """
        if self.current_episode is None:
            logger.warning("end_episode called without start_episode")
            return
        
        # Complete the episode record
        self.current_episode.end_time = time.time()
        self.current_episode.duration = (
            self.current_episode.end_time - self.current_episode.start_time
        )
        self.current_episode.task_success = task_success
        self.current_episode.failure_reason = failure_reason
        
        if episode_info:
            self.current_episode.extra_info = episode_info
        
        # Update counters
        if task_success:
            self.success_count += 1
        else:
            self.failed_count += 1
        
        # Add to episodes list
        self.episodes.append(self.current_episode)
        
        # Update log file
        self._update_log()
        
        # Clear current episode
        self.current_episode = None
    
    def record_error(self, error_message: str = ""):
        """
        Record an error (interrupted/crashed episode).
        
        Args:
            error_message: Description of the error
        """
        self.error_count += 1
        
        if self.log_filepath:
            self._append_to_log(
                f"\n[ERROR] {datetime.now().strftime('%H:%M:%S')}\n"
                f"  {error_message}\n"
            )
        
        self._update_log()
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in a human-readable way."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.1f}s"
    
    def _append_to_log(self, text: str):
        """Append text to the log file."""
        if not self.log_filepath:
            return
        
        try:
            with open(self.log_filepath, 'a') as f:
                f.write(text)
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")
    
    def _update_log(self):
        """Update the log file with current statistics."""
        if not self.log_filepath:
            return
        
        # Get current stats
        counts = self.get_video_counts()
        success_rate = self.get_success_rate()
        avg_duration = self.get_average_duration()
        avg_success_duration = self.get_average_success_duration()
        avg_failed_duration = self.get_average_failed_duration()
        elapsed_time = time.time() - self.run_start_time
        
        # Get last episode info
        last_episode = self.episodes[-1] if self.episodes else None
        
        # Build status update
        status_lines = [
            "\n" + "-" * 50 + "\n",
            f"[STATUS UPDATE] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "-" * 50 + "\n",
        ]
        
        # Last episode summary
        if last_episode:
            outcome = "SUCCESS ✓" if last_episode.task_success else "FAILED ✗"
            status_lines.append(
                f"Last Episode: {last_episode.scene_id} ep{last_episode.episode_num} - {outcome}\n"
            )
            status_lines.append(
                f"  Duration: {self._format_duration(last_episode.duration or 0)}\n"
            )
            if last_episode.failure_reason:
                status_lines.append(f"  Reason: {last_episode.failure_reason}\n")
        
        status_lines.append("\n")
        
        # Current statistics
        status_lines.extend([
            "=== CURRENT STATISTICS ===\n",
            f"Total Episodes: {counts['total']}\n",
            f"  - Success: {counts['success']}\n",
            f"  - Failed:  {counts['failed']}\n",
            f"  - Error:   {counts['error']}\n",
            f"\n",
            f"Success Rate: {success_rate:.1%} ({counts['success']}/{counts['success'] + counts['failed']})\n",
            f"\n",
            f"Average Duration (all):     {self._format_duration(avg_duration)}\n",
            f"Average Duration (success): {self._format_duration(avg_success_duration)}\n",
            f"Average Duration (failed):  {self._format_duration(avg_failed_duration)}\n",
            f"\n",
            f"Total Elapsed Time: {self._format_duration(elapsed_time)}\n",
            "-" * 50 + "\n",
        ])
        
        self._append_to_log("".join(status_lines))
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all statistics.
        
        Returns:
            Dictionary with all statistics
        """
        counts = self.get_video_counts()
        elapsed_time = time.time() - self.run_start_time
        
        return {
            'total_episodes': counts['total'],
            'success_count': counts['success'],
            'failed_count': counts['failed'],
            'error_count': counts['error'],
            'success_rate': self.get_success_rate(),
            'average_duration': self.get_average_duration(),
            'average_success_duration': self.get_average_success_duration(),
            'average_failed_duration': self.get_average_failed_duration(),
            'total_elapsed_time': elapsed_time,
        }
    
    def finalize(self):
        """
        Finalize the log with final statistics.
        Call this at the end of the inference run.
        """
        if not self.log_filepath:
            return
        
        summary = self.get_summary()
        elapsed_time = summary['total_elapsed_time']
        
        final_lines = [
            "\n\n" + "=" * 70 + "\n",
            "FINAL RESULTS\n",
            "=" * 70 + "\n",
            f"Run Name: {self.run_name}\n",
            f"Agent Type: {self.agent_type}\n",
            f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"Total Runtime: {self._format_duration(elapsed_time)}\n",
            "\n",
            "=== EPISODE COUNTS ===\n",
            f"Total: {summary['total_episodes']}\n",
            f"Success: {summary['success_count']}\n",
            f"Failed: {summary['failed_count']}\n",
            f"Error/Interrupted: {summary['error_count']}\n",
            "\n",
            "=== SUCCESS RATE ===\n",
            f"{summary['success_rate']:.2%} ({summary['success_count']}/{summary['success_count'] + summary['failed_count']})\n",
            "\n",
            "=== TIMING ===\n",
            f"Average Task Duration: {self._format_duration(summary['average_duration'])}\n",
            f"Avg Successful Task:   {self._format_duration(summary['average_success_duration'])}\n",
            f"Avg Failed Task:       {self._format_duration(summary['average_failed_duration'])}\n",
            "\n",
        ]
        
        # Episode breakdown by scene
        if self.episodes:
            scenes = {}
            for ep in self.episodes:
                if ep.scene_id not in scenes:
                    scenes[ep.scene_id] = {'success': 0, 'failed': 0, 'durations': []}
                if ep.task_success:
                    scenes[ep.scene_id]['success'] += 1
                else:
                    scenes[ep.scene_id]['failed'] += 1
                if ep.duration:
                    scenes[ep.scene_id]['durations'].append(ep.duration)
            
            final_lines.append("=== PER-SCENE BREAKDOWN ===\n")
            for scene_id, data in sorted(scenes.items()):
                total = data['success'] + data['failed']
                rate = data['success'] / total if total > 0 else 0
                avg_dur = sum(data['durations']) / len(data['durations']) if data['durations'] else 0
                final_lines.append(
                    f"{scene_id}: {data['success']}/{total} ({rate:.0%}), "
                    f"avg {self._format_duration(avg_dur)}\n"
                )
        
        final_lines.append("\n" + "=" * 70 + "\n")
        
        self._append_to_log("".join(final_lines))
        
        logger.info(f"Inference log saved to: {self.log_filepath}")
    
    def print_current_stats(self):
        """Print current statistics to console."""
        summary = self.get_summary()
        
        print("\n" + "=" * 50)
        print("CURRENT INFERENCE STATISTICS")
        print("=" * 50)
        print(f"Total Episodes: {summary['total_episodes']}")
        print(f"  Success: {summary['success_count']}")
        print(f"  Failed:  {summary['failed_count']}")
        print(f"  Error:   {summary['error_count']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Avg Duration: {self._format_duration(summary['average_duration'])}")
        print(f"Avg Success:  {self._format_duration(summary['average_success_duration'])}")
        print("=" * 50 + "\n")


# Convenience function for creating logger
def create_inference_logger(
    video_dir: Optional[str],
    run_name: str,
    agent_type: str,
    enabled: bool = True
) -> Optional[InferenceLogger]:
    """
    Create an inference logger if video saving is enabled.
    
    Args:
        video_dir: Video directory path (or None if not saving videos)
        run_name: Name for this run
        agent_type: Type of agent
        enabled: Whether logging is enabled
        
    Returns:
        InferenceLogger instance or None if disabled
    """
    if not enabled or not video_dir:
        return None
    
    return InferenceLogger(
        video_dir=video_dir,
        run_name=run_name,
        agent_type=agent_type
    )

