"""Progress tracking and logging for training pipeline"""

import logging
import time
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
import threading
from tqdm import tqdm


@dataclass
class StepInfo:
    """Information about a training step"""
    name: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    message: str = ""
    progress: float = 0.0
    substeps: List['StepInfo'] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get step duration"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def duration_str(self) -> str:
        """Get formatted duration string"""
        if self.duration:
            return str(self.duration).split('.')[0]  # Remove microseconds
        return "N/A"


class ProgressTracker:
    """Track training progress with logging and optional progress bars"""
    
    def __init__(self, 
                 total_steps: int = 1,
                 verbose: bool = True,
                 use_progress_bar: bool = True,
                 log_file: Optional[str] = None,
                 experiment_name: str = "training"):
        """
        Initialize progress tracker
        
        Args:
            total_steps: Total number of major steps
            verbose: Enable verbose logging
            use_progress_bar: Show progress bars
            log_file: Optional log file path
            experiment_name: Name of the experiment
        """
        self.total_steps = total_steps
        self.verbose = verbose
        self.use_progress_bar = use_progress_bar
        self.experiment_name = experiment_name
        
        # Step tracking
        self.steps: List[StepInfo] = []
        self.current_step: Optional[StepInfo] = None
        self.current_step_index: int = 0
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        
        # Progress bars
        self.main_progress_bar: Optional[tqdm] = None
        self.step_progress_bar: Optional[tqdm] = None
        
        # Setup logging
        self._setup_logging(log_file)
        
        # Initialize main progress bar
        if self.use_progress_bar:
            self._setup_progress_bars()
            
        self.logger.info(f"Progress tracker initialized for '{experiment_name}' with {total_steps} steps")
    
    def _setup_logging(self, log_file: Optional[str] = None) -> None:
        """Setup logging configuration"""
        self.logger = logging.getLogger(f"ProgressTracker_{self.experiment_name}")
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # File handler if specified
            if log_file:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(log_file)
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
    
    def _setup_progress_bars(self) -> None:
        """Setup progress bars"""
        if not self.use_progress_bar:
            return
            
        # Main progress bar for overall progress
        self.main_progress_bar = tqdm(
            total=self.total_steps,
            desc="Training Pipeline",
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        # Step progress bar for current step progress
        self.step_progress_bar = tqdm(
            total=100,
            desc="Current Step",
            position=1,
            leave=False,
            bar_format='{l_bar}{bar}| {n_fmt}% [{elapsed}, {rate_fmt}]'
        )
    
    def start_step(self, step_name: str, estimated_duration: Optional[float] = None) -> None:
        """
        Start a new step
        
        Args:
            step_name: Name of the step
            estimated_duration: Estimated duration in seconds (optional)
        """
        # Complete previous step if it exists and is running
        if self.current_step and self.current_step.status == "running":
            self.complete_step("Step completed automatically")
        
        # Create new step
        step_info = StepInfo(name=step_name, start_time=datetime.now(), status="running")
        self.steps.append(step_info)
        self.current_step = step_info
        
        # Reset step progress
        if self.step_progress_bar:
            self.step_progress_bar.reset()
            self.step_progress_bar.set_description(step_name[:50])
        
        # Log step start
        self.logger.info(f"Starting step {self.current_step_index + 1}/{self.total_steps}: {step_name}")
        
        if estimated_duration:
            self.logger.info(f"Estimated duration: {estimated_duration:.1f} seconds")
    
    def complete_step(self, message: str = "Step completed") -> None:
        """
        Complete current step
        
        Args:
            message: Completion message
        """
        if not self.current_step:
            self.logger.warning("No current step to complete")
            return
        
        # Update step info
        self.current_step.end_time = datetime.now()
        self.current_step.status = "completed"
        self.current_step.message = message
        self.current_step.progress = 100.0
        
        # Update progress bars
        if self.main_progress_bar:
            self.main_progress_bar.update(1)
        
        if self.step_progress_bar:
            self.step_progress_bar.n = 100
            self.step_progress_bar.refresh()
        
        # Log completion
        duration_str = self.current_step.duration_str
        self.logger.info(f"Completed step {self.current_step_index + 1}/{self.total_steps}: "
                        f"{self.current_step.name} ({duration_str}) - {message}")
        
        # Move to next step
        self.current_step_index += 1
        self.current_step = None
    
    def fail_step(self, error_message: str) -> None:
        """
        Mark current step as failed
        
        Args:
            error_message: Error message
        """
        if not self.current_step:
            self.logger.error(f"Step failed: {error_message}")
            return
        
        # Update step info
        self.current_step.end_time = datetime.now()
        self.current_step.status = "failed"
        self.current_step.message = error_message
        
        # Log failure
        duration_str = self.current_step.duration_str
        self.logger.error(f"Failed step {self.current_step_index + 1}/{self.total_steps}: "
                         f"{self.current_step.name} ({duration_str}) - {error_message}")
        
        # Close progress bars
        self._close_progress_bars()
    
    def update_progress(self, progress: float, message: str = None) -> None:
        """
        Update progress within current step
        
        Args:
            progress: Progress percentage (0-100)
            message: Optional progress message
        """
        if not self.current_step:
            return
        
        # Clamp progress to valid range
        progress = max(0.0, min(100.0, progress))
        self.current_step.progress = progress
        
        if message:
            self.current_step.message = message
        
        # Update step progress bar
        if self.step_progress_bar:
            self.step_progress_bar.n = int(progress)
            if message:
                self.step_progress_bar.set_postfix_str(message[:30])
            self.step_progress_bar.refresh()
        
        # Log significant progress updates
        if self.verbose and (progress % 25 == 0 or message):
            log_msg = f"Step progress: {progress:.1f}%"
            if message:
                log_msg += f" - {message}"
            self.logger.info(log_msg)
    
    def add_substep(self, substep_name: str) -> None:
        """
        Add a substep to the current step
        
        Args:
            substep_name: Name of the substep
        """
        if not self.current_step:
            return
        
        substep = StepInfo(name=substep_name, start_time=datetime.now(), status="running")
        self.current_step.substeps.append(substep)
        
        if self.verbose:
            self.logger.info(f"  -> {substep_name}")
    
    def complete_substep(self, message: str = "Substep completed") -> None:
        """
        Complete the last substep
        
        Args:
            message: Completion message
        """
        if not self.current_step or not self.current_step.substeps:
            return
        
        last_substep = self.current_step.substeps[-1]
        if last_substep.status == "running":
            last_substep.end_time = datetime.now()
            last_substep.status = "completed"
            last_substep.message = message
            
            if self.verbose:
                self.logger.info(f"  -> {last_substep.name} completed ({last_substep.duration_str})")
    
    def set_total_steps(self, total_steps: int) -> None:
        """
        Update total steps count
        
        Args:
            total_steps: New total steps count
        """
        self.total_steps = total_steps
        
        if self.main_progress_bar:
            self.main_progress_bar.total = total_steps
            self.main_progress_bar.refresh()
    
    def finish(self, message: str = "Training completed") -> None:
        """
        Finish tracking and cleanup
        
        Args:
            message: Final completion message
        """
        # Complete any running step
        if self.current_step and self.current_step.status == "running":
            self.complete_step("Auto-completed")
        
        self.end_time = datetime.now()
        total_duration = self.end_time - self.start_time
        
        # Log final summary
        self.logger.info("="*60)
        self.logger.info(f"{message}")
        self.logger.info(f"Total duration: {str(total_duration).split('.')[0]}")
        self.logger.info(f"Completed {len([s for s in self.steps if s.status == 'completed'])}/{len(self.steps)} steps")
        self.logger.info("="*60)
        
        # Close progress bars
        self._close_progress_bars()
    
    def _close_progress_bars(self) -> None:
        """Close all progress bars"""
        if self.step_progress_bar:
            self.step_progress_bar.close()
            self.step_progress_bar = None
        
        if self.main_progress_bar:
            self.main_progress_bar.close()
            self.main_progress_bar = None
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get progress summary
        
        Returns:
            Progress summary dictionary
        """
        completed_steps = [s for s in self.steps if s.status == "completed"]
        failed_steps = [s for s in self.steps if s.status == "failed"]
        
        total_duration = None
        if self.end_time:
            total_duration = self.end_time - self.start_time
        elif self.current_step:
            total_duration = datetime.now() - self.start_time
        
        return {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration": str(total_duration).split('.')[0] if total_duration else None,
            "total_steps": self.total_steps,
            "completed_steps": len(completed_steps),
            "failed_steps": len(failed_steps),
            "current_step": self.current_step_index,
            "steps": [
                {
                    "name": step.name,
                    "status": step.status,
                    "duration": step.duration_str,
                    "message": step.message,
                    "progress": step.progress,
                    "substeps": len(step.substeps)
                }
                for step in self.steps
            ]
        }
    
    def print_summary(self) -> None:
        """Print a formatted summary of progress"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print(f"TRAINING PROGRESS SUMMARY - {summary['experiment_name']}")
        print("="*60)
        print(f"Start Time: {summary['start_time']}")
        if summary['end_time']:
            print(f"End Time: {summary['end_time']}")
        if summary['total_duration']:
            print(f"Total Duration: {summary['total_duration']}")
        print(f"Progress: {summary['completed_steps']}/{summary['total_steps']} steps completed")
        
        if summary['failed_steps'] > 0:
            print(f"Failed Steps: {summary['failed_steps']}")
        
        print("\nStep Details:")
        print("-" * 60)
        
        for i, step in enumerate(summary['steps'], 1):
            status_symbol = {
                'completed': '✓',
                'failed': '✗',
                'running': '→',
                'pending': '○'
            }.get(step['status'], '?')
            
            print(f"{i:2d}. {status_symbol} {step['name']:<30} [{step['duration']:>8}] {step['message']}")
            
            if step['substeps'] > 0:
                print(f"     └─ {step['substeps']} substeps")
        
        print("="*60)
    
    def create_callback(self, step_name: str) -> Callable[[float, str], None]:
        """
        Create a callback function for updating progress
        
        Args:
            step_name: Name of the step this callback is for
            
        Returns:
            Callback function that accepts (progress, message)
        """
        def callback(progress: float, message: str = ""):
            if self.current_step and self.current_step.name == step_name:
                self.update_progress(progress, message)
        
        return callback
    
    def context_manager(self, step_name: str):
        """
        Context manager for automatic step management
        
        Args:
            step_name: Name of the step
            
        Returns:
            Context manager
        """
        return _StepContextManager(self, step_name)


class _StepContextManager:
    """Context manager for automatic step lifecycle management"""
    
    def __init__(self, tracker: ProgressTracker, step_name: str):
        self.tracker = tracker
        self.step_name = step_name
        
    def __enter__(self):
        self.tracker.start_step(self.step_name)
        return self.tracker
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.tracker.complete_step("Step completed successfully")
        else:
            self.tracker.fail_step(f"Step failed: {exc_val}")
        return False  # Don't suppress exceptions


class MultiProcessProgressTracker:
    """Progress tracker that works across multiple processes"""
    
    def __init__(self, 
                 total_steps: int,
                 shared_state_file: str,
                 verbose: bool = True):
        """
        Initialize multi-process progress tracker
        
        Args:
            total_steps: Total number of steps
            shared_state_file: File to store shared progress state
            verbose: Enable verbose logging
        """
        self.total_steps = total_steps
        self.shared_state_file = Path(shared_state_file)
        self.verbose = verbose
        self.lock = threading.Lock()
        
        # Initialize shared state
        self._init_shared_state()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _init_shared_state(self) -> None:
        """Initialize shared state file"""
        import json
        
        initial_state = {
            "total_steps": self.total_steps,
            "current_step": 0,
            "steps": [],
            "start_time": datetime.now().isoformat()
        }
        
        self.shared_state_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.shared_state_file, 'w') as f:
            json.dump(initial_state, f, indent=2)
    
    def update_step(self, step_index: int, status: str, message: str = "") -> None:
        """
        Update step status in shared state
        
        Args:
            step_index: Index of the step
            status: Step status
            message: Optional message
        """
        import json
        
        with self.lock:
            # Read current state
            with open(self.shared_state_file, 'r') as f:
                state = json.load(f)
            
            # Update step
            while len(state["steps"]) <= step_index:
                state["steps"].append({
                    "status": "pending",
                    "message": "",
                    "timestamp": None
                })
            
            state["steps"][step_index] = {
                "status": status,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
            
            state["current_step"] = step_index
            
            # Write updated state
            with open(self.shared_state_file, 'w') as f:
                json.dump(state, f, indent=2)
        
        if self.verbose:
            self.logger.info(f"Step {step_index + 1}/{self.total_steps}: {status} - {message}")
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress state
        
        Returns:
            Current progress dictionary
        """
        import json
        
        try:
            with open(self.shared_state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error reading progress state: {e}")
            return {}