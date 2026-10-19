"""
Logging utilities for training, including Weights & Biases integration.
"""

import os
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBLogger:
    """
    Weights & Biases logger for training metrics.

    Features:
    - Log scalar metrics (loss, accuracy, etc.)
    - Log model gradients (mean, std, min, max per layer)
    - Log learning rate
    - Watch model parameters
    - Conditional enabling (no-op if disabled)
    """

    def __init__(
        self,
        project: str = "wanvideo-training",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        entity: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        resume: Optional[str] = None,
        dir: Optional[str] = None,
    ):
        """
        Initialize WandB logger.

        Args:
            project: WandB project name
            name: Run name (auto-generated if None)
            config: Configuration dictionary to log
            enabled: Whether to enable logging
            entity: WandB entity (username or team)
            tags: List of tags for the run
            notes: Notes about the run
            resume: Resume mode ('allow', 'must', 'never', or None)
            dir: Directory to store wandb run files (defaults to ./wandb if None)
        """
        self.enabled = enabled and WANDB_AVAILABLE
        self.dir = dir

        if not WANDB_AVAILABLE and enabled:
            print("⚠️  WandB not installed. Install with: pip install wandb")
            print("    Logging will be disabled.")

        if self.enabled:
            try:
                if dir is not None:
                    os.makedirs(dir, exist_ok=True)
                self.run = wandb.init(
                    project=project,
                    name=name,
                    config=config,
                    entity=entity,
                    tags=tags,
                    notes=notes,
                    resume=resume,
                    dir=dir,
                )
                print(f"✓ WandB initialized: {wandb.run.url}")
            except Exception as e:
                print(f"⚠️  Failed to initialize WandB: {e}")
                self.enabled = False
        else:
            self.run = None

    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None, commit: bool = True):
        """
        Log scalar metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Global step number (uses wandb's internal counter if None)
            commit: Whether to commit the metrics immediately
        """
        if self.enabled:
            wandb.log(metrics, step=step, commit=commit)

    def log_gradients(
        self,
        model: nn.Module,
        step: Optional[int] = None,
        prefix: str = "gradients",
        log_histogram: bool = False
    ):
        """
        Log model gradients.

        Args:
            model: PyTorch model
            step: Global step number
            prefix: Prefix for metric names
            log_histogram: Whether to log gradient histograms (expensive)
        """
        if not self.enabled:
            return

        grad_metrics = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()

                # Log statistics
                grad_metrics[f"{prefix}/{name}/mean"] = grad.mean().item()
                grad_metrics[f"{prefix}/{name}/std"] = grad.std().item()
                grad_metrics[f"{prefix}/{name}/max"] = grad.max().item()
                grad_metrics[f"{prefix}/{name}/min"] = grad.min().item()
                grad_metrics[f"{prefix}/{name}/norm"] = grad.norm().item()

                # Optional: log histogram (expensive)
                if log_histogram:
                    grad_metrics[f"{prefix}/{name}/histogram"] = wandb.Histogram(grad.cpu().numpy())

        if grad_metrics:
            wandb.log(grad_metrics, step=step)

    def log_learning_rate(self, lr: Union[float, Dict[str, float]], step: Optional[int] = None):
        """
        Log learning rate.

        Args:
            lr: Learning rate (single value or dict for multiple param groups)
            step: Global step number
        """
        if not self.enabled:
            return

        if isinstance(lr, dict):
            wandb.log(lr, step=step)
        else:
            wandb.log({"learning_rate": lr}, step=step)

    def log_model_parameters(
        self,
        model: nn.Module,
        step: Optional[int] = None,
        prefix: str = "parameters",
        log_histogram: bool = False
    ):
        """
        Log model parameters statistics.

        Args:
            model: PyTorch model
            step: Global step number
            prefix: Prefix for metric names
            log_histogram: Whether to log parameter histograms
        """
        if not self.enabled:
            return

        param_metrics = {}

        for name, param in model.named_parameters():
            p = param.detach()

            # Log statistics
            param_metrics[f"{prefix}/{name}/mean"] = p.mean().item()
            param_metrics[f"{prefix}/{name}/std"] = p.std().item()
            param_metrics[f"{prefix}/{name}/max"] = p.max().item()
            param_metrics[f"{prefix}/{name}/min"] = p.min().item()
            param_metrics[f"{prefix}/{name}/norm"] = p.norm().item()

            # Optional: log histogram
            if log_histogram:
                param_metrics[f"{prefix}/{name}/histogram"] = wandb.Histogram(p.cpu().numpy())

        if param_metrics:
            wandb.log(param_metrics, step=step)

    def watch_model(
        self,
        model: nn.Module,
        log: str = "all",
        log_freq: int = 100,
        log_graph: bool = True
    ):
        """
        Watch model for gradients and parameters.

        Args:
            model: PyTorch model
            log: What to log ('gradients', 'parameters', 'all', or None)
            log_freq: Logging frequency (in steps)
            log_graph: Whether to log model graph
        """
        if self.enabled:
            wandb.watch(model, log=log, log_freq=log_freq, log_graph=log_graph)

    def log_images(self, images: Dict[str, Any], step: Optional[int] = None):
        """
        Log images.

        Args:
            images: Dictionary of image names and values
                   Values can be: PIL Image, numpy array, torch.Tensor, or wandb.Image
            step: Global step number
        """
        if self.enabled:
            wandb.log(images, step=step)

    def log_table(self, key: str, columns: list, data: list, step: Optional[int] = None):
        """
        Log tabular data.

        Args:
            key: Table name
            columns: List of column names
            data: List of rows (each row is a list of values)
            step: Global step number
        """
        if self.enabled:
            table = wandb.Table(columns=columns, data=data)
            wandb.log({key: table}, step=step)

    def log_artifact(self, artifact_path: str, artifact_type: str, name: Optional[str] = None):
        """
        Log an artifact (model checkpoint, dataset, etc.).

        Args:
            artifact_path: Path to the artifact
            artifact_type: Type of artifact ('model', 'dataset', etc.)
            name: Artifact name (defaults to filename)
        """
        if not self.enabled:
            return

        if name is None:
            name = os.path.basename(artifact_path)

        artifact = wandb.Artifact(name, type=artifact_type)
        artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)

    def log_code(self, root: str = ".", include_fn: Optional[callable] = None):
        """
        Log code files.

        Args:
            root: Root directory to log
            include_fn: Function to filter files (receives filename, returns bool)
        """
        if self.enabled:
            wandb.run.log_code(root=root, include_fn=include_fn)

    def finish(self):
        """Finish the WandB run."""
        if self.enabled and self.run is not None:
            wandb.finish()
            print("✓ WandB logging finished")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


class DummyLogger:
    """Dummy logger that does nothing (for when WandB is disabled)."""

    def __init__(self, *args, **kwargs):
        pass

    def log_metrics(self, *args, **kwargs):
        pass

    def log_gradients(self, *args, **kwargs):
        pass

    def log_learning_rate(self, *args, **kwargs):
        pass

    def log_model_parameters(self, *args, **kwargs):
        pass

    def watch_model(self, *args, **kwargs):
        pass

    def log_images(self, *args, **kwargs):
        pass

    def log_table(self, *args, **kwargs):
        pass

    def log_artifact(self, *args, **kwargs):
        pass

    def log_code(self, *args, **kwargs):
        pass

    def finish(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def create_logger(enabled: bool = True, **kwargs) -> Union[WandBLogger, DummyLogger]:
    """
    Factory function to create a logger.

    Args:
        enabled: Whether to enable logging
        **kwargs: Arguments to pass to WandBLogger

    Returns:
        WandBLogger if enabled and available, otherwise DummyLogger
    """
    if enabled:
        return WandBLogger(enabled=True, **kwargs)
    else:
        return DummyLogger()
