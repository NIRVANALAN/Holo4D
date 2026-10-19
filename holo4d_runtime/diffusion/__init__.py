from .flow_match import FlowMatchScheduler

# Keep inference imports lightweight: training dependencies (e.g. `peft`) are optional for inference-only usage.
try:
    from .training_module import DiffusionTrainingModule  # noqa: F401
    from .logger import ModelLogger  # noqa: F401
    from .runner import launch_training_task, launch_data_process_task  # noqa: F401
except ModuleNotFoundError as e:
    # Allow importing `diffsynth.diffusion.parsers` (and other inference utilities) without training deps installed.
    if getattr(e, "name", None) in {"peft"}:
        DiffusionTrainingModule = None  # type: ignore[assignment]
        ModelLogger = None  # type: ignore[assignment]
        launch_training_task = None  # type: ignore[assignment]
        launch_data_process_task = None  # type: ignore[assignment]
    else:
        raise

from .parsers import *  # noqa: F403
from .loss import *  # noqa: F403
