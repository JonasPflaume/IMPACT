"""Path helpers shared by the experiment trajectory visualizers."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def task_results_dir(task):
    """Return the task's repository-local results directory, creating it if needed."""
    result_dir = REPO_ROOT / "results" / task
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def resolve_trajectory(task, supplied_path=None):
    """Resolve an explicit trajectory or find the newest one for ``task``."""
    if supplied_path:
        path = Path(supplied_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Trajectory file does not exist: {path}")
        return path

    result_dir = task_results_dir(task)
    candidates = [path for path in result_dir.rglob("trajectory_*.txt") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No trajectory_*.txt files found under {result_dir}. "
            "Run an experiment first or pass a trajectory path explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def resolve_output_dir(task, supplied_path=None):
    """Resolve an output directory, defaulting to ``results/<task>`` in this repo."""
    output_dir = (
        Path(supplied_path).expanduser().resolve()
        if supplied_path
        else task_results_dir(task)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
