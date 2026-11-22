"""TwinGuard Evaluation Module."""
from eval.metrics import compute_scenario_metrics
from eval.runner import run_evaluation_suite
from eval.plotter import (
    plot_detection_rate_vs_drift,
    plot_time_to_detect,
    plot_example_trajectories,
)

__all__ = [
    "compute_scenario_metrics",
    "run_evaluation_suite",
    "plot_detection_rate_vs_drift",
    "plot_time_to_detect",
    "plot_example_trajectories",
]
