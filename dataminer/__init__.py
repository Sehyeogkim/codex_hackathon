"""DataMiner: RGB human video to validated Franka training data.

The package exposes one high-level function, :func:`run_video_job`.  Individual
stages remain importable for testing and for the four Reflex worker roles.
"""

from .pipeline import JobDependencies, run_job as run_video_job

__all__ = ["JobDependencies", "run_video_job"]
