"""Reduce noisy third-party log output during analysis."""

import os


def quiet_third_party_logs() -> None:
    """Silence verbose MediaPipe / TensorFlow console output."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("GLOG_minloglevel", "2")
    try:
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)
    except ImportError:
        pass
