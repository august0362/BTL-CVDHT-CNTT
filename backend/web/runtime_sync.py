"""Synchronize shared model handles, forward hooks and process-wide CPU settings."""
from threading import RLock
from functools import wraps

# UI tasks run independently. Only stages that use shared model/process state
# take this lock; it is never held while waiting for a page or a widget rerun.
MODEL_RUNTIME_LOCK = RLock()


def model_stage(function):
    @wraps(function)
    def synchronized(*args, **kwargs):
        with MODEL_RUNTIME_LOCK:
            return function(*args, **kwargs)
    return synchronized
