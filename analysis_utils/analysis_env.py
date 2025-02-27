import contextlib
import os

# environment variables
PID = os.getpid()

# analysis information
ANALYSIS_TYPE = os.environ.get("ANALYSIS_TYPE")
if ANALYSIS_TYPE is not None:
    ANALYSIS_TYPE = ANALYSIS_TYPE.split(",")
print(f"[{PID}] ANALYSIS_TYPE: {ANALYSIS_TYPE}")

ANALYSIS_SAVE_DIR = os.environ.get("ANALYSIS_SAVE_DIR")
print(f"[{PID}] ANALYSIS_SAVE_DIR: {ANALYSIS_SAVE_DIR}")

OVERWRITE_ANALYSIS_DATA = os.environ.get("OVERWRITE_ANALYSIS_DATA", "0") == "1"
print(f"[{PID}] OVERWRITE_ANALYSIS_DATA: {OVERWRITE_ANALYSIS_DATA}")

ANALYSIS_ENABLED = ANALYSIS_TYPE is not None and ANALYSIS_SAVE_DIR is not None
print(f"[{PID}] ANALYSIS_ENABLED: {ANALYSIS_ENABLED}")


@contextlib.contextmanager
def analysis_disabled():
    """
    with analysis_disabled():
        ...
    """
    global ANALYSIS_ENABLED
    old_value = ANALYSIS_ENABLED
    ANALYSIS_ENABLED = False
    try:
        yield
    finally:
        ANALYSIS_ENABLED = old_value


@contextlib.contextmanager
def analysis_enabled():
    """
    with analysis_enabled():
        ...
    """
    global ANALYSIS_ENABLED
    old_value = ANALYSIS_ENABLED
    ANALYSIS_ENABLED = True
    try:
        yield
    finally:
        ANALYSIS_ENABLED = old_value


def reset_analysis():
    global ANALYSIS_ENABLED
    ANALYSIS_ENABLED = ANALYSIS_TYPE is not None and ANALYSIS_SAVE_DIR is not None
