import contextlib
import os

from .basic_utils.operations.operation_string import str2dict

# system environments
PID = os.getpid()

# analysis environments
ALL_ANALYSIS_ENVS = {}

ANALYSIS_TYPE = os.environ.get("ANALYSIS_TYPE")
if ANALYSIS_TYPE is not None:
    ALL_ANALYSIS_ENVS["ANALYSIS_TYPE"] = ANALYSIS_TYPE
    ANALYSIS_TYPE = ANALYSIS_TYPE.split(",")

ANALYSIS_SAVE_DIR = os.environ.get("ANALYSIS_SAVE_DIR")
if ANALYSIS_SAVE_DIR is not None:
    ALL_ANALYSIS_ENVS["ANALYSIS_SAVE_DIR"] = ANALYSIS_SAVE_DIR

OVERWRITE_ANALYSIS_DATA = (os.environ.get("OVERWRITE_ANALYSIS_DATA", "0") == "1")
if OVERWRITE_ANALYSIS_DATA:
    ALL_ANALYSIS_ENVS["OVERWRITE_ANALYSIS_DATA"] = str(int(OVERWRITE_ANALYSIS_DATA))

print(f"[{PID}] ALL_ANALYSIS_ENVS: {ALL_ANALYSIS_ENVS}")

# analysis args
ANALYSIS_ARGS = {}

if os.environ.get("ANALYSIS_ARGS") is not None:
    ANALYSIS_ARGS = str2dict(os.environ.get("ANALYSIS_ARGS"))

print(f"[{PID}] ANALYSIS_ARGS: {ANALYSIS_ARGS}")

# debug mode
ANALYSIS_DEBUG = (os.environ.get("ANALYSIS_DEBUG", "0") == "1")
print(f"[{PID}] ANALYSIS_DEBUG: {ANALYSIS_DEBUG}")

# analysis status flag
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
