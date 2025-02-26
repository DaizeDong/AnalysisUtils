import os

# analysis information
ANALYSIS_TYPE = os.environ.get("ANALYSIS_TYPE")
if ANALYSIS_TYPE is not None:
    ANALYSIS_TYPE = ANALYSIS_TYPE.split(",")
print("ANALYSIS_TYPE:", ANALYSIS_TYPE)

ANALYSIS_SAVE_DIR = os.environ.get("ANALYSIS_SAVE_DIR")
print("ANALYSIS_SAVE_DIR:", ANALYSIS_SAVE_DIR)

OVERWRITE_ANALYSIS_DATA = os.environ.get("OVERWRITE_ANALYSIS_DATA", "0") == "1"
print("OVERWRITE_ANALYSIS_DATA:", OVERWRITE_ANALYSIS_DATA)

ANALYSIS_ENABLED = ANALYSIS_TYPE is not None and ANALYSIS_SAVE_DIR is not None
print("ANALYSIS_ENABLED:", ANALYSIS_ENABLED)


def disable_analysis():
    global ANALYSIS_ENABLED
    ANALYSIS_ENABLED = False


def enable_analysis():
    global ANALYSIS_ENABLED
    ANALYSIS_ENABLED = True


def reset_analysis_env():
    global ANALYSIS_ENABLED
    ANALYSIS_ENABLED = ANALYSIS_TYPE is not None and ANALYSIS_SAVE_DIR is not None
