from collections import OrderedDict

from .analysis_cache import *
from .analysis_env import *
from .basic_utils.io import create_dir, save_json

# system variables
try:
    if "ENVIRON_SAVE_DIR" in os.environ:
        save_dir = os.path.join(os.environ['ENVIRON_SAVE_DIR'], "environs")
        save_file = os.path.join(save_dir, f"{PID}.json")
        delete_file_or_dir(save_dir, suppress_errors=True)  # remove old results
        create_dir(save_dir)
        save_json(
            OrderedDict({key: os.environ[key] for key in sorted(os.environ)}),
            save_file,
            indent=4,
        )
        print(f'[{PID}] Saved system variable to {save_file}')
    else:
        print(f'[{PID}] ENVIRON_SAVE_DIR not set, skip saving system variables.')

except Exception as e:
    print(f"[{PID}] Save ENVIRON failed: {e}")
