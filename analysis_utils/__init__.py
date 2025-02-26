import re
from collections import OrderedDict

from .analysis_cache import *
from .analysis_env import *
from .basic_utils.io import create_dir, save_json

# system variables
try:
    if "ENVIRON_SAVE_DIR" in os.environ:
        save_dir = os.environ['ENVIRON_SAVE_DIR']
        save_file = os.path.join(save_dir, f"{os.getpid()}.json")
        create_dir(save_dir)
        save_json(
            OrderedDict({key: os.environ[key] for key in sorted(os.environ)}),
            save_file,
            indent=4,
        )
        print(f'Saved system variable to {save_file}')

except Exception as e:
    print(e)
    print(f"Save ENVIRON failed.")
