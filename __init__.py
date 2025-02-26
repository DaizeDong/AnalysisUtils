import os
import re
from collections import OrderedDict

from .analysis_cache import ANALYSIS_CACHE_DYNAMIC, ANALYSIS_CACHE_STATIC
from .analysis_env import ANALYSIS_ENABLED, ANALYSIS_SAVE_DIR, ANALYSIS_TYPE
from basic_utils.io import create_dir, save_json

# system variables
try:
    if "ENVIRON_SAVE_DIR" in os.environ:
        if "PMIX_NAMESPACE" in os.environ:
            if any(f"environ-{os.environ['PMIX_NAMESPACE']}" in string for string in os.listdir(os.path.join(os.environ['ENVIRON_SAVE_DIR'], os.environ['MLP_TASK_ID'] if "MLP_TASK_ID" in os.environ else "."))):  # use existing runid
                runid = [int(re.search(r'run-(\d+)-' + f"environ-{os.environ['PMIX_NAMESPACE']}", string).group(1)) for string in os.listdir(os.path.join(os.environ['ENVIRON_SAVE_DIR'], os.environ['MLP_TASK_ID'] if "MLP_TASK_ID" in os.environ else ".")) if re.search(r'run-(\d+)-' + f"environ-{os.environ['PMIX_NAMESPACE']}", string)][0]
            else:  # create a new runid according to existing maximum runids
                runid = str(max([-1] + [int(re.search(r'run-(\d+)-', string).group(1)) for string in os.listdir(os.path.join(os.environ['ENVIRON_SAVE_DIR'], os.environ['MLP_TASK_ID'] if "MLP_TASK_ID" in os.environ else ".")) if re.search(r"run-(\d+)-", string)]) + 1)
        else:  # create a new runid according to existing maximum runids
            runid = str(max([-1] + [int(re.search(r'run-(\d+)-', string).group(1)) for string in os.listdir(os.path.join(os.environ['ENVIRON_SAVE_DIR'], os.environ['MLP_TASK_ID'] if "MLP_TASK_ID" in os.environ else ".")) if re.search(r"run-(\d+)-", string)]) + 1)

        save_dir = os.path.join(
            os.environ['ENVIRON_SAVE_DIR'],
            os.environ['MLP_TASK_ID'] if "MLP_TASK_ID" in os.environ else ".",
            f"run-{runid}-environ-{os.environ['PMIX_NAMESPACE']}" if "PMIX_NAMESPACE" in os.environ else f"run-{runid}-environ",
        )
        create_dir(save_dir)
        save_json(
            OrderedDict({key: os.environ[key] for key in sorted(os.environ)}),
            os.path.join(save_dir, f"{os.getpid()}.json"),
            indent=4,
        )
        print(f'Saved system variable to {os.path.join(save_dir, f"{os.getpid()}.json")}')

except Exception as e:
    print(e)
    print(f"Save ENVIRON failed.")
