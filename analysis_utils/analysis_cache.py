import os
from datetime import datetime
from pickle import HIGHEST_PROTOCOL

import torch

from .analysis_env import ANALYSIS_ENABLED, ANALYSIS_SAVE_DIR, ANALYSIS_TYPE, OVERWRITE_ANALYSIS_DATA, PID
from .basic_utils.io import create_dir, delete_file_or_dir, save_json

ANALYSIS_CACHE_DYNAMIC = []  # used for recording dynamic information like model inputs across different batches
ANALYSIS_CACHE_STATIC = {}  # used for recording static information like model weights
ANALYSIS_CACHE_BATCH_ID = [0]  # used for saving cache at sample level (save after each forward)

if ANALYSIS_SAVE_DIR is not None and OVERWRITE_ANALYSIS_DATA:
    delete_file_or_dir(os.path.join(ANALYSIS_SAVE_DIR, "dynamic"), suppress_errors=True)  # remove old results
    delete_file_or_dir(os.path.join(ANALYSIS_SAVE_DIR, "static"), suppress_errors=True)  # remove old results
    print(f"Removed old results in {ANALYSIS_SAVE_DIR}.")


def save_analysis_cache_single_batch(save_static=True, reset_cache=True):
    # print("ANALYSIS_CACHE_DYNAMIC:", ANALYSIS_CACHE_DYNAMIC)
    # print("ANALYSIS_CACHE_STATIC:", ANALYSIS_CACHE_STATIC)

    if ANALYSIS_ENABLED:  # 🔍
        if len(ANALYSIS_CACHE_DYNAMIC) > 0:
            save_dir = os.path.join(ANALYSIS_SAVE_DIR, "dynamic", f"{PID}")
            save_file = os.path.join(save_dir, f"{ANALYSIS_CACHE_BATCH_ID[0]}.pt")
            create_dir(save_dir, suppress_errors=True)
            torch.save(ANALYSIS_CACHE_DYNAMIC, save_file, pickle_protocol=HIGHEST_PROTOCOL)
            if reset_cache:
                ANALYSIS_CACHE_DYNAMIC.clear()
            print(f"[{PID}] Dynamic cache successfully saved to {save_file}.")
        else:
            print(f"[{PID}] Skip saving the `ANALYSIS_CACHE_DYNAMIC` as it is empty.")

        if save_static:
            if len(ANALYSIS_CACHE_STATIC) > 0:
                save_dir = os.path.join(ANALYSIS_SAVE_DIR, "static", f"{PID}")
                save_file = os.path.join(save_dir, f"{ANALYSIS_CACHE_BATCH_ID[0]}.pt")
                create_dir(save_dir, suppress_errors=True)
                torch.save(ANALYSIS_CACHE_STATIC, save_file, pickle_protocol=HIGHEST_PROTOCOL)
                if reset_cache:
                    ANALYSIS_CACHE_STATIC.clear()
                print(f"[{PID}] Static cache successfully saved to {save_file}.")
            else:
                print(f"[{PID}] Skip saving the `ANALYSIS_CACHE_STATIC` as it is empty.")

        save_json(
            {
                "ANALYSIS_TYPE": ANALYSIS_TYPE,
                "ANALYSIS_SAVE_DIR": ANALYSIS_SAVE_DIR,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            os.path.join(ANALYSIS_SAVE_DIR, "info.json"),
            indent=4,
        )
        print(f"[{PID}] Analysis json successfully saved to {os.path.join(ANALYSIS_SAVE_DIR, 'info.json')}.")


def save_analysis_cache():
    # print("ANALYSIS_CACHE_DYNAMIC:", ANALYSIS_CACHE_DYNAMIC)
    # print("ANALYSIS_CACHE_STATIC:", ANALYSIS_CACHE_STATIC)

    if ANALYSIS_ENABLED:  # 🔍
        if len(ANALYSIS_CACHE_DYNAMIC) > 0:
            save_dir = os.path.join(ANALYSIS_SAVE_DIR, "dynamic")
            save_file = os.path.join(save_dir, f"{PID}")
            create_dir(save_dir, suppress_errors=True)
            torch.save(ANALYSIS_CACHE_DYNAMIC, save_file, pickle_protocol=HIGHEST_PROTOCOL)
            print(f"[{PID}] Dynamic cache successfully saved to {save_file}.")
        else:
            print(f"[{PID}] Skip saving the `ANALYSIS_CACHE_DYNAMIC` as it is empty.")

        if len(ANALYSIS_CACHE_STATIC) > 0:
            save_dir = os.path.join(ANALYSIS_SAVE_DIR, "static")
            save_file = os.path.join(save_dir, f"{PID}")
            create_dir(save_dir, suppress_errors=True)
            torch.save(ANALYSIS_CACHE_STATIC, save_file, pickle_protocol=HIGHEST_PROTOCOL)
            print(f"[{PID}] Static cache successfully saved to {save_file}.")
        else:
            print(f"[{PID}] Skip saving the `ANALYSIS_CACHE_STATIC` as it is empty.")

        save_json(
            {
                "ANALYSIS_TYPE": ANALYSIS_TYPE,
                "ANALYSIS_SAVE_DIR": ANALYSIS_SAVE_DIR,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            os.path.join(ANALYSIS_SAVE_DIR, "info.json"),
            indent=4,
        )
        print(f"[{PID}] Analysis json successfully saved to {os.path.join(ANALYSIS_SAVE_DIR, 'info.json')}.")
