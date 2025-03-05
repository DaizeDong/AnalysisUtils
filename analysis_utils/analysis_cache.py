import os
from datetime import datetime
from pickle import HIGHEST_PROTOCOL

import torch
from tqdm import tqdm

from .analysis_env import ANALYSIS_ENABLED, ANALYSIS_SAVE_DIR, ANALYSIS_TYPE, OVERWRITE_ANALYSIS_DATA, PID
from .basic_utils.io import create_dir, delete_file_or_dir, save_json
from .basic_utils.operations.operation_tensor import concat_tensors

ANALYSIS_CACHE_DYNAMIC = []  # used for recording dynamic information like model inputs across different batches
ANALYSIS_CACHE_STATIC = {}  # used for recording static information like model weights
ANALYSIS_CACHE_BATCH_ID = 0  # used for saving cache at sample level (save after each forward)
ANALYSIS_TOKEN_NUM = 0  # used for recording the total number of tokens processed during analysis

if ANALYSIS_SAVE_DIR is not None and OVERWRITE_ANALYSIS_DATA:
    delete_file_or_dir(os.path.join(ANALYSIS_SAVE_DIR, "dynamic"), suppress_errors=True)  # remove old results
    delete_file_or_dir(os.path.join(ANALYSIS_SAVE_DIR, "static"), suppress_errors=True)  # remove old results
    print(f"[{PID}] Removed old cache results in {ANALYSIS_SAVE_DIR}.")


def compress_tensors(tensor_list, dim=0):
    """
    Process a nested structure of tensors, yielding concatenated tensors in a memory-efficient way.

    Rules:
    - Skip None values in the list.
    - If tensors have dim[0] = 1, merge consecutive ones into a single tensor.
    - If tensor dim[0] > 1, yield it as is.
    - If the structure is dict/list, preserve its structure while applying the above rules.

    Args:
        tensor_list (list): Nested structure (list/dict of tensors).
        dim (int): Dimension along which to concatenate tensors.

    Yields:
        The processed tensor structure.
    """

    def get_tensor_size(value, dim=0):
        """Get the size of the first tensor in `value` along the specified dimension."""
        if value is None:
            return 0
        elif isinstance(value, torch.Tensor):
            return value.size(dim)
        elif isinstance(value, dict):
            return get_tensor_size(next(iter(value.values())), dim)
        elif isinstance(value, list):
            return get_tensor_size(value[0], dim)
        else:  # unsupported types
            return -1

    buffer = []

    for element in tqdm(tensor_list, desc="Compressing tensors"):
        if get_tensor_size(element, dim) <= 0:
            continue

        if get_tensor_size(element, dim) == 1:
            buffer.append(element)
        else:  # meet a tensor with size > 1
            # yield the buffer
            if buffer:
                yield concat_tensors(buffer, dim=dim, auto_reshape=True, strict=False)
                buffer = []
            # yield the tensor
            yield element

    if buffer:
        yield concat_tensors(buffer, dim=dim, auto_reshape=True, strict=False)  # remaining `size=1` tensors


def save_analysis_cache_single_batch(save_static=True, reset_cache=True, compress=False):
    """Save analysis cache for a single batch."""
    if ANALYSIS_ENABLED:  # 🔍
        if len(ANALYSIS_CACHE_DYNAMIC) > 0:
            save_dir = os.path.join(ANALYSIS_SAVE_DIR, "dynamic", f"{PID}")
            save_file = os.path.join(save_dir, f"{ANALYSIS_CACHE_BATCH_ID}.pt")
            create_dir(save_dir, suppress_errors=True)
            if compress:
                compressed_tensors = [v for v in compress_tensors(ANALYSIS_CACHE_DYNAMIC)]
                print(f"[{PID}] Reduced the number of tensors from {len(ANALYSIS_CACHE_DYNAMIC)} to {len(compressed_tensors)} by compression.")
                torch.save(compressed_tensors, save_file, pickle_protocol=HIGHEST_PROTOCOL)
                print(f"[{PID}] Total {len(compressed_tensors)} dynamic cache successfully saved to {save_file}.")
            else:
                torch.save(ANALYSIS_CACHE_DYNAMIC, save_file, pickle_protocol=HIGHEST_PROTOCOL)
                print(f"[{PID}] Total {len(ANALYSIS_CACHE_DYNAMIC)} dynamic cache successfully saved to {save_file}.")
            if reset_cache:
                ANALYSIS_CACHE_DYNAMIC.clear()
        else:
            print(f"[{PID}] Skip saving the `ANALYSIS_CACHE_DYNAMIC` as it is empty.")

        if save_static:
            if len(ANALYSIS_CACHE_STATIC) > 0:
                save_dir = os.path.join(ANALYSIS_SAVE_DIR, "static", f"{PID}")
                save_file = os.path.join(save_dir, f"{ANALYSIS_CACHE_BATCH_ID}.pt")
                create_dir(save_dir, suppress_errors=True)
                torch.save(ANALYSIS_CACHE_STATIC, save_file, pickle_protocol=HIGHEST_PROTOCOL)
                if reset_cache:
                    ANALYSIS_CACHE_STATIC.clear()
                print(f"[{PID}] Total {len(ANALYSIS_CACHE_STATIC)} static cache successfully saved to {save_file}.")
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


def save_analysis_cache(compress=False):
    """Save all analysis cache at once."""
    if ANALYSIS_ENABLED:  # 🔍
        if len(ANALYSIS_CACHE_DYNAMIC) > 0:
            save_dir = os.path.join(ANALYSIS_SAVE_DIR, "dynamic")
            save_file = os.path.join(save_dir, f"{PID}.pt")
            create_dir(save_dir, suppress_errors=True)
            if compress:
                compressed_tensors = [v for v in compress_tensors(ANALYSIS_CACHE_DYNAMIC)]
                print(f"[{PID}] Reduced the number of tensors from {len(ANALYSIS_CACHE_DYNAMIC)} to {len(compressed_tensors)} by compression.")
                torch.save(compressed_tensors, save_file, pickle_protocol=HIGHEST_PROTOCOL)
                print(f"[{PID}] Total {len(compressed_tensors)} dynamic cache successfully saved to {save_file}.")
            else:
                torch.save(ANALYSIS_CACHE_DYNAMIC, save_file, pickle_protocol=HIGHEST_PROTOCOL)
                print(f"[{PID}] Total {len(ANALYSIS_CACHE_DYNAMIC)} dynamic cache successfully saved to {save_file}.")
        else:
            print(f"[{PID}] Skip saving the `ANALYSIS_CACHE_DYNAMIC` as it is empty.")

        if len(ANALYSIS_CACHE_STATIC) > 0:
            save_dir = os.path.join(ANALYSIS_SAVE_DIR, "static")
            save_file = os.path.join(save_dir, f"{PID}.pt")
            create_dir(save_dir, suppress_errors=True)
            torch.save(ANALYSIS_CACHE_STATIC, save_file, pickle_protocol=HIGHEST_PROTOCOL)
            print(f"[{PID}] Total {len(ANALYSIS_CACHE_STATIC)} static cache successfully saved to {save_file}.")
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
