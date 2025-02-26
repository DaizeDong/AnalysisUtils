# Analysis Utils for Model Evaluation

A plug-and-play analysis module containing global storage & environments for fast integration into various evaluation frameworks.

## Usage

Register the repository as a submodule:

```bash
git clone --recursive https://github.com/DaizeDong/AnalysisUtils.git
cd AnalysisUtils
pip install -e .
```

Set system variables to activate the analysis module:

```bash
export ANALYSIS_TYPE="weights,inputs,outputs"
export ANALYSIS_SAVE_DIR="XXXXXXXXX"
export OVERWRITE_ANALYSIS_DATA="1"  # optional
```

Change the code to use the analysis module:

- Initialize at the entry point.
- Record information during inference.
- Save results when done.

```python
import torch
import analysis_utils
from analysis_utils import ANALYSIS_TYPE, ANALYSIS_CACHE_STATIC, ANALYSIS_CACHE_DYNAMIC, save_analysis_cache

# load model
model = torch.nn.Linear(128, 256, device="cuda")
model.eval()

if "weights" in ANALYSIS_TYPE:
    ANALYSIS_CACHE_STATIC["weights"] = model.weight.clone().cpu()

# forward inputs
for _ in range(100):
    ANALYSIS_CACHE_DYNAMIC.append({})

    inputs = torch.randn(1, 128, device="cuda")
    outputs = model(inputs)

    if "inputs" in ANALYSIS_TYPE and "inputs" not in ANALYSIS_CACHE_DYNAMIC[-1]:
        ANALYSIS_CACHE_DYNAMIC[-1]["inputs"] = inputs.cpu()

    if "outputs" in ANALYSIS_TYPE and "outputs" not in ANALYSIS_CACHE_DYNAMIC[-1]:
        ANALYSIS_CACHE_DYNAMIC[-1]["outputs"] = inputs.cpu()

save_analysis_cache()
```