# Analysis Utils for Model Evaluation

A plug-and-play analysis module containing logic for global storage & environments, designed for fast integration into various evaluation frameworks.

## Usage

Install the repository in the editable mode:

```bash
git clone --recursive https://github.com/DaizeDong/AnalysisUtils.git
cd AnalysisUtils
pip install -e . --no-build-isolation
```

Set system variables to activate the analysis module before launching:

```bash
export ANALYSIS_TYPE="weights,inputs,outputs" # [required] will be converted into a str list if not empty (the analysis is enabled as long as the name `ANALYSIS_TYPE` is defined)
export ANALYSIS_SAVE_DIR="XXXXXXXXX"          # [required] path to save the analysis results
export OVERWRITE_ANALYSIS_DATA="1"            # [optional] default is "0" (not overwrite)
```

Change the code to use the analysis module:

- Initialize at the entry point.
- Record information during forward.
- Save results when finished.

```python
import torch
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

# save results to `ANALYSIS_SAVE_DIR`
save_analysis_cache()
```