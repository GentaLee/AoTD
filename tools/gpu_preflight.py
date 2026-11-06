import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import torch

root = Path(os.environ['AOTD_ROOT'])
result = dict(torch=torch.__version__, cuda=torch.version.cuda, packages={}, gpus=[])
for name in ['transformers', 'accelerate', 'datasets', 'decord', 'numpy', 'deepspeed', 'pytorch-triton', 'helion']:
    try:
        result['packages'][name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        result['packages'][name] = None
assert torch.cuda.is_available()
for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        x = torch.ones((64,64), device=f'cuda:{i}', dtype=torch.bfloat16)
        value = (x @ x).sum().item()
        assert value == 262144.0
        result['gpus'].append(dict(index=i, name=torch.cuda.get_device_name(i),
                                  capability=torch.cuda.get_device_capability(i), bf16_matmul_sum=value))
subprocess.run(['nvidia-smi'], check=True)
print(json.dumps(result, indent=2), flush=True)
(Path(os.environ['AOTD_RUN_DIR'])/'environment.json').write_text(json.dumps(result, indent=2))
