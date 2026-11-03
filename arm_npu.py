"""NPU inference operator. Loading does not install packages or change model code."""
import os
from pathlib import Path
import torch
import torch_npu  # noqa: F401

_loaded = False


def load_library():
    global _loaded
    if _loaded:
        return
    path = Path(os.environ.get("ARM_LIBRARY", Path(__file__).parent / "build" / "libarm_torch.so"))
    if not path.is_file():
        raise FileNotFoundError(f"Build the operator first: bash build.sh; missing {path}")
    torch.ops.load_library(str(path.resolve()))
    _loaded = True


def add_rms_norm(x1, x2, gamma, epsilon=1e-6, model_compatible=False):
    load_library()
    return torch.ops.training_camp.add_rms_norm(x1, x2, gamma, epsilon, model_compatible)


def reference(x1, x2, gamma, epsilon=1e-6, model_compatible=False):
    """Three-output reference. model_compatible preserves Qwen2/3 rounding points."""
    dtype = x1.dtype
    if model_compatible:
        residual = x1 + x2
        values = residual.float()
        rstd = torch.rsqrt(values.square().mean(-1) + epsilon)
        y = (values * rstd.unsqueeze(-1)).to(dtype) * gamma
    else:
        values = x1.float() + x2.float()
        residual = values.to(dtype)
        rstd = torch.rsqrt((values.square() * (1.0 / values.shape[-1])).sum(-1) + epsilon)
        y = ((values * rstd.unsqueeze(-1)) * gamma.float()).to(dtype)
    return y, rstd, residual
