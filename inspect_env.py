"""Read-only remote probe. No downloads and no package changes."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys


def main():
    print("python:", sys.version.split()[0], sys.executable)
    print("machine:", platform.machine())
    for name in ("torch", "torch-npu", "transformers", "numpy"):
        try:
            print(name + ":", importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            print(name + ": not installed")
    root = Path(os.environ.get("ASCEND_HOME_PATH", "/home/developer/Ascend/cann-9.0.0"))
    print("CANN root:", root)
    for name in ("bisheng", "ccec", "cmake", "c++"):
        print(name + ":", shutil.which(name))
    for relative in ("version.cfg", "aarch64-linux/ascend_toolkit_install.info"):
        file = root / relative
        if file.is_file():
            print(str(file) + ":\n" + file.read_text(errors="replace")[:2000])
    import torch
    import torch_npu
    print("NPU available:", torch.npu.is_available())
    print("NPU count:", torch.npu.device_count())
    with torch.inference_mode():
        x = torch.tensor([1.0, 2.0, 3.0], device="npu:0")
        print("NPU add:", (x + x).cpu().tolist())
    print("vendor add_rms_norm API:", hasattr(torch_npu, "npu_add_rms_norm"))
    print("Local model configurations (maximum 20):")
    found = 0
    roots = [Path("/mnt/workspace/models"), Path.home() / ".cache/huggingface/hub"]
    for directory in roots:
        if not directory.is_dir():
            continue
        for file in directory.rglob("config.json"):
            try:
                data = json.loads(file.read_text())
            except (OSError, ValueError):
                continue
            print(json.dumps({"path": str(file.parent), "model_type": data.get("model_type"),
                              "architectures": data.get("architectures"),
                              "hidden_size": data.get("hidden_size")}, ensure_ascii=False))
            found += 1
            if found == 20:
                return
    if not found:
        print("No config.json found in the two standard model directories; pass your model path explicitly.")


if __name__ == "__main__":
    main()
