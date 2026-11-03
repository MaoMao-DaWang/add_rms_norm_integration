import argparse
import json
import time
import torch
import torch_npu  # noqa: F401
from arm_npu import add_rms_norm, reference


def measure(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    begin = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    begin.record()
    wall = time.perf_counter()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return {"wall_us": (time.perf_counter() - wall) * 1e6 / iterations,
            "stream_interval_us": begin.elapsed_time(end) * 1000 / iterations}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--model-compatible", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive; warmup must be nonnegative")
    dtype = getattr(torch, args.dtype)
    a = torch.randn(args.rows, args.hidden, device="npu:0", dtype=dtype)
    b = torch.randn_like(a)
    g = torch.ones(args.hidden, device=a.device, dtype=dtype)
    functions = {"custom_ascendc": lambda: add_rms_norm(a, b, g, 1e-6, args.model_compatible),
                 "torch_composite": lambda: reference(a, b, g, 1e-6, args.model_compatible)}
    expected = functions["torch_composite"]()
    actual = functions["custom_ascendc"]()
    from test_operator import check
    check(actual, expected, dtype)
    result = {name: measure(fn, args.warmup, args.iterations) for name, fn in functions.items()}
    print(json.dumps({"shape": [args.rows, args.hidden], "dtype": args.dtype,
                      "model_compatible": args.model_compatible, "timings": result}, indent=2))


if __name__ == "__main__":
    main()
