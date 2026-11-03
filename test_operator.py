import argparse
import torch
import torch_npu  # noqa: F401
from arm_npu import add_rms_norm, reference


def check(actual, expected, dtype):
    tolerance = {torch.float32: (1e-4, 1e-4), torch.float16: (2e-3, 2e-3),
                 torch.bfloat16: (2e-2, 2e-2)}[dtype]
    for i, (a, b) in enumerate(zip(actual, expected)):
        assert a.shape == b.shape and a.dtype == b.dtype
        if i == 2:
            torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
        else:
            rtol, atol = (1e-4, 1e-4) if i == 1 else tolerance
            torch.testing.assert_close(a.float().cpu(), b.float().cpu(), rtol=rtol, atol=atol)


@torch.inference_mode()
def run(quick=False):
    torch.manual_seed(2280)
    shapes = [(1, 1), (3, 7), (33, 17), (2, 3, 129), (2, 2, 3, 33),
              (1, 2049), (2, 4096), (1, 16384)]
    if not quick:
        shapes += [(65, 31), (129, 1024), (32, 8192), (128, 16384)]
    device = "npu:0"
    count = 0
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for shape in shapes:
            a = (torch.rand(shape) * 4 - 2).to(device=device, dtype=dtype)
            b = (torch.rand(shape) * 4 - 2).to(device=device, dtype=dtype)
            g = (torch.rand(shape[-1]) * 4 - 2).to(device=device, dtype=dtype)
            before = [x.cpu().clone() for x in (a, b, g)]
            for compatible in (False, True):
                for eps in ([1e-6] if quick else [1e-8, 1e-6, 0.5]):
                    expected = reference(*before, eps, compatible)
                    actual = add_rms_norm(a, b, g, eps, compatible)
                    check(actual, expected, dtype)
                    pointers = {x.data_ptr() for x in (*actual, a, b, g)}
                    assert len(pointers) == 6, "outputs must not alias inputs or each other"
                    count += 1
            for x, original in zip((a, b, g), before):
                torch.testing.assert_close(x.cpu(), original, rtol=0, atol=0)
        # Zero rows, positive epsilon, and an explicitly non-default stream.
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            a = torch.zeros((33, 31), device=device, dtype=dtype)
            b = torch.zeros_like(a)
            g = torch.ones(31, device=device, dtype=dtype)
            actual = add_rms_norm(a, b, g, 1e-6)
            expected = reference(a, b, g, 1e-6)
        stream.synchronize()
        check(actual, expected, dtype)
        count += 1
    torch.npu.synchronize()
    print(f"PASS: {count} NPU cases; three outputs, both rounding modes, tails, non-default stream")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(args.quick)
