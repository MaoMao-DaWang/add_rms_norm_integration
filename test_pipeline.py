"""Exercise queued PyTorch producers -> direct Ascend C consumer without CPU reads."""
import torch
import torch_npu  # noqa: F401
from arm_npu import add_rms_norm, reference
from test_operator import check


@torch.inference_mode()
def main():
    count = 0
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        # Binary fractions make producer arithmetic exactly representable, so
        # CPU-vs-NPU producer rounding cannot hide a queue-ordering failure.
        base_cpu = ((torch.arange(33 * 129).reshape(33, 129) % 17) - 8).to(dtype) / 16
        gamma_cpu = ((torch.arange(129) % 7) + 1).to(dtype)
        for compatible in (False, True):
            expected = []
            for i in range(8):
                a_cpu = (base_cpu + (i + 1) / 16) * 0.5
                b_cpu = base_cpu * 0.125 - 0.125
                expected.append(reference(a_cpu, b_cpu, gamma_cpu, 1e-6, compatible))
            for use_secondary in (False, True):
                stream = torch.npu.Stream() if use_secondary else torch.npu.current_stream()
                actual = []
                with torch.npu.stream(stream):
                    base = base_cpu.to("npu:0")
                    gamma = gamma_cpu.to("npu:0")
                    # No .cpu(), .item(), synchronize(), or CPU reference work
                    # between the producers and the direct kernel launches.
                    for i in range(8):
                        a = (base + (i + 1) / 16) * 0.5
                        b = base * 0.125 - 0.125
                        actual.append(add_rms_norm(a, b, gamma, 1e-6, compatible))
                stream.synchronize()
                for result, golden in zip(actual, expected):
                    check(result, golden, dtype)
                    count += 1
    print(f"PASS: {count} producer-consumer cases; no intermediate CPU reads; default/non-default streams")


if __name__ == "__main__":
    main()
