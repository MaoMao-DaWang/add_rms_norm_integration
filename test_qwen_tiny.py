"""NPU integration check with a randomly initialized two-layer Qwen3.

No checkpoint/tokenizer download. This tests integration, not pretrained model
quality or representative end-to-end performance.
"""
import argparse
import json
from pathlib import Path
import torch
import torch_npu  # noqa: F401
from transformers import Qwen3Config, Qwen3ForCausalLM
from qwen_adapter import enable_qwen_fusion


@torch.inference_mode()
def check_dtype(dtype_name):
    torch.manual_seed(2026)
    config = Qwen3Config(
        vocab_size=257, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, max_position_embeddings=256, attention_dropout=0.0,
        layer_types=["full_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        pad_token_id=0, bos_token_id=1, eos_token_id=None,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).to(device="npu:0", dtype=getattr(torch, dtype_name)).eval()
    ids = torch.tensor([[5, 6, 7, 8, 9, 10]], device="npu:0", dtype=torch.long)
    mask = torch.ones_like(ids)
    baseline = model(input_ids=ids, attention_mask=mask, use_cache=True)
    baseline_logits = baseline.logits.float().cpu()
    next_id = baseline.logits[:, -1:].argmax(-1)
    decode_mask = torch.ones((1, ids.shape[-1] + 1), device=ids.device, dtype=torch.long)
    baseline_decode = model(input_ids=next_id, attention_mask=decode_mask,
                            past_key_values=baseline.past_key_values, use_cache=True)
    baseline_decode_logits = baseline_decode.logits.float().cpu()
    generation_args = dict(input_ids=ids, attention_mask=mask, do_sample=False,
                           max_new_tokens=4, min_new_tokens=4, use_cache=True)
    baseline_ids = model.generate(**generation_args).cpu()
    tolerance = {"float32": 2e-4, "float16": 2e-3, "bfloat16": 1e-2}[dtype_name]
    handle = enable_qwen_fusion(model)
    try:
        fused = model(input_ids=ids, attention_mask=mask, use_cache=True)
        fused_logits = fused.logits.float().cpu()
        assert handle.calls == 2, f"Expected two decoder custom-op calls, got {handle.calls}"
        torch.testing.assert_close(fused_logits, baseline_logits, rtol=tolerance, atol=tolerance)
        fused_decode = model(input_ids=next_id, attention_mask=decode_mask,
                             past_key_values=fused.past_key_values, use_cache=True)
        fused_decode_logits = fused_decode.logits.float().cpu()
        assert handle.calls == 4, f"KV-cache decode did not invoke each fused layer: {handle.calls}"
        assert fused_decode.past_key_values.get_seq_length() == ids.shape[-1] + 1
        torch.testing.assert_close(fused_decode_logits, baseline_decode_logits,
                                   rtol=tolerance, atol=tolerance)
        fused_ids = model.generate(**generation_args).cpu()
        assert torch.equal(fused_ids, baseline_ids), "Four-token greedy sequences differ"
        result = {"dtype": dtype_name, "status": "PASS", "layers": 2, "hidden": 128,
                  "prefill_max_abs_error": (fused_logits - baseline_logits).abs().max().item(),
                  "decode_max_abs_error": (fused_decode_logits - baseline_decode_logits).abs().max().item(),
                  "rtol": tolerance, "atol": tolerance, "custom_op_calls": handle.calls,
                  "kv_cache_length": fused_decode.past_key_values.get_seq_length(),
                  "greedy_tokens_equal": True, "generated_ids": fused_ids.tolist()}
    finally:
        handle.restore()
    assert all("forward" not in layer.__dict__ for layer in model.model.layers)
    restored_logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float().cpu()
    torch.testing.assert_close(restored_logits, baseline_logits, rtol=tolerance, atol=tolerance)
    result["original_forward_restored"] = True
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-dtypes", action="store_true")
    parser.add_argument("--output", default="results/tiny_qwen_report.json")
    args = parser.parse_args()
    report = {"kind": "random_tiny_qwen3_integration_only", "tests": []}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for name in (["float32", "float16", "bfloat16"] if args.all_dtypes else ["float16"]):
            result = check_dtype(name)
            report["tests"].append(result)
            print(json.dumps(result, ensure_ascii=False))
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
        raise
    finally:
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report: {output}")


if __name__ == "__main__":
    main()
