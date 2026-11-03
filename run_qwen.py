"""Offline, reversible Qwen2/Qwen3 inference comparison on the same checkpoint."""
import argparse
import json
from pathlib import Path
import time
import torch
import torch_npu  # noqa: F401
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from qwen_adapter import enable_qwen_fusion


def generate_timed(model, inputs, tokens, repeats, warmup):
    options = dict(do_sample=False, use_cache=True, max_new_tokens=tokens, min_new_tokens=tokens)
    for _ in range(warmup):
        model.generate(**inputs, **options)
    torch.npu.synchronize()
    durations = []
    ids = None
    for _ in range(repeats):
        torch.npu.synchronize()
        start = time.perf_counter()
        ids = model.generate(**inputs, **options)
        torch.npu.synchronize()
        durations.append(time.perf_counter() - start)
    generated = ids.shape[-1] - inputs["input_ids"].shape[-1]
    average = sum(durations) / len(durations)
    return ids.cpu(), {"generation_seconds": durations, "mean_seconds": average,
                       "new_tokens": generated, "tokens_per_second": generated / average}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local checkpoint directory; no downloads")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--prompt", default="Please explain what RMSNorm does in one sentence.")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--logit-atol", type=float, default=None)
    parser.add_argument("--logit-rtol", type=float, default=0.01)
    parser.add_argument("--profile", action="store_true", help="Export CPU dispatch trace of the fused NPU call")
    parser.add_argument("--output", default="results/qwen_report.json")
    args = parser.parse_args()
    if args.tokens < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("tokens/repeats must be positive; warmup must be nonnegative")
    path = Path(args.model).expanduser().resolve()
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if config.model_type not in ("qwen2", "qwen3"):
        parser.error(f"Unsupported model_type={config.model_type}; this adapter supports plain Qwen2/Qwen3")
    if config.hidden_size > 16384:
        parser.error("Model hidden size exceeds 16384")
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        path, config=config, local_files_only=True, trust_remote_code=False,
        dtype=dtype, attn_implementation="eager").to("npu:0").eval()
    inputs = tokenizer(args.prompt, return_tensors="pt").to("npu:0")
    if model.generation_config.pad_token_id is None and tokenizer.eos_token_id is not None:
        model.generation_config.pad_token_id = tokenizer.eos_token_id
    baseline_logits = model(**inputs, use_cache=False).logits.float().cpu()
    baseline_ids, baseline_timing = generate_timed(model, inputs, args.tokens, args.repeats, args.warmup)
    handle = enable_qwen_fusion(model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"model": str(path), "model_type": config.model_type, "dtype": args.dtype,
              "hidden_size": config.hidden_size, "decoder_layers": config.num_hidden_layers,
              "torch": torch.__version__, "transformers": __import__("transformers").__version__}
    try:
        fused_logits = model(**inputs, use_cache=False).logits.float().cpu()
        difference = (fused_logits - baseline_logits).abs()
        atol = args.logit_atol
        if atol is None:
            atol = {"float32": 1e-3, "float16": 0.05, "bfloat16": 0.2}[args.dtype]
        close = torch.isclose(fused_logits, baseline_logits, atol=atol, rtol=args.logit_rtol)
        report["logits"] = {"max_abs_error": difference.max().item(),
                            "matched_ratio": close.float().mean().item(),
                            "atol": atol, "rtol": args.logit_rtol, "passed": bool(close.all())}
        report["custom_op_calls_after_logits"] = handle.calls
        assert handle.calls == config.num_hidden_layers, "Expected one fused call per decoder layer"
        if not close.all():
            raise AssertionError("Logits differ beyond the explicitly reported model tolerances")
        fused_ids, fused_timing = generate_timed(model, inputs, args.tokens, args.repeats, args.warmup)
        report["greedy_tokens_equal"] = torch.equal(baseline_ids, fused_ids)
        report["baseline"] = baseline_timing
        report["fused"] = fused_timing
        report["generation_speedup"] = baseline_timing["mean_seconds"] / fused_timing["mean_seconds"]
        report["custom_op_calls_total"] = handle.calls
        report["generated_text"] = tokenizer.decode(fused_ids[0], skip_special_tokens=True)
        if args.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                        record_shapes=True) as prof:
                model(**inputs, use_cache=False)
                torch.npu.synchronize()
            trace = output.parent / "qwen_dispatch_trace.json"
            prof.export_chrome_trace(str(trace))
            report["dispatch_trace"] = str(trace)
        if not report["greedy_tokens_equal"]:
            raise AssertionError("Greedy token sequences differ; inspect correctness before claiming speedup")
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
        raise
    finally:
        handle.restore()
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
