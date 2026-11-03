"""Conservative, reversible eager-mode fusion for plain Qwen2/Qwen3 decoder layers.

Rewrite the installed forward's three matching statements, preserving its actual
signature, attention/KV-cache code, output convention, and all other statements.
Unsupported structures fail before any model instance is changed.
"""
import ast
import functools
import inspect
import textwrap
import types
import __future__

_PATTERN = ast.parse("""
hidden_states = residual + hidden_states
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
""").body
_REPLACEMENT = """
hidden_states, _arm_unused_rstd, residual = _arm_fused_call(
    hidden_states, residual, self.post_attention_layernorm.weight,
    float(self.post_attention_layernorm.variance_epsilon), True)
"""


def rewrite_forward(function, fused_callable):
    """Build a new function; never edit the installed transformers source file."""
    if function.__code__.co_freevars:
        raise ValueError("Decoder forward with closure variables is unsupported")
    module = ast.parse(textwrap.dedent(inspect.getsource(function)))
    definitions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if len(definitions) != 1:
        raise ValueError("Expected exactly one decoder forward function")
    node = definitions[0]
    target = [ast.dump(stmt, include_attributes=False) for stmt in _PATTERN]
    matches = [i for i in range(len(node.body) - 2) if
               [ast.dump(stmt, include_attributes=False) for stmt in node.body[i:i + 3]] == target]
    if len(matches) != 1:
        raise ValueError(f"Expected one Add->RMSNorm pattern, found {len(matches)}; no patch applied")
    i = matches[0]
    node.body[i:i + 3] = ast.parse(_REPLACEMENT).body
    node.decorator_list = []
    ast.fix_missing_locations(module)
    namespace = dict(function.__globals__)
    namespace["_arm_fused_call"] = fused_callable
    code = compile(module, "<add_rms_norm_qwen_adapter>", "exec",
                   flags=__future__.annotations.compiler_flag, dont_inherit=True)
    exec(code, namespace)
    return functools.update_wrapper(namespace[node.name], function)


class PatchHandle:
    def __init__(self, originals, counters):
        self.originals = originals
        self.counters = counters

    @property
    def calls(self):
        return self.counters["calls"]

    def restore(self):
        for layer, original, had_override in self.originals:
            if had_override:
                layer.forward = original
            else:
                delattr(layer, "forward")
        self.originals.clear()


def enable_qwen_fusion(model):
    import torch
    from arm_npu import add_rms_norm, load_library
    if getattr(model.config, "model_type", None) not in ("qwen2", "qwen3"):
        raise ValueError("Only plain Qwen2/Qwen3 are supported; Next/MoE/VL/Qwen3.5 need separate adapters")
    if model.training:
        raise ValueError("Call model.eval(); this implementation supports inference only")
    layers = model.model.layers
    if not layers:
        raise ValueError("No decoder layers found")
    counters = {"calls": 0}

    def fused(*args):
        if torch.is_grad_enabled():
            raise RuntimeError("Use torch.inference_mode() for fused model inference")
        a, b, weight, eps, compatible = args
        result = add_rms_norm(a.contiguous(), b.contiguous(), weight.contiguous(), eps, compatible)
        counters["calls"] += 1
        return result

    pending = []
    functions = {}
    for layer in layers:
        cls = type(layer)
        if cls.__name__ not in ("Qwen2DecoderLayer", "Qwen3DecoderLayer"):
            raise ValueError(f"Unsupported decoder class: {cls.__name__}")
        if "forward" in layer.__dict__:
            raise ValueError("Decoder already has a forward override; restore it before patching")
        norm = layer.post_attention_layernorm
        if type(norm).__name__ not in ("Qwen2RMSNorm", "Qwen3RMSNorm"):
            raise ValueError("Unsupported normalization variant")
        if norm.weight.numel() > 16384 or norm.weight.device.type != "npu":
            raise ValueError("Normalization weight must be on NPU with hidden size <= 16384")
        if cls not in functions:
            functions[cls] = rewrite_forward(cls.forward, fused)
        pending.append((layer, functions[cls]))
    load_library()
    originals = []
    for layer, forward in pending:
        originals.append((layer, layer.forward, False))
        layer.forward = types.MethodType(forward, layer)
    return PatchHandle(originals, counters)
