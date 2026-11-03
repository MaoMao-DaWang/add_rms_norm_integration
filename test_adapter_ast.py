"""CPU-only structural checks. Does not claim to validate Transformers or NPU execution."""
import unittest
import numpy as np
from qwen_adapter import rewrite_forward


class Norm:
    weight = np.array([0.5, 1.0, 2.0], np.float32)
    variance_epsilon = 1e-6

    def __call__(self, values):
        return values / np.sqrt(np.mean(values * values, axis=-1, keepdims=True) +
                                self.variance_epsilon) * self.weight


class Decoder:
    post_attention_layernorm = Norm()

    def forward(self, hidden_states, cache=None, *, factor=0.25):
        residual = hidden_states
        hidden_states = hidden_states * factor
        if cache is not None:
            cache.append(hidden_states.copy())
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states * 2
        hidden_states = residual + hidden_states
        return hidden_states, cache


def no_pattern(self, hidden_states):
    return hidden_states


class AdapterTest(unittest.TestCase):
    def test_output_cache_signature_and_original(self):
        import inspect
        calls = []

        def fused(a, b, weight, eps, compatible):
            calls.append(compatible)
            residual = a + b
            rstd = 1 / np.sqrt(np.mean(residual * residual, axis=-1) + eps)
            return residual * rstd[:, None] * weight, rstd, residual

        original = Decoder.forward
        patched = rewrite_forward(original, fused)
        self.assertEqual(inspect.signature(patched), inspect.signature(original))
        x = np.array([[1., 2., 3.], [-1., 0., 1.]], np.float32)
        a, ac = original(Decoder(), x, [], factor=0.5)
        b, bc = patched(Decoder(), x, [], factor=0.5)
        np.testing.assert_allclose(a, b, rtol=1e-6)
        np.testing.assert_array_equal(ac[0], bc[0])
        self.assertEqual(calls, [True])
        self.assertIs(Decoder.forward, original)

    def test_unmatched_forward_rejected(self):
        with self.assertRaisesRegex(ValueError, "found 0"):
            rewrite_forward(no_pattern, lambda *args: None)

    def test_closure_rejected(self):
        value = 2

        def closed(self, hidden_states):
            return hidden_states * value

        with self.assertRaisesRegex(ValueError, "closure"):
            rewrite_forward(closed, lambda *args: None)


if __name__ == "__main__":
    unittest.main()
