"""Independent FP64 CPU reference and causal-prefix invariance."""
import math
import unittest
import torch
from attention import attention, make_inputs


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for W9 validation")
        torch.backends.cuda.matmul.allow_tf32 = False

    def test_equivalence_all_benchmark_shapes(self):
        for sequence in (1, 127, 128, 512, 1024):
            with self.subTest(sequence=sequence):
                inputs = make_inputs(sequence)
                torch.testing.assert_close(attention(*inputs), attention(*inputs, inplace=True),
                                           rtol=1e-5, atol=1e-6)

    def test_independent_cpu_reference(self):
        for sequence in (1, 7, 33):
            q, k, v, mask = make_inputs(sequence, batch=2, heads=2, dim=16)
            qc, kc, vc = [x.cpu().double() for x in (q, k, v)]
            # Each query sees only its prefix, without using the implementation's mask.
            rows = []
            for t in range(sequence):
                scores = (qc[:, :, t:t+1] @ kc[:, :, :t+1].transpose(-2, -1)) / math.sqrt(16)
                rows.append(torch.softmax(scores, -1) @ vc[:, :, :t+1])
            expected = torch.cat(rows, dim=2)
            for inplace in (False, True):
                with self.subTest(sequence=sequence, inplace=inplace):
                    actual = attention(q, k, v, mask, inplace).cpu().double()
                    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_future_tokens_do_not_change_prefix(self):
        q, k, v, mask = make_inputs(33, heads=2, dim=16)
        cutoff = 17
        q2, k2, v2 = [x.clone() for x in (q, k, v)]
        for x in (q2, k2, v2):
            x[:, :, cutoff:] += 10
        for inplace in (False, True):
            before = attention(q, k, v, mask, inplace)
            after = attention(q2, k2, v2, mask, inplace)
            torch.testing.assert_close(before[:, :, :cutoff], after[:, :, :cutoff],
                                       rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

