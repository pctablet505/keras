"""Tests for PyTorch backend core utilities."""

import numpy as np
import pytest
import torch

from keras.src import backend
from keras.src import testing
from keras.src.backend.torch.core import convert_to_tensor
from keras.src.backend.torch.core import get_device
from keras.src.backend.torch.core import slice as torch_slice


def _get_backed_symint(hint=2):
    """Create a backed SymInt via torch.export dynamic shapes."""

    class _M(torch.nn.Module):
        def forward(self, x):
            return x + x.shape[0]

    ep = torch.export.export(
        _M(),
        (torch.randn(hint, 3),),
        dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
    )
    for node in ep.graph.nodes:
        if node.op == "placeholder":
            return node.meta["val"].shape[0]
    raise RuntimeError("Could not extract SymInt from exported program")


def _get_backed_symfloat(hint=2):
    """Create a backed SymFloat from a backed SymInt."""
    return torch.sym_float(_get_backed_symint(hint))


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Requires torch backend"
)
class TorchCoreTest(testing.TestCase):
    def _assert_sym_convert(
        self,
        value,
        expected_dtype,
        expected_item=None,
        expected_shape=None,
        expected_values=None,
        dtype=None,
    ):
        result = convert_to_tensor(value, dtype=dtype)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, expected_dtype)
        if expected_item is not None:
            self.assertEqual(result.item(), expected_item)
        if expected_shape is not None:
            self.assertEqual(tuple(result.shape), expected_shape)
        if expected_values is not None:
            self.assertListEqual(result.tolist(), expected_values)

    def test_convert_to_tensor_symint_scalar(self):
        self._assert_sym_convert(
            _get_backed_symint(5), torch.int64, expected_item=5
        )

    def test_convert_to_tensor_symfloat_scalar(self):
        self._assert_sym_convert(
            _get_backed_symfloat(5), torch.float32, expected_item=5.0
        )

    def test_convert_to_tensor_list_of_symint(self):
        self._assert_sym_convert(
            [_get_backed_symint(3), _get_backed_symint(4)],
            torch.int64,
            expected_shape=(2,),
            expected_values=[3, 4],
        )

    def test_convert_to_tensor_tuple_of_symfloat(self):
        self._assert_sym_convert(
            (_get_backed_symfloat(3), _get_backed_symfloat(4)),
            torch.float32,
            expected_shape=(2,),
            expected_values=[3.0, 4.0],
        )

    def test_convert_to_tensor_nested_list_of_symint(self):
        self._assert_sym_convert(
            [[_get_backed_symint(3), _get_backed_symint(4)]],
            torch.int64,
            expected_shape=(1, 2),
            expected_values=[[3, 4]],
        )

    def test_convert_to_tensor_explicit_dtype_for_symint(self):
        self._assert_sym_convert(
            _get_backed_symint(5),
            torch.float32,
            dtype="float32",
        )

    def test_slice_fast_path_accepts_symint(self):
        """slice fast path should accept SymInt without crashing."""
        x = torch.arange(24).reshape(2, 3, 4)
        batch = _get_backed_symint(2)
        start_indices = [0, 0, 0]
        shape = [batch, 2, 2]
        result = torch_slice(x, start_indices, shape)
        self.assertEqual(tuple(result.shape), (2, 2, 2))

    def test_slice_with_zero_d_tensor_start_indices(self):
        """slice must accept 0-d integer tensors as start indices."""
        device = get_device()
        x = torch.arange(10, dtype=torch.float32).reshape(2, 5).to(device)
        out = torch_slice(
            x,
            [torch.tensor(0).to(device), torch.tensor(1).to(device)],
            [2, 3],
        )
        expected = x[0:2, 1:4]
        self.assertAllClose(out.cpu().numpy(), expected.cpu().numpy())

    def test_to_static_index_rejects_tensor(self):
        """_to_static_index must reject torch.Tensor, even 0-d integer ones.

        Rejecting tensors here forces `slice()` to fall through to the
        tensor-native `torch.narrow` path instead of specializing a
        data-dependent bound to a concrete Python int on the fast path.
        """
        from keras.src.backend.torch.core import _to_static_index

        with self.assertRaises(TypeError):
            _to_static_index(torch.tensor(0))
        with self.assertRaises(TypeError):
            _to_static_index(torch.tensor(0.0))

    def test_slice_export_preserves_dynamic_dim(self):
        """A numpy-int bound alongside a symbolic dim must stay on the fast
        path under torch.export.

        Mixing a numpy integer bound with a symbolic dim fails the
        all-int/SymInt check, so the call falls through to the tensor-based
        slow path and the symbolic dim gets tensorized, specializing it under
        export. `_to_static_index` coerces the numpy int to a Python int so
        the fast path still applies and the symbolic dim passes through
        untouched.
        """

        class _SliceModule(torch.nn.Module):
            def forward(self, x):
                n = x.shape[0]
                return torch_slice(x, [np.int64(0), 0], [n, 2])

        ep = torch.export.export(
            _SliceModule(),
            (torch.arange(8).reshape(2, 4),),
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=2)}},
        )
        # Re-running with a different batch size must work; a specialized dim
        # would have failed export above or fixed the first output dim to 2.
        out = ep.module()(torch.arange(12).reshape(3, 4))
        self.assertEqual(tuple(out.shape), (3, 2))
