"""Tests for PyTorch backend core utilities."""

import pytest
import torch

from keras.src import backend
from keras.src import testing
from keras.src.backend.torch.core import convert_to_tensor
from keras.src.backend.torch.core import get_device
from keras.src.backend.torch.core import slice as torch_slice
from keras.src.backend.torch.core import slice_update


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

    def test_slice_mismatched_start_indices_and_shape_length_raises(self):
        """slice() must raise ValueError when lengths of start_indices and
        shape differ, instead of silently truncating via zip()."""
        x = torch.arange(24).reshape(2, 3, 4)
        with self.assertRaises(ValueError) as cm:
            torch_slice(x, [0, 0], [2, 2, 2])
        message = str(cm.exception)
        self.assertIn("length 2", message)
        self.assertIn("length 3", message)

    def test_slice_update_1d_correctness(self):
        """slice_update on a 1-D tensor produces the correct values."""
        device = get_device()
        x = torch.zeros(8, dtype=torch.float32).to(device)
        updates = torch.tensor([9.0, 10.0, 11.0, 12.0]).to(device)
        out = slice_update(x, [1], updates)
        expected = torch.tensor([0.0, 9.0, 10.0, 11.0, 12.0, 0.0, 0.0, 0.0])
        self.assertAllClose(out.cpu().numpy(), expected.numpy())

    def test_slice_update_2d_correctness(self):
        """slice_update on a 2-D tensor at a non-zero start position."""
        device = get_device()
        x = torch.ones(4, 3, dtype=torch.float32).to(device)
        updates = torch.full((2, 3), 5.0).to(device)
        out = slice_update(x, [1, 0], updates)
        expected = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [5.0, 5.0, 5.0],
                [5.0, 5.0, 5.0],
                [1.0, 1.0, 1.0],
            ]
        )
        self.assertAllClose(out.cpu().numpy(), expected.numpy())

    def test_slice_update_nd_offset_correctness(self):
        """slice_update with an interior start position on an N-D tensor."""
        device = get_device()
        x = torch.zeros(3, 6, dtype=torch.float32).to(device)
        updates = torch.ones(2, 3).to(device)
        out = slice_update(x, [1, 2], updates)
        self.assertAllClose(
            out[1:3, 2:5].cpu().numpy(), torch.ones(2, 3).numpy()
        )
        # Untouched region must remain zero.
        self.assertAllClose(out[0, :].cpu().numpy(), torch.zeros(6).numpy())
