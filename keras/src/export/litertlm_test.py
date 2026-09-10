import os
import tempfile
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

import keras
from keras.src import testing
from keras.src.export.litertlm import GenericKVAdapter
from keras.src.export.litertlm import _build_signatures_map
from keras.src.export.litertlm import export_litertlm
from keras.src.export.litertlm import resolve_litertlm_spec
from keras.src.export.litertlm_spec import LiteRTLMSpec


class LiteRTLMSpecTest(testing.TestCase):
    def test_spec_initialization(self):
        spec = LiteRTLMSpec(
            prefill_fn=lambda *args, **kwargs: {},
            decode_fn=lambda *args, **kwargs: {},
            prefill_sample_inputs={"tokens": np.zeros((1, 8))},
            decode_sample_inputs={"tokens": np.zeros((1, 1))},
            tokenizer_path="/path/to/tokenizer.model",
            context_length=512,
            num_layers=4,
            num_kv_heads=2,
            head_dim=32,
            stop_token_ids=[1, 2],
            model_type="test_model",
        )
        self.assertEqual(spec.context_length, 512)
        self.assertEqual(spec.num_layers, 4)
        self.assertEqual(spec.num_kv_heads, 2)
        self.assertEqual(spec.head_dim, 32)
        self.assertEqual(spec.stop_token_ids, [1, 2])
        self.assertEqual(spec.model_type, "test_model")


class GenericKVAdapterTest(testing.TestCase):
    @unittest.skipUnless(
        keras.config.backend() == "torch", "Torch backend only"
    )
    def test_adapter_stack_and_unstack(self):
        import torch

        num_layers = 2
        cache_length = 16
        num_kv_heads = 2
        head_dim = 8

        class MockModel(torch.nn.Module):
            def call_with_cache(self, tokens, cache, cache_update_index):
                # cache shape: [batch, layers, 2, length, heads, dim]
                self.received_cache_shape = cache.shape
                self.received_index = cache_update_index
                updated_cache = cache + 1.0
                logits = torch.ones((tokens.shape[0], tokens.shape[1], 100))
                return logits, None, updated_cache

        mock_model = MockModel()
        adapter = GenericKVAdapter(
            mock_model, num_layers=num_layers, cache_length=cache_length
        )

        # Prepare flat inputs
        tokens = torch.zeros((1, 4), dtype=torch.int32)
        input_pos = torch.arange(4, dtype=torch.int32)
        kv_cache = {}
        for i in range(num_layers):
            kv_cache[f"kv_cache_k_{i}"] = torch.zeros(
                (1, cache_length, num_kv_heads, head_dim)
            )
            kv_cache[f"kv_cache_v_{i}"] = torch.zeros(
                (1, cache_length, num_kv_heads, head_dim)
            )

        # Test prefill
        prefill_out = adapter.prefill(tokens, input_pos, **kv_cache)
        # Prefill contract: returns ONLY KV cache tensors (no logits)
        self.assertNotIn("logits", prefill_out)
        self.assertIn("kv_cache_k_0", prefill_out)
        self.assertIn("kv_cache_v_0", prefill_out)
        self.assertIn("kv_cache_k_1", prefill_out)
        self.assertIn("kv_cache_v_1", prefill_out)
        self.assertEqual(
            mock_model.received_cache_shape,
            (1, 2, 2, cache_length, num_kv_heads, head_dim),
        )
        self.assertEqual(int(mock_model.received_index), 0)

        # Test decode
        decode_tokens = torch.zeros((1, 1), dtype=torch.int32)
        decode_pos = torch.tensor([4], dtype=torch.int32)
        decode_out = adapter.decode(decode_tokens, decode_pos, **kv_cache)
        # Decode contract: returns logits AND updated KV cache
        self.assertIn("logits", decode_out)
        self.assertEqual(decode_out["logits"].shape, (1, 1, 100))
        self.assertIn("kv_cache_k_0", decode_out)
        self.assertEqual(int(mock_model.received_index), 4)

    @unittest.skipUnless(
        keras.config.backend() == "torch", "Torch backend only"
    )
    def test_rank2_input_pos_and_2_tuple_return(self):
        import torch

        class Mock2TupleModel(torch.nn.Module):
            def call_with_cache(self, tokens, cache, cache_update_index):
                self.received_index = cache_update_index
                return torch.zeros((1, 1, 10)), cache

        mock_model = Mock2TupleModel()
        adapter = GenericKVAdapter(mock_model, num_layers=1, cache_length=8)
        tokens = torch.zeros((1, 2), dtype=torch.int32)
        input_pos = torch.tensor([[5, 6]], dtype=torch.int32)  # Rank 2
        kv_cache = {
            "kv_cache_k_0": torch.zeros((1, 8, 1, 8)),
            "kv_cache_v_0": torch.zeros((1, 8, 1, 8)),
        }
        res = adapter.prefill(tokens, input_pos, **kv_cache)
        self.assertIn("kv_cache_k_0", res)
        self.assertEqual(int(mock_model.received_index), 5)


class ResolveLiteRTLMSpecTest(testing.TestCase):
    def test_resolve_from_direct_spec(self):
        spec = LiteRTLMSpec(
            prefill_fn=lambda: None,
            decode_fn=lambda: None,
            prefill_sample_inputs={},
            decode_sample_inputs={},
            tokenizer_path="test.model",
            context_length=128,
            num_layers=1,
            num_kv_heads=1,
            head_dim=16,
        )
        resolved = resolve_litertlm_spec(spec=spec)
        self.assertIs(resolved, spec)

    def test_resolve_from_model_get_litertlm_spec(self):
        expected_spec = LiteRTLMSpec(
            prefill_fn=lambda: None,
            decode_fn=lambda: None,
            prefill_sample_inputs={},
            decode_sample_inputs={},
            tokenizer_path="tok.json",
            context_length=256,
            num_layers=2,
            num_kv_heads=1,
            head_dim=32,
        )

        class CustomModel:
            def get_litertlm_spec(self, **kwargs):
                return expected_spec

        model = CustomModel()
        resolved = resolve_litertlm_spec(model=model)
        self.assertIs(resolved, expected_spec)

    def test_resolve_from_explicit_kwargs(self):
        resolved = resolve_litertlm_spec(
            prefill_fn=lambda: None,
            decode_fn=lambda: None,
            prefill_sample_inputs={"tokens": None},
            decode_sample_inputs={"tokens": None},
            tokenizer_path="/tmp/tokenizer.model",
            context_length=1024,
            num_layers=8,
            num_kv_heads=4,
            head_dim=64,
            stop_token_ids=[2],
            model_type="custom_lm",
        )
        self.assertEqual(resolved.context_length, 1024)
        self.assertEqual(resolved.num_layers, 8)
        self.assertEqual(resolved.num_kv_heads, 4)
        self.assertEqual(resolved.head_dim, 64)
        self.assertEqual(resolved.stop_token_ids, [2])
        self.assertEqual(resolved.model_type, "custom_lm")

    def test_resolve_missing_required_raises_error(self):
        with self.assertRaisesRegex(
            ValueError, "Could not resolve `LiteRTLMSpec`"
        ):
            resolve_litertlm_spec(model=None)

    def test_bounds_validation(self):
        with self.assertRaisesRegex(ValueError, "context_length must be > 0"):
            LiteRTLMSpec(
                prefill_fn=lambda: None,
                decode_fn=lambda: None,
                prefill_sample_inputs={},
                decode_sample_inputs={},
                context_length=0,
                num_layers=1,
                num_kv_heads=1,
                head_dim=16,
            )
        with self.assertRaisesRegex(ValueError, "num_layers must be > 0"):
            LiteRTLMSpec(
                prefill_fn=lambda: None,
                decode_fn=lambda: None,
                prefill_sample_inputs={},
                decode_sample_inputs={},
                context_length=128,
                num_layers=-1,
                num_kv_heads=1,
                head_dim=16,
            )
        with self.assertRaisesRegex(ValueError, "kv_cache_layout must be"):
            LiteRTLMSpec(
                prefill_fn=lambda: None,
                decode_fn=lambda: None,
                prefill_sample_inputs={},
                decode_sample_inputs={},
                context_length=128,
                num_layers=1,
                num_kv_heads=1,
                head_dim=16,
                kv_cache_layout="INVALID",
            )


class SignatureMapBucketingTest(testing.TestCase):
    @unittest.skipUnless(
        keras.config.backend() == "torch", "Torch backend only"
    )
    def test_bucketing_signature_names(self):
        import torch

        sample_inputs_32 = {"tokens": torch.zeros((1, 32))}
        sample_inputs_64 = {"tokens": torch.zeros((1, 64))}
        sample_inputs_128 = {"tokens": torch.zeros((1, 128))}

        spec = LiteRTLMSpec(
            prefill_fn=lambda *a, **k: {},
            decode_fn=lambda *a, **k: {},
            prefill_sample_inputs=[
                sample_inputs_32,
                sample_inputs_64,
                sample_inputs_128,
            ],
            decode_sample_inputs={"tokens": torch.zeros((1, 1))},
            tokenizer_path="tok.model",
            context_length=128,
            num_layers=1,
            num_kv_heads=1,
            head_dim=16,
        )
        signatures = _build_signatures_map(spec, torch)
        self.assertIn("prefill_32", signatures)
        self.assertIn("prefill_64", signatures)
        self.assertIn("prefill_128", signatures)
        self.assertIn("decode", signatures)


class ExportLiteRTLMValidationTest(testing.TestCase):
    def test_filepath_validation(self):
        with self.assertRaisesRegex(ValueError, "ending in `\\.litertlm`"):
            export_litertlm(filepath="model.tflite")

    def test_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            real_file = os.path.join(tmpdir, "real.litertlm")
            with open(real_file, "w") as f:
                f.write("test")
            symlink_file = os.path.join(tmpdir, "symlink.litertlm")
            os.symlink(real_file, symlink_file)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                export_litertlm(filepath=symlink_file)

    @patch("keras.src.backend.backend")
    def test_backend_validation(self, mock_backend):
        mock_backend.return_value = "tensorflow"
        with self.assertRaisesRegex(
            ValueError, "only supported with PyTorch backend"
        ):
            export_litertlm(filepath="model.litertlm")

    @unittest.skipUnless(
        keras.config.backend() == "torch", "Torch backend only"
    )
    @patch("keras.src.export.litertlm.export_litert_via_torch")
    @patch("keras.src.export.litertlm._serialize_llm_metadata")
    def test_export_litertlm_mocked_end_to_end(
        self, mock_serialize_metadata, mock_export_litert
    ):
        import torch

        # Mock litert_lm_builder and litert_torch
        mock_builder_inst = MagicMock()
        mock_litert_lm_builder = MagicMock()
        mock_litert_lm_builder.LitertLmFileBuilder.return_value = (
            mock_builder_inst
        )
        mock_litert_torch = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            tok_file = os.path.join(tmpdir, "tokenizer.model")
            with open(tok_file, "w") as f:
                f.write("fake_tokenizer")

            spec = LiteRTLMSpec(
                prefill_fn=lambda *a, **k: {},
                decode_fn=lambda *a, **k: {},
                prefill_sample_inputs={"tokens": torch.zeros((1, 8))},
                decode_sample_inputs={"tokens": torch.zeros((1, 1))},
                tokenizer_path=tok_file,
                context_length=128,
                num_layers=1,
                num_kv_heads=1,
                head_dim=16,
            )

            with patch.dict(
                "sys.modules",
                {
                    "litert_torch": mock_litert_torch,
                    "litert_lm_builder": mock_litert_lm_builder,
                },
            ):
                out_path = os.path.join(tmpdir, "model.litertlm")
                result = export_litertlm(spec=spec, filepath=out_path)

                self.assertEqual(result, out_path)
                mock_export_litert.assert_called_once()
                mock_serialize_metadata.assert_called_once()
                mock_builder_inst.add_tflite_model.assert_called_once()
                mock_builder_inst.add_sentencepiece_tokenizer.assert_called_with(
                    tok_file
                )
                mock_builder_inst.build.assert_called_once()


class ModelExportFormatDispatchTest(testing.TestCase):
    @unittest.skipUnless(
        keras.config.backend() == "torch", "Torch backend only"
    )
    @patch("keras.src.export.litertlm.export_litertlm")
    def test_model_export_dispatches_to_litertlm(self, mock_export_litertlm):
        mock_export_litertlm.return_value = "model.litertlm"

        model = keras.Sequential([keras.layers.Dense(10, input_shape=(10,))])
        res = model.export("model.litertlm", format="litertlm", custom_arg=123)

        self.assertEqual(res, "model.litertlm")
        mock_export_litertlm.assert_called_once_with(
            model=model,
            filepath="model.litertlm",
            verbose=None,
            custom_arg=123,
        )
