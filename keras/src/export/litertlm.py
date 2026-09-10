"""Export Keras models to LiteRT-LM `.litertlm` Task Bundles."""

import os
import tempfile
from typing import Any
from typing import Dict
from typing import Optional

from keras.src import backend
from keras.src.export.litert import export_litert_via_torch
from keras.src.export.litertlm_spec import LiteRTLMSpec
from keras.src.utils import io_utils


def export_litertlm(
    model=None,
    filepath: str = "model.litertlm",
    spec: Optional[LiteRTLMSpec] = None,
    verbose: Optional[bool] = None,
    **kwargs,
):
    """Export a model to a LiteRT-LM `.litertlm` Task Bundle.

    This function exports a model with `prefill` and `decode` signatures
    required by the on-device LiteRT-LM runtime, packages the tokenizer asset,
    and attaches an `LlmMetadata` protobuf into the final `.litertlm` container.

    Args:
        model: Optional Keras model instance. Can be `None` if an explicit
            `spec` is provided.
        filepath: String or Path. Destination path for the `.litertlm` file.
        spec: Optional `LiteRTLMSpec` defining the prefill/decode callables,
            sample inputs, tokenizer asset, and metadata. If `None`, it is
            inferred from `model` or remaining keyword arguments.
        verbose: Optional boolean. Whether to print progress messages.
        **kwargs: Additional parameters passed to `resolve_litertlm_spec` or
            the underlying `litert_torch.convert()`.

    Returns:
        The output filepath.
    """
    if backend.backend() != "torch":
        raise ValueError(
            "LiteRT-LM export is currently only supported with PyTorch "
            f"backend. Current backend: {backend.backend()}."
        )

    filepath = str(filepath)
    if not filepath.endswith(".litertlm"):
        raise ValueError(
            "LiteRT-LM export requires a filepath ending in `.litertlm`. "
            f"Received: filepath={filepath}"
        )

    # Check for required external packages
    try:
        import litert_torch  # noqa: F401
        import torch
    except ImportError:
        raise ImportError(
            "LiteRT-LM export requires `litert-torch`. "
            "Install it via: pip install litert-torch"
        )

    try:
        import litert_lm_builder
    except ImportError:
        raise ImportError(
            "LiteRT-LM export requires `litert-lm-builder`. "
            "Install it via: pip install litert-lm-builder"
        )

    # Resolve LiteRTLMSpec
    resolved_spec = resolve_litertlm_spec(model, spec=spec, **kwargs)

    # Build signature map for LiteRT
    signatures = _build_signatures_map(resolved_spec, torch)

    quant_config = resolved_spec.quant_config or kwargs.get(
        "quant_config", None
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        tflite_path = os.path.join(temp_dir, "model.tflite")

        # Export multi-signature TFLite model via litert_torch chaining
        export_litert_via_torch(
            model=None,
            filepath=tflite_path,
            signatures=signatures,
            quant_config=quant_config,
            verbose=False,
        )

        # Build metadata protobuf
        meta_path = os.path.join(temp_dir, "llm_metadata.pb")
        _serialize_llm_metadata(resolved_spec, meta_path)

        # Package Task Bundle via litert-lm-builder
        builder = litert_lm_builder.LitertLmFileBuilder()
        builder.add_system_metadata(
            litert_lm_builder.Metadata(
                key="Authors",
                value="Keras",
                dtype=litert_lm_builder.DType.STRING,
            )
        )
        builder.add_tflite_model(
            tflite_path,
            litert_lm_builder.TfLiteModelType.PREFILL_DECODE,
            backend_constraint=resolved_spec.backend_constraint,
        )

        # Attach tokenizer
        tok_path = str(resolved_spec.tokenizer_path)
        if tok_path.endswith(".json"):
            builder.add_hf_tokenizer(tok_path)
        else:
            builder.add_sentencepiece_tokenizer(tok_path)

        builder.add_llm_metadata(meta_path)

        with open(filepath, "wb") as f:
            builder.build(f)

    if verbose:
        io_utils.print_msg(f"Saved LiteRT-LM bundle to '{filepath}'.")

    return filepath


def resolve_litertlm_spec(
    model=None, spec: Optional[LiteRTLMSpec] = None, **kwargs
) -> LiteRTLMSpec:
    """Resolve a `LiteRTLMSpec` from model, spec, or keyword arguments."""
    # 1. Directly provided spec
    if spec is not None and isinstance(spec, LiteRTLMSpec):
        return spec
    if "spec" in kwargs and isinstance(kwargs["spec"], LiteRTLMSpec):
        return kwargs["spec"]

    # 2. Model provides its own spec via get_litertlm_spec()
    if (
        model is not None
        and hasattr(model, "get_litertlm_spec")
        and callable(model.get_litertlm_spec)
    ):
        return model.get_litertlm_spec(**kwargs)

    # 3. Explicit kwargs passed by user
    required_kwargs = {
        "prefill_fn",
        "decode_fn",
        "prefill_sample_inputs",
        "decode_sample_inputs",
        "tokenizer_path",
        "context_length",
    }
    if required_kwargs.issubset(kwargs.keys()):
        return LiteRTLMSpec(
            prefill_fn=kwargs["prefill_fn"],
            decode_fn=kwargs["decode_fn"],
            prefill_sample_inputs=kwargs["prefill_sample_inputs"],
            decode_sample_inputs=kwargs["decode_sample_inputs"],
            tokenizer_path=kwargs["tokenizer_path"],
            context_length=int(kwargs["context_length"]),
            num_layers=int(kwargs.get("num_layers", 1)),
            num_kv_heads=int(kwargs.get("num_kv_heads", 1)),
            head_dim=int(kwargs.get("head_dim", 64)),
            stop_token_ids=kwargs.get("stop_token_ids", None),
            start_token_id=kwargs.get("start_token_id", None),
            model_type=kwargs.get("model_type", "generic_model"),
            backend_constraint=kwargs.get("backend_constraint", None),
            quant_config=kwargs.get("quant_config", None),
            extra_signatures=kwargs.get("extra_signatures", None),
        )

    # 4. Duck-typing for models with call_with_cache
    if model is not None and hasattr(model, "call_with_cache"):
        return _build_spec_from_causal_lm(model, **kwargs)

    raise ValueError(
        "Could not resolve `LiteRTLMSpec` for LiteRT-LM export. Either:\n"
        "1) Pass a pre-constructed `LiteRTLMSpec` via `spec=...`\n"
        "2) Implement a `get_litertlm_spec(**kwargs)` method on model\n"
        "3) Pass the required arguments explicitly: `prefill_fn`, `decode_fn`, "
        "`prefill_sample_inputs`, `decode_sample_inputs`, `tokenizer_path`, "
        "and `context_length`."
    )


def _build_signatures_map(spec: LiteRTLMSpec, torch) -> Dict[str, Any]:
    """Construct the signature dictionary for multi-signature LiteRT."""
    signatures = {}

    prefill_fn = spec.prefill_fn
    decode_fn = spec.decode_fn

    # Wrap callables in torch.nn.Module if needed
    if not isinstance(prefill_fn, torch.nn.Module):

        class _PrefillModule(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, *args, **kwargs):
                return self.fn(*args, **kwargs)

        prefill_module = _PrefillModule(prefill_fn).eval()
    else:
        prefill_module = prefill_fn.eval()

    if not isinstance(decode_fn, torch.nn.Module):

        class _DecodeModule(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, *args, **kwargs):
                return self.fn(*args, **kwargs)

        decode_module = _DecodeModule(decode_fn).eval()
    else:
        decode_module = decode_fn.eval()

    # Prefill signature(s) - handle bucketing
    if isinstance(spec.prefill_sample_inputs, list):
        if len(spec.prefill_sample_inputs) == 1:
            signatures["prefill"] = (
                prefill_module,
                spec.prefill_sample_inputs[0],
            )
        else:
            for idx, inputs in enumerate(spec.prefill_sample_inputs):
                seq_len = None
                if isinstance(inputs, dict):
                    if "tokens" in inputs and hasattr(
                        inputs["tokens"], "shape"
                    ):
                        seq_len = inputs["tokens"].shape[-1]
                    elif "input_pos" in inputs and hasattr(
                        inputs["input_pos"], "shape"
                    ):
                        seq_len = inputs["input_pos"].shape[-1]
                sig_name = (
                    f"prefill_{seq_len}"
                    if seq_len is not None
                    else f"prefill_{idx}"
                )
                signatures[sig_name] = (prefill_module, inputs)
    else:
        signatures["prefill"] = (prefill_module, spec.prefill_sample_inputs)

    # Decode signature
    signatures["decode"] = (decode_module, spec.decode_sample_inputs)

    # Extra auxiliary signatures (e.g. vision encoder)
    if spec.extra_signatures:
        signatures.update(spec.extra_signatures)

    return signatures


def _serialize_llm_metadata(spec: LiteRTLMSpec, path: str):
    """Serialize an `LlmMetadata` protobuf file."""
    from litert_lm_builder.litertlm_builder import llm_metadata_pb2

    meta = llm_metadata_pb2.LlmMetadata()

    if spec.start_token_id is not None:
        meta.start_token.token_ids.ids.append(int(spec.start_token_id))

    for stop_id in spec.stop_token_ids or []:
        meta.stop_tokens.add().token_ids.ids.append(int(stop_id))

    meta.max_num_tokens = int(spec.context_length)

    model_type = (
        getattr(spec, "model_type", "generic_model") or "generic_model"
    )
    if hasattr(meta.llm_model_type, model_type):
        getattr(meta.llm_model_type, model_type).SetInParent()
    else:
        meta.llm_model_type.generic_model.SetInParent()

    with open(path, "wb") as f:
        f.write(meta.SerializeToString())


class GenericKVAdapter:
    """Standard adapter bridging flat per-layer KV tensors with model call.

    Stacks flat `kv_cache_k_{i}` and `kv_cache_v_{i}` into the standard Keras
    cache format `[batch, num_layers, 2, cache_length, num_kv_heads, head_dim]`,
    invokes `model.call_with_cache`, and unstacks back to flat outputs.
    """

    def __init__(self, model, num_layers: int, cache_length: int):
        self.model = model
        self.num_layers = num_layers
        self.cache_length = cache_length

    def prefill(self, tokens, input_pos, **kv_cache):
        import torch

        cache = self._stack_kv_cache(kv_cache, torch)
        cache_update_index = input_pos[0]
        logits, _, updated_cache = self.model.call_with_cache(
            tokens, cache, cache_update_index
        )
        # Prefill signature contract: return ONLY updated KV-cache tensors
        return self._unstack_kv_cache(updated_cache)

    def decode(self, tokens, input_pos, **kv_cache):
        import torch

        cache = self._stack_kv_cache(kv_cache, torch)
        cache_update_index = input_pos.reshape(())
        logits, _, updated_cache = self.model.call_with_cache(
            tokens, cache, cache_update_index
        )
        outputs = self._unstack_kv_cache(updated_cache)
        outputs["logits"] = logits
        return outputs

    def _stack_kv_cache(self, kv_cache, torch):
        k_list = [kv_cache[f"kv_cache_k_{i}"] for i in range(self.num_layers)]
        v_list = [kv_cache[f"kv_cache_v_{i}"] for i in range(self.num_layers)]
        k_stack = torch.stack(k_list, dim=1)
        v_stack = torch.stack(v_list, dim=1)
        return torch.stack([k_stack, v_stack], dim=2)

    def _unstack_kv_cache(self, cache):
        outputs = {}
        for i in range(self.num_layers):
            outputs[f"kv_cache_k_{i}"] = cache[:, i, 0, ...]
            outputs[f"kv_cache_v_{i}"] = cache[:, i, 1, ...]
        return outputs


def _build_spec_from_causal_lm(model, **kwargs) -> LiteRTLMSpec:
    """Helper to auto-construct a spec from a model with call_with_cache."""
    import torch

    backbone = getattr(model, "backbone", model)
    num_layers = getattr(
        backbone, "num_layers", getattr(model, "num_layers", None)
    )
    if num_layers is None:
        raise ValueError(
            "Could not determine `num_layers` from model or backbone."
        )

    cache_length = getattr(
        backbone,
        "max_sequence_length",
        getattr(model, "max_sequence_length", None),
    )
    if cache_length is None and hasattr(model, "preprocessor"):
        cache_length = getattr(model.preprocessor, "sequence_length", None)
    if cache_length is None:
        cache_length = kwargs.get("context_length", 2048)

    num_kv_heads = getattr(
        backbone,
        "num_key_value_heads",
        getattr(
            backbone, "num_heads", getattr(backbone, "num_query_heads", 1)
        ),
    )

    head_dim = getattr(backbone, "head_dim", None)
    if head_dim is None:
        hidden_dim = getattr(backbone, "hidden_dim", None)
        num_qh = getattr(
            backbone, "num_query_heads", getattr(backbone, "num_heads", None)
        )
        if hidden_dim is not None and num_qh is not None and num_qh > 0:
            head_dim = hidden_dim // num_qh
    if head_dim is None:
        head_dim = 64

    # Resolve tokenizer path
    tokenizer_path = kwargs.get("tokenizer_path", None)
    if tokenizer_path is None and hasattr(model, "preprocessor"):
        tok = getattr(model.preprocessor, "tokenizer", None)
        if tok is not None:
            tokenizer_path = getattr(
                tok, "proto", getattr(tok, "model_path", None)
            )
            if tokenizer_path is None and hasattr(tok, "assets_dir"):
                cand = os.path.join(tok.assets_dir, "tokenizer.model")
                if os.path.exists(cand):
                    tokenizer_path = cand

    if tokenizer_path is None:
        raise ValueError(
            "Could not automatically detect `tokenizer_path` from model. "
            "Please pass `tokenizer_path=...` explicitly to export."
        )

    # Stop token IDs
    stop_token_ids = kwargs.get("stop_token_ids", None)
    if stop_token_ids is None and hasattr(model, "preprocessor"):
        tok = getattr(model.preprocessor, "tokenizer", None)
        if tok is not None:
            stop_ids = []
            if getattr(tok, "end_token_id", None) is not None:
                stop_ids.append(int(tok.end_token_id))
            if getattr(tok, "end_token2_id", None) is not None:
                stop_ids.append(int(tok.end_token2_id))
            stop_token_ids = stop_ids

    adapter = GenericKVAdapter(
        model, num_layers=num_layers, cache_length=cache_length
    )

    # Sample inputs
    prefill_seq_len = kwargs.get("prefill_seq_len", cache_length)
    if isinstance(prefill_seq_len, int):
        prefill_seq_lens = [prefill_seq_len]
    else:
        prefill_seq_lens = sorted(set(prefill_seq_len))

    prefill_sample_inputs = []
    for s_len in prefill_seq_lens:
        sample = {
            "tokens": torch.zeros((1, s_len), dtype=torch.int32),
            "input_pos": torch.arange(s_len, dtype=torch.int32),
        }
        for i in range(num_layers):
            sample[f"kv_cache_k_{i}"] = torch.zeros(
                (1, cache_length, num_kv_heads, head_dim), dtype=torch.float32
            )
            sample[f"kv_cache_v_{i}"] = torch.zeros(
                (1, cache_length, num_kv_heads, head_dim), dtype=torch.float32
            )
        prefill_sample_inputs.append(sample)

    decode_sample = {
        "tokens": torch.zeros((1, 1), dtype=torch.int32),
        "input_pos": torch.zeros((1,), dtype=torch.int32),
    }
    for i in range(num_layers):
        decode_sample[f"kv_cache_k_{i}"] = torch.zeros(
            (1, cache_length, num_kv_heads, head_dim), dtype=torch.float32
        )
        decode_sample[f"kv_cache_v_{i}"] = torch.zeros(
            (1, cache_length, num_kv_heads, head_dim), dtype=torch.float32
        )

    return LiteRTLMSpec(
        prefill_fn=adapter.prefill,
        decode_fn=adapter.decode,
        prefill_sample_inputs=(
            prefill_sample_inputs
            if len(prefill_sample_inputs) > 1
            else prefill_sample_inputs[0]
        ),
        decode_sample_inputs=decode_sample,
        tokenizer_path=tokenizer_path,
        context_length=cache_length,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        stop_token_ids=stop_token_ids or [],
        backend_constraint=kwargs.get("backend_constraint", None),
        quant_config=kwargs.get("quant_config", None),
    )
