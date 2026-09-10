"""Export Keras models to LiteRT-LM `.litertlm` Task Bundles."""

import os
import tempfile
from typing import Any
from typing import Dict
from typing import Optional

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.export.litert import export_litert_via_torch
from keras.src.export.litertlm_spec import LiteRTLMSpec
from keras.src.utils import io_utils


@keras_export("keras.export.export_litertlm")
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

    filepath = os.path.abspath(os.path.expanduser(str(filepath)))
    if not filepath.endswith(".litertlm"):
        raise ValueError(
            "LiteRT-LM export requires a filepath ending in `.litertlm`. "
            f"Received: filepath={filepath}"
        )

    if os.path.islink(filepath):
        raise ValueError(
            f"Destination filepath is a symbolic link: {filepath}. "
            "Writing to symlinks is disallowed for security."
        )

    dest_dir = os.path.dirname(filepath)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

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

    # Ensure model is in eval mode to prevent stochastic dropout during tracing
    if model is not None and hasattr(model, "eval") and callable(model.eval):
        model.eval()

    # Validate tokenizer path if provided
    if resolved_spec.tokenizer_path is not None:
        tok_path = str(resolved_spec.tokenizer_path)
        if not os.path.isfile(tok_path) or os.path.islink(tok_path):
            raise ValueError(
                f"Tokenizer path must be a valid regular file. Got: {tok_path}"
            )

    # Build signature map for LiteRT
    signatures = _build_signatures_map(resolved_spec, torch, model=model)

    quant_config = resolved_spec.quant_config or kwargs.get(
        "quant_config", None
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        tflite_path = os.path.join(temp_dir, "model.tflite")

        # Export multi-signature TFLite model via litert_torch chaining
        export_litert_via_torch(
            model=model,
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

        # Attach tokenizer if provided
        if resolved_spec.tokenizer_path is not None:
            tok_path = str(resolved_spec.tokenizer_path)
            if tok_path.endswith(".json"):
                builder.add_hf_tokenizer(tok_path)
            else:
                builder.add_sentencepiece_tokenizer(tok_path)

        builder.add_llm_metadata(meta_path)

        # Atomic write: stage to temp file and rename
        staged_bundle_path = os.path.join(temp_dir, "staged.litertlm")
        with open(staged_bundle_path, "wb") as f:
            builder.build(f)

        os.replace(staged_bundle_path, filepath)

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
        "context_length",
        "num_layers",
        "num_kv_heads",
        "head_dim",
    }
    if required_kwargs.issubset(kwargs.keys()):
        return LiteRTLMSpec(
            prefill_fn=kwargs["prefill_fn"],
            decode_fn=kwargs["decode_fn"],
            prefill_sample_inputs=kwargs["prefill_sample_inputs"],
            decode_sample_inputs=kwargs["decode_sample_inputs"],
            tokenizer_path=kwargs.get("tokenizer_path", None),
            context_length=int(kwargs["context_length"]),
            num_layers=int(kwargs["num_layers"]),
            num_kv_heads=int(kwargs["num_kv_heads"]),
            head_dim=int(kwargs["head_dim"]),
            stop_token_ids=kwargs.get("stop_token_ids", None),
            start_token_id=kwargs.get("start_token_id", None),
            model_type=kwargs.get("model_type", "generic_model"),
            backend_constraint=kwargs.get("backend_constraint", None),
            quant_config=kwargs.get("quant_config", None),
            extra_signatures=kwargs.get("extra_signatures", None),
            jinja_prompt_template=kwargs.get("jinja_prompt_template", None),
            kv_cache_layout=kwargs.get("kv_cache_layout", "BTNH"),
        )

    raise ValueError(
        "Could not resolve `LiteRTLMSpec` for LiteRT-LM export. Either:\n"
        "1) Pass a pre-constructed `LiteRTLMSpec` via `spec=...`\n"
        "2) Implement a `get_litertlm_spec(**kwargs)` method on model\n"
        "3) Pass the required arguments explicitly: `prefill_fn`, `decode_fn`, "
        "`prefill_sample_inputs`, `decode_sample_inputs`, `context_length`, "
        "`num_layers`, `num_kv_heads`, and `head_dim`."
    )


class ExportSignatureWrapper:
    """Helper module ensuring parameter sharing across signatures."""

    def __new__(cls, model, fn, torch):
        class _WrapperModule(torch.nn.Module):
            def __init__(self, m, f):
                super().__init__()
                if isinstance(m, torch.nn.Module):
                    self.model = m
                self.fn = f

            def forward(self, *args, **kwargs):
                return self.fn(*args, **kwargs)

        return _WrapperModule(model, fn)


def _build_signatures_map(
    spec: LiteRTLMSpec, torch, model=None
) -> Dict[str, Any]:
    """Construct the signature dictionary for multi-signature LiteRT."""
    signatures = {}

    prefill_fn = spec.prefill_fn
    decode_fn = spec.decode_fn

    # Wrap callables in torch.nn.Module to preserve parameter sharing
    if not isinstance(prefill_fn, torch.nn.Module):
        prefill_module = ExportSignatureWrapper(
            model, prefill_fn, torch
        ).eval()
    else:
        prefill_module = prefill_fn.eval()

    if not isinstance(decode_fn, torch.nn.Module):
        decode_module = ExportSignatureWrapper(model, decode_fn, torch).eval()
    else:
        decode_module = decode_fn.eval()

    def _get_seq_len(inputs):
        if isinstance(inputs, dict):
            if "tokens" in inputs and hasattr(inputs["tokens"], "shape"):
                return inputs["tokens"].shape[-1]
            if "input_pos" in inputs and hasattr(inputs["input_pos"], "shape"):
                return inputs["input_pos"].shape[-1]
        return None

    # Prefill signature(s) - handle bucketing with prefill_{seq_len} standard
    if isinstance(spec.prefill_sample_inputs, list):
        for idx, inputs in enumerate(spec.prefill_sample_inputs):
            s_len = _get_seq_len(inputs)
            sig_name = (
                f"prefill_{s_len}" if s_len is not None else f"prefill_{idx}"
            )
            signatures[sig_name] = (prefill_module, inputs)
    else:
        s_len = _get_seq_len(spec.prefill_sample_inputs)
        sig_name = f"prefill_{s_len}" if s_len is not None else "prefill"
        signatures[sig_name] = (prefill_module, spec.prefill_sample_inputs)

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
        if hasattr(meta.start_token, "token_ids"):
            meta.start_token.token_ids.ids.append(int(spec.start_token_id))
        elif hasattr(meta.start_token, "token_id"):
            meta.start_token.token_id = int(spec.start_token_id)

    for stop_id in spec.stop_token_ids or []:
        st = meta.stop_tokens.add()
        if hasattr(st, "token_ids"):
            st.token_ids.ids.append(int(stop_id))
        elif hasattr(st, "token_id"):
            st.token_id = int(stop_id)

    meta.max_num_tokens = int(spec.context_length)

    if spec.jinja_prompt_template:
        meta.jinja_prompt_template = str(spec.jinja_prompt_template)

    model_type = (
        getattr(spec, "model_type", "generic_model") or "generic_model"
    )
    if hasattr(meta, "llm_model_type"):
        if hasattr(meta.llm_model_type, "DESCRIPTOR"):
            valid_types = meta.llm_model_type.DESCRIPTOR.fields_by_name
            if model_type in valid_types:
                getattr(meta.llm_model_type, model_type).SetInParent()
            elif "generic_model" in valid_types:
                meta.llm_model_type.generic_model.SetInParent()
        elif hasattr(meta.llm_model_type, model_type):
            getattr(meta.llm_model_type, model_type).SetInParent()

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
        cache_update_index = input_pos.flatten()[0]
        res = self.model.call_with_cache(tokens, cache, cache_update_index)
        if len(res) == 3:
            logits, _, updated_cache = res
        else:
            logits, updated_cache = res
        # Prefill signature contract: return ONLY updated KV-cache tensors
        return self._unstack_kv_cache(updated_cache)

    def decode(self, tokens, input_pos, **kv_cache):
        import torch

        cache = self._stack_kv_cache(kv_cache, torch)
        cache_update_index = input_pos.flatten()[0]
        res = self.model.call_with_cache(tokens, cache, cache_update_index)
        if len(res) == 3:
            logits, _, updated_cache = res
        else:
            logits, updated_cache = res
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
            outputs[f"kv_cache_k_{i}"] = cache[:, i, 0, ...].contiguous()
            outputs[f"kv_cache_v_{i}"] = cache[:, i, 1, ...].contiguous()
        return outputs

