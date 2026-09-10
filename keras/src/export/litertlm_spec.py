import pathlib
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from keras.src.api_export import keras_export


@keras_export(["keras.export.LiteRTLMConfig", "keras.export.LiteRTLMSpec"])
@dataclass
class LiteRTLMConfig:
    """Configuration for exporting a model to a LiteRT-LM `.litertlm` bundle.

    Keras core models (or external models) provide this configuration to
    configure the `prefill` and `decode` signatures, KV-cache structure,
    tokenizer asset, and metadata required by the LiteRT-LM runtime.

    Args:
        prefill_fn: Callable for the prefill forward pass. Must accept
            `(tokens, input_pos, **kv_cache)` or
            `(tokens, input_pos, stacked_cache)` and return the updated cache.
        decode_fn: Callable for the decode forward pass. Must accept
            `(tokens, input_pos, **kv_cache)` or
            `(tokens, input_pos, stacked_cache)` and return
            `(logits, updated_kv_cache)` or a dict with `"logits"` and KV.
        context_length: Maximum sequence length supported by the model cache.
        num_layers: Number of transformer/attention layers.
        num_kv_heads: Number of key-value attention heads.
        head_dim: Dimensionality of each attention head.
        prefill_sample_inputs: Optional dict or list of dicts of sample inputs
            for prefill. Auto-generated via `create_sample_inputs` if `None`.
        decode_sample_inputs: Optional dict of sample input tensors for decode.
            Auto-generated via `create_sample_inputs` if `None`.
        tokenizer_path: Optional string or Path to the tokenizer asset. Can be
            a SentencePiece model (`.model` or `.proto`) or a HuggingFace
            tokenizer (`tokenizer.json`).
        stop_token_ids: Optional list of token IDs (or list of token ID lists)
            that end generation.
        start_token_id: Optional start-of-sequence token ID.
        model_type: Model architecture identifier. Defaults to
            `"generic_model"`.
        backend_constraint: Optional LiteRT-LM backend constraint hint
            (e.g. `"cpu"`, `"gpu"`, `"npu"`).
        quant_config: Optional `litert_torch` quantization configuration
            (e.g. from `litert_torch.generative.quantize.quant_recipes`).
        extra_signatures: Optional dictionary of additional named signatures
            (e.g. separate `"vision_encoder"` or `"vision_adapter"`).
        jinja_prompt_template: Optional Jinja template for chat formatting.
        kv_cache_layout: Memory layout for KV tensors (`"BNTH"` or `"BTNH"`).
            Defaults to `"BNTH"` (canonical LiteRT-LM runtime layout).
    """

    prefill_fn: Callable[..., Any]
    decode_fn: Callable[..., Any]
    context_length: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    prefill_sample_inputs: Optional[
        Union[Dict[str, Any], List[Dict[str, Any]]]
    ] = None
    decode_sample_inputs: Optional[Dict[str, Any]] = None
    tokenizer_path: Optional[Union[str, pathlib.Path]] = None
    stop_token_ids: Optional[Union[List[int], List[List[int]]]] = None
    start_token_id: Optional[int] = None
    model_type: str = "generic_model"
    backend_constraint: Optional[str] = None
    quant_config: Optional[Any] = None
    extra_signatures: Optional[Dict[str, Any]] = None
    jinja_prompt_template: Optional[str] = None
    kv_cache_layout: str = "BNTH"

    def __post_init__(self):
        if self.context_length <= 0:
            raise ValueError(
                f"context_length must be > 0. Got: {self.context_length}"
            )
        if self.num_layers <= 0:
            raise ValueError(
                f"num_layers must be > 0. Got: {self.num_layers}"
            )
        if self.num_kv_heads <= 0:
            raise ValueError(
                f"num_kv_heads must be > 0. Got: {self.num_kv_heads}"
            )
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be > 0. Got: {self.head_dim}")
        if self.kv_cache_layout not in ("BNTH", "BTNH"):
            raise ValueError(
                f"kv_cache_layout must be 'BNTH' or 'BTNH'. "
                f"Got: {self.kv_cache_layout}"
            )

        if (
            self.prefill_sample_inputs is None
            or self.decode_sample_inputs is None
        ):
            prefill_gen, decode_gen = self.create_sample_inputs(
                context_length=self.context_length,
                num_layers=self.num_layers,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                kv_cache_layout=self.kv_cache_layout,
            )
            if self.prefill_sample_inputs is None:
                self.prefill_sample_inputs = prefill_gen
            if self.decode_sample_inputs is None:
                self.decode_sample_inputs = decode_gen

    @classmethod
    def create_sample_inputs(
        cls,
        context_length: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        prefill_buckets: Optional[List[int]] = None,
        kv_cache_layout: str = "BNTH",
        dtype: str = "float16",
    ) -> Tuple[Union[Dict[str, Any], List[Dict[str, Any]]], Dict[str, Any]]:
        """Construct standard sample input dictionaries for LiteRT-LM."""
        import torch

        if prefill_buckets is None:
            prefill_buckets = [128]

        torch_dtype = getattr(torch, dtype, torch.float16)

        def _make_kv():
            kv = {}
            for i in range(num_layers):
                if kv_cache_layout == "BNTH":
                    shape = (1, num_kv_heads, context_length, head_dim)
                else:
                    shape = (1, context_length, num_kv_heads, head_dim)
                kv[f"kv_cache_k_{i}"] = torch.zeros(shape, dtype=torch_dtype)
                kv[f"kv_cache_v_{i}"] = torch.zeros(shape, dtype=torch_dtype)
            return kv

        prefill_list = []
        for bucket_len in prefill_buckets:
            prefill_list.append(
                {
                    "tokens": torch.zeros((1, bucket_len), dtype=torch.int32),
                    "input_pos": torch.arange(0, bucket_len, dtype=torch.int32),
                    **_make_kv(),
                }
            )

        prefill_inputs = (
            prefill_list if len(prefill_list) > 1 else prefill_list[0]
        )

        decode_inputs = {
            "tokens": torch.zeros((1, 1), dtype=torch.int32),
            "input_pos": torch.zeros((1,), dtype=torch.int32),
            **_make_kv(),
        }
        return prefill_inputs, decode_inputs


# Backwards-compatibility alias
LiteRTLMSpec = LiteRTLMConfig
