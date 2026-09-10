import pathlib
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from keras.src.api_export import keras_export


@keras_export("keras.export.LiteRTLMSpec")
@dataclass
class LiteRTLMSpec:
    """Specification for exporting a model to a LiteRT-LM `.litertlm` bundle.

    Keras core models (or external models) provide this specification to
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
        prefill_sample_inputs: Dict or list of Dicts of sample input tensors
            for prefill. Providing a list of dicts enables bucketing.
        decode_sample_inputs: Dict of sample input tensors for decode.
        context_length: Maximum sequence length supported by the model cache.
        num_layers: Number of transformer/attention layers.
        num_kv_heads: Number of key-value attention heads.
        head_dim: Dimensionality of each attention head.
        tokenizer_path: Optional string or Path to the tokenizer asset. Can be
            a SentencePiece model (`.model` or `.proto`) or a HuggingFace
            tokenizer (`tokenizer.json`).
        stop_token_ids: Optional list of token IDs that end generation.
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
        kv_cache_layout: Memory layout for KV tensors (`"BTNH"` or `"BNTH"`).
    """

    prefill_fn: Callable[..., Any]
    decode_fn: Callable[..., Any]
    prefill_sample_inputs: Union[Dict[str, Any], List[Dict[str, Any]]]
    decode_sample_inputs: Dict[str, Any]
    context_length: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    tokenizer_path: Optional[Union[str, pathlib.Path]] = None
    stop_token_ids: Optional[List[int]] = None
    start_token_id: Optional[int] = None
    model_type: str = "generic_model"
    backend_constraint: Optional[str] = None
    quant_config: Optional[Any] = None
    extra_signatures: Optional[Dict[str, Any]] = None
    jinja_prompt_template: Optional[str] = None
    kv_cache_layout: str = "BTNH"

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
        if self.kv_cache_layout not in ("BTNH", "BNTH"):
            raise ValueError(
                f"kv_cache_layout must be 'BTNH' or 'BNTH'. "
                f"Got: {self.kv_cache_layout}"
            )
