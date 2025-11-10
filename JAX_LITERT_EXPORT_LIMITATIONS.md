# JAX Backend LiteRT Export Limitations

## Summary

**Exporting large models (>200M parameters) from JAX backend to LiteRT is currently not possible** due to fundamental protobuf size limitations in the TensorFlow serialization infrastructure.

## Root Cause

All JAX→TFLite conversion paths require creating serialized intermediate representations that have a hard 2GB size limit:

1. **jax2tf → TensorFlow GraphDef**: Creates a GraphDef proto that exceeds 2GB for large models
2. **jax2tf (native_serialization) → StableHLO**: Creates StableHLO tensor proto that exceeds 2GB  
3. **experimental_from_jax → HLO**: HLO proto serialization fails for large models
4. **from_concrete_functions**: Still requires `_freeze_concrete_function()` which hits the same limits

## Testing Results

| Approach | llama3.2_1b (1.5B params) | Custom (191M params) |
|----------|---------------------------|----------------------|
| jax2tf + enable_xla=False | ✗ GraphDef 4.9GB | ✓ Works |
| jax2tf + native_serialization=True | ✗ StableHLO 4.9GB | ✓ Works |
| experimental_from_jax | ✗ HLO serialization failed | ✓ Works |
| from_concrete_functions (Orbax pattern) | ✗ Invalid GraphDef | ✓ Works |
| **TensorFlow backend** | **✓ Works (557MB TFLite)** | **✓ Works** |

## The 2GB Protobuf Limit

The limit is enforced in multiple places:

1. **Python-level check** (`tensorflow/python/framework/ops.py:2364`):
   ```python
   if bytesize >= (1 << 31) or bytesize < 0:
     raise ValueError("GraphDef cannot be larger than 2GB.")
   ```

2. **C++ protobuf library**: Uses 32-bit signed integer for message size (INT_MAX = 2^31-1 bytes ≈ 2GB)

3. **Cannot be overridden**: This is a fundamental limitation of the protobuf format, not a configuration setting

## Why TensorFlow Backend Works

`TFLiteConverter.from_keras_model()` with TensorFlow backend:
- Operates directly on in-memory TensorFlow graphs
- Converts to TFLite without intermediate protobuf serialization
- Successfully exports llama3.2_1b (1.5B params) → 557MB TFLite file

## Recommended Solution

**For production LiteRT export of large models (>200M params):**

```bash
# Use TensorFlow backend
export KERAS_BACKEND=tensorflow
python your_export_script.py
```

The keras-hub export code should:
1. Detect JAX backend + large model combination
2. Provide clear error message explaining the limitation
3. Suggest switching to TensorFlow backend

## Theoretical Fix (Not Implemented)

To fix this at the source would require:

1. Modify `jax2tf.convert()` to support chunked graph generation
2. Integrate TensorFlow's proto_splitter library  
3. Update TFLiteConverter to read chunked graphs
4. Modify protobuf library or use alternative serialization

**Estimated effort**: Multiple engineer-months; requires deep expertise in JAX, TensorFlow, and protobuf internals.

## References

- Test results: `/Users/hellorahul/Projects/keras/test_llama3_litert_export.py`
- Orbax test: `/Users/hellorahul/Projects/keras/test_orbax_llama3.py`
- TensorFlow proto_splitter: `tensorflow/tools/proto_splitter/`
- Size limit check: `tensorflow/python/framework/ops.py:2364`
