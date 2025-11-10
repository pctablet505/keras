"""
Test if Orbax JaxModule approach works with llama3.2_1b.

This tests whether the official Orbax Export + JaxModule pattern can handle
large models like llama3.2_1b, or if it also hits the same serialization limits.
"""

import os
import sys

KERAS_PATH = "/Users/hellorahul/Projects/keras"
KERAS_HUB_PATH = "/Users/hellorahul/Projects/keras-hub"
sys.path.insert(0, KERAS_PATH)
sys.path.insert(0, KERAS_HUB_PATH)

os.environ["KERAS_BACKEND"] = "jax"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import keras_hub
import numpy as np
import jax.numpy as jnp

print("=" * 70)
print("Testing Orbax JaxModule with llama3.2_1b")
print("=" * 70)

# Create llama3.2_1b model
print("\n1. Creating llama3.2_1b model...")
backbone = keras_hub.models.Llama3Backbone.from_preset("llama3.2_1b")
params_count = backbone.count_params()
print(f"   Model created: {params_count:,} parameters ({params_count/1e6:.1f}M)")

# Extract params and create apply function
print("\n2. Extracting params and creating apply function...")

# Get all variables as JAX arrays
all_vars = backbone.trainable_variables + backbone.non_trainable_variables
params = [v.numpy() for v in all_vars]

def apply_fn(params_list, inputs):
    """Apply function that takes params and inputs."""
    # Assign params temporarily
    for var, param in zip(all_vars, params_list):
        var.assign(param)
    return backbone(inputs, training=False)

print(f"   Extracted {len(params)} parameters")

# Try to create JaxModule with Orbax
print("\n3. Creating JaxModule with Orbax...")

try:
    from orbax.export import JaxModule, constants
    import tensorflow as tf
    
    # Create JaxModule (this does jax2tf conversion internally)
    jax_module = JaxModule(
        params=params,
        apply_fn=apply_fn,
        input_polymorphic_shape='(batch, seq)',
    )
    
    print("   ✓ JaxModule created successfully!")
    
    # Now try to get concrete function for TFLite conversion
    print("\n4. Getting concrete function...")
    print(f"   Available methods: {list(jax_module.methods.keys())}")
    
    # Use DEFAULT_METHOD_KEY
    method_key = constants.DEFAULT_METHOD_KEY
    concrete_fn = jax_module.methods[method_key].get_concrete_function(
        {
            'token_ids': tf.TensorSpec(shape=(1, 8), dtype=tf.int32),
            'padding_mask': tf.TensorSpec(shape=(1, 8), dtype=tf.int32),
        }
    )
    
    print("   ✓ Concrete function obtained!")
    
    # Try TFLite conversion
    print("\n5. Converting to TFLite with from_concrete_functions...")
    
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    
    print("   Running conversion...")
    tflite_model = converter.convert()
    
    print(f"\n✓ SUCCESS! TFLite model size: {len(tflite_model) / (1024**2):.2f} MB")
    
except Exception as e:
    print(f"\n✗ FAILED: {type(e).__name__}: {str(e)[:200]}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
