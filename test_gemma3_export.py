import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import keras_hub

print("Loading gemma3_1b with load_weights=False...")
model = keras_hub.models.Gemma3CausalLM.from_preset(
    "gemma3_1b",
    load_weights=False,
)
print("Model loaded.")

# Check if accessing weights causes allocation
weights = model.weights
print(f"Number of weights: {len(weights)}")

# Try to access value on first few weights
for i, w in enumerate(weights[:5]):
    print(f"Weight {i}: shape={w.shape}, dtype={w.dtype}")
    try:
        val = w.value
        print(f"  value dtype={val.dtype}, size={val.dtype.size}")
    except Exception as e:
        print(f"  ERROR accessing value: {e}")

# Check total size estimation
try:
    import math
    total = 0
    for w in weights:
        shape = w.shape
        if shape is None or None in shape:
            continue
        val = w.value
        total += math.prod(shape) * val.dtype.size
    print(f"Total bytes: {total}")
except Exception as e:
    print(f"ERROR computing size: {e}")

# Try export
print("Exporting to ONNX...")
try:
    model.export("/tmp/gemma3_test.onnx", format="onnx")
    print("Export succeeded!")
except Exception as e:
    print(f"Export failed: {e}")
