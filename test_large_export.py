import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np

print("Building large model (~2GB weights)...")
# 1B params in float32 = 4GB, too much. Use bfloat16 (2 bytes) with 800M params = 1.6GB
# Or float32 with 400M params = 1.6GB
inputs = keras.Input(shape=(1024,), dtype="float32")
x = keras.layers.Dense(400_000, dtype="float32")(inputs)  # ~400M params = 1.6GB
x = keras.layers.Dense(10, dtype="float32")(x)
model = keras.Model(inputs, x)

# Build by calling
model(np.zeros((1, 1024), dtype="float32"))

print(f"Number of weights: {len(model.weights)}")
total = sum(int(np.prod(w.shape)) * 4 for w in model.weights)
print(f"Approx total bytes: {total}")

print("Exporting to ONNX...")
try:
    model.export("/tmp/large_test.onnx", format="onnx")
    print("Export succeeded!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Export failed: {e}")
