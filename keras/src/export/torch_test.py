"""Tests for PyTorch ExportedProgram exporting utilities."""

import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src import tree
from keras.src.testing.test_utils import named_product


class CustomModel(models.Model):
    def __init__(self, layer_list):
        super().__init__()
        self.layer_list = layer_list

    def call(self, input):
        output = input
        for layer in self.layer_list:
            output = layer(output)
        return output


def get_model(type="sequential", input_shape=(10,), layer_list=None):
    layer_list = layer_list or [
        layers.Dense(10, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(1, activation="sigmoid"),
    ]
    if type == "sequential":
        model = models.Sequential(layer_list)
        model.build(input_shape=(None,) + input_shape)
        return model
    if type == "functional":
        input = output = tree.map_shape_structure(layers.Input, input_shape)
        for layer in layer_list:
            output = layer(output)
        return models.Model(inputs=input, outputs=output)
    if type == "subclass":
        model = CustomModel(layer_list)
        model.build(input_shape=(None,) + input_shape)
        dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
        _ = model(dummy_input)
        return model
    if type == "multi_input":
        input1 = layers.Input(shape=input_shape, name="input1")
        input2 = layers.Input(shape=input_shape, name="input2")
        x1 = layers.Dense(10, activation="relu")(input1)
        x2 = layers.Dense(10, activation="relu")(input2)
        combined = layers.concatenate([x1, x2])
        output = layers.Dense(1, activation="sigmoid")(combined)
        return models.Model(inputs=[input1, input2], outputs=output)
    if type == "multi_output":
        inputs = layers.Input(shape=input_shape)
        shared = layers.Dense(20, activation="relu")(inputs)
        output1 = layers.Dense(1, activation="sigmoid", name="output1")(shared)
        output2 = layers.Dense(3, activation="softmax", name="output2")(shared)
        return models.Model(inputs=inputs, outputs=[output1, output2])
    raise ValueError(f"Unknown model type: {type}")


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return np.array(x)
    return np.array(x)


def _convert_to_numpy(structure):
    return tree.map_structure(_to_numpy, structure)


def _get_torch_device():
    from keras.src.backend.torch.core import get_device

    return get_device()


def _to_torch_tensor(np_array):
    import torch

    return torch.tensor(np_array).to(_get_torch_device())


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="`export_torch` only supports the PyTorch backend.",
)
class ExportTorchTest(testing.TestCase):
    ATOL = 1e-5
    RTOL = 1e-5

    def _verify_export_and_inference(
        self, model, ref_input, filepath=None, **export_kwargs
    ):
        import torch

        if filepath is None:
            filepath = os.path.join(self.get_temp_dir(), "model.pt2")

        ref_output = _convert_to_numpy(model(ref_input))

        model.export(filepath, format="torch", **export_kwargs)
        self.assertTrue(os.path.exists(filepath))

        loaded_program = torch.export.load(filepath)
        loaded_model = loaded_program.module()

        torch_input = tree.map_structure(_to_torch_tensor, ref_input)
        loaded_output = _convert_to_numpy(loaded_model(torch_input))

        self.assertAllClose(
            ref_output, loaded_output, atol=self.ATOL, rtol=self.RTOL
        )

        return ref_output, loaded_output

    @parameterized.named_parameters(
        named_product(model_type=["sequential", "functional"])
    )
    def test_standard_model_export(self, model_type):
        model = get_model(model_type)
        ref_input = np.random.normal(size=(1, 10)).astype("float32")
        self._verify_export_and_inference(model, ref_input)

    def test_export_subclass_model(self):
        model = get_model("subclass")
        ref_input = np.random.normal(size=(1, 10)).astype("float32")
        self._verify_export_and_inference(model, ref_input)

    def test_model_with_multiple_inputs(self):
        model = get_model("multi_input")
        ref_input1 = np.random.normal(size=(1, 10)).astype("float32")
        ref_input2 = np.random.normal(size=(1, 10)).astype("float32")
        ref_input = [ref_input1, ref_input2]
        self._verify_export_and_inference(model, ref_input)

    def test_multi_output_model_export(self):
        import torch

        model = get_model("multi_output")
        ref_input = np.random.normal(size=(1, 10)).astype("float32")
        ref_output = _convert_to_numpy(model(ref_input))

        temp_filepath = os.path.join(self.get_temp_dir(), "multi_out.pt2")
        model.export(temp_filepath, format="torch")
        self.assertTrue(os.path.exists(temp_filepath))

        loaded_program = torch.export.load(temp_filepath)
        loaded_model = loaded_program.module()
        loaded_output = _convert_to_numpy(
            loaded_model(_to_torch_tensor(ref_input))
        )

        # Handle both tuple/list and single output cases
        if isinstance(loaded_output, (tuple, list)):
            for ref_out, loaded_out in zip(ref_output, loaded_output):
                self.assertAllClose(
                    ref_out, loaded_out, atol=self.ATOL, rtol=self.RTOL
                )
        else:
            self.assertAllClose(
                ref_output, loaded_output, atol=self.ATOL, rtol=self.RTOL
            )

    def test_export_with_custom_input_signature(self):
        import torch

        model = get_model("sequential")
        ref_input = np.random.normal(size=(1, 10)).astype("float32")
        model(ref_input)  # Build the model

        input_signature = [layers.InputSpec(shape=(1, 10), dtype="float32")]
        temp_filepath = os.path.join(self.get_temp_dir(), "custom_sig.pt2")

        model.export(
            temp_filepath,
            format="torch",
            input_signature=input_signature,
        )
        self.assertTrue(os.path.exists(temp_filepath))

        loaded_program = torch.export.load(temp_filepath)
        loaded_model = loaded_program.module()
        test_input = torch.randn(1, 10).to(_get_torch_device())
        output = loaded_model(test_input)
        self.assertEqual(_to_numpy(output).shape[-1], 1)

        ref_output = _convert_to_numpy(model(_to_numpy(test_input)))
        self.assertAllClose(
            ref_output, _to_numpy(output), atol=self.ATOL, rtol=self.RTOL
        )

    def test_export_with_verbose(self):
        model = get_model("sequential")
        ref_input = np.random.normal(size=(1, 10)).astype("float32")
        model(ref_input)

        temp_filepath = os.path.join(self.get_temp_dir(), "verbose_model.pt2")
        model.export(temp_filepath, format="torch", verbose=True)
        self.assertTrue(os.path.exists(temp_filepath))

        self._verify_export_and_inference(
            model, ref_input, filepath=temp_filepath
        )

    def test_export_error_handling(self):
        model = get_model("sequential")
        ref_input = np.random.normal(size=(1, 10)).astype("float32")
        model(ref_input)

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model.pt2")

        with self.assertRaises(ValueError):
            model.export(temp_filepath, format="invalid_format")

    def test_export_invalid_filepath(self):
        model = get_model("sequential")
        ref_input = np.random.normal(size=(1, 10)).astype("float32")
        model(ref_input)

        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model.txt")

        # Should raise ValueError for wrong extension
        with self.assertRaises(ValueError):
            model.export(temp_filepath, format="torch")

    def test_model_with_input_structure(self):
        import torch

        ref_input_arr = np.random.normal(size=(1, 10)).astype("float32")

        input1 = layers.Input(shape=(10,), name="input_1")
        input2 = layers.Input(shape=(10,), name="input_2")
        output = layers.Add()([input1, input2])
        model_list = models.Model(inputs=[input1, input2], outputs=output)

        ref_input_list = [ref_input_arr, ref_input_arr * 2]
        list_filepath = os.path.join(self.get_temp_dir(), "list_input.pt2")
        self._verify_export_and_inference(
            model_list, ref_input_list, filepath=list_filepath
        )

        input_x = layers.Input(shape=(10,), name="x")
        input_y = layers.Input(shape=(10,), name="y")
        output = layers.Add()([input_x, input_y])
        model_dict = models.Model(
            inputs={"x": input_x, "y": input_y}, outputs=output
        )

        ref_input_dict = {"x": ref_input_arr, "y": ref_input_arr * 2}
        temp_filepath = os.path.join(self.get_temp_dir(), "dict_input.pt2")
        ref_output = _convert_to_numpy(model_dict(ref_input_dict))

        model_dict.export(temp_filepath, format="torch")
        self.assertTrue(os.path.exists(temp_filepath))

        loaded_program = torch.export.load(temp_filepath)
        loaded_model = loaded_program.module()
        loaded_output = _convert_to_numpy(
            loaded_model(
                {
                    "x": _to_torch_tensor(ref_input_dict["x"]),
                    "y": _to_torch_tensor(ref_input_dict["y"]),
                }
            )
        )
        self.assertAllClose(
            ref_output, loaded_output, atol=self.ATOL, rtol=self.RTOL
        )

    def test_model_with_named_inputs(self):
        import torch

        input_x = layers.Input(shape=(10,), name="input_x")
        input_y = layers.Input(shape=(10,), name="input_y")
        output = layers.Add()([input_x, input_y])
        model = models.Model(
            inputs={"x": input_x, "y": input_y}, outputs=output
        )

        temp_filepath = os.path.join(self.get_temp_dir(), "named_inputs.pt2")
        model.export(temp_filepath, format="torch")

        loaded_program = torch.export.load(temp_filepath)
        signature = loaded_program.graph_signature

        input_names = [spec.arg.name for spec in signature.input_specs]
        self.assertTrue(
            any("input_x" in name or "x" in name for name in input_names),
            f"Expected 'input_x' or 'x' in {input_names}",
        )
        self.assertTrue(
            any("input_y" in name or "y" in name for name in input_names),
            f"Expected 'input_y' or 'y' in {input_names}",
        )

        ref_input = {
            "x": np.random.normal(size=(1, 10)).astype("float32"),
            "y": np.random.normal(size=(1, 10)).astype("float32"),
        }
        self._verify_export_and_inference(
            model, ref_input, filepath=temp_filepath
        )

    def test_export_with_batch_normalization(self):
        model = models.Sequential(
            [
                layers.Dense(16, input_shape=(10,)),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dense(8),
                layers.BatchNormalization(),
                layers.Dense(1),
            ]
        )

        ref_input = np.random.normal(size=(1, 10)).astype("float32")
        # Ensure model is built and in eval mode for consistent behavior
        model(ref_input, training=False)
        self._verify_export_and_inference(model, ref_input)

    def test_export_functional_with_residual(self):
        inputs = layers.Input(shape=(10,))
        x = layers.Dense(16, activation="relu")(inputs)
        residual = layers.Dense(16)(inputs)
        x = layers.Add()([x, residual])
        x = layers.Activation("relu")(x)
        outputs = layers.Dense(1)(x)
        model = models.Model(inputs=inputs, outputs=outputs)

        ref_input = np.random.normal(size=(1, 10)).astype("float32")
        self._verify_export_and_inference(model, ref_input)

    def test_export_with_concrete_shapes(self):
        import torch

        model = models.Sequential([layers.Dense(1, input_shape=(5,))])
        temp_filepath = os.path.join(self.get_temp_dir(), "concrete_shapes.pt2")

        # Provide concrete input signature
        from keras.src.layers.input_spec import InputSpec

        input_sig = [InputSpec(shape=(1, 5), dtype="float32")]

        model.export(temp_filepath, format="torch", input_signature=input_sig)

        loaded_program = torch.export.load(temp_filepath)
        self.assertIsNotNone(loaded_program.graph_signature)

    def test_export_with_none_in_signature(self):
        model = models.Sequential([layers.Dense(1, input_shape=(5,))])
        temp_filepath = os.path.join(self.get_temp_dir(), "none_batch.pt2")

        # torch.export will replace None with a concrete value (1 by default)
        from keras.src.layers.input_spec import InputSpec

        input_sig = [InputSpec(shape=(None, 5), dtype="float32")]

        model.export(temp_filepath, format="torch", input_signature=input_sig)
        self.assertTrue(os.path.exists(temp_filepath))
