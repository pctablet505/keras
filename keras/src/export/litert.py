from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import tree
from keras.src.export.export_utils import get_input_signature
from keras.src.utils import io_utils
from keras.src.utils.module_utils import tensorflow as tf


def export_litert(
    model,
    filepath,
    input_signature=None,
    verbose=None,
    **kwargs,
):
    """Export the model as a LiteRT artifact for inference.

    Args:
        model: The Keras model to export.
        filepath: The path to save the exported artifact.
        input_signature: Optional input signature specification. If
            `None`, it will be inferred.
        **kwargs: Additional keyword arguments passed to the exporter.
    """
    if backend.backend() == "torch":
        return export_litert_via_torch(
            model,
            filepath,
            input_signature=input_signature,
            verbose=verbose,
            **kwargs,
        )

    if backend.backend() != "tensorflow":
        raise ImportError(
            "The LiteRT export API is currently only available "
            "with the TensorFlow and PyTorch backends."
        )

    exporter = LiteRTExporter(
        model=model,
        input_signature=input_signature,
        **kwargs,
    )
    exporter.export(filepath)
    io_utils.print_msg(f"Saved artifact at '{filepath}'.")


class LiteRTExporter:
    """Exporter for the LiteRT (TFLite) format.

    This class handles the conversion of Keras models for LiteRT runtime and
    generates a `.tflite` model file. For efficient inference on mobile and
    embedded devices, it creates a single callable signature based on the
    model's `call()` method.
    """

    def __init__(
        self,
        model,
        input_signature=None,
        **kwargs,
    ):
        """Initialize the LiteRT exporter.

        Args:
            model: The Keras model to export
            input_signature: Input signature specification (e.g., TensorFlow
                TensorSpec or list of TensorSpec)
            **kwargs: Additional export parameters
        """
        self.model = model
        self.input_signature = input_signature
        self.kwargs = kwargs

    def export(self, filepath):
        """Exports the Keras model to a TFLite file.

        Args:
            filepath: Output path for the exported model

        Returns:
            Path to exported model
        """
        # 1. Resolve / infer input signature
        if self.input_signature is None:
            # Use the standard get_input_signature which handles all model types
            # and preserves nested structures (dicts, lists, etc.)
            self.input_signature = get_input_signature(self.model)

        # 2. Determine input structure and create adapter if needed
        # There are 3 cases:
        # Case 1: Single input (not nested)
        # Case 2: Flat list of inputs (list where flattened == original)
        # Case 3: Nested structure (dicts, nested lists, etc.)

        # Special handling for Functional models: get_input_signature wraps
        # the structure in a list, so unwrap it for analysis
        input_struct = self.input_signature
        if (
            isinstance(self.input_signature, list)
            and len(self.input_signature) == 1
        ):
            input_struct = self.input_signature[0]

        if not tree.is_nested(input_struct):
            # Case 1: Single input - use as-is
            model_to_convert = self.model
            signature_for_conversion = self.input_signature
        elif isinstance(input_struct, list) and len(input_struct) == len(
            tree.flatten(input_struct)
        ):
            # Case 2: Flat list of inputs - use as-is
            model_to_convert = self.model
            signature_for_conversion = self.input_signature
        else:
            # Case 3: Nested structure (dict, nested lists, etc.)
            # Create adapter model that converts flat list to nested structure
            adapted_model = self._create_nested_inputs_adapter(input_struct)

            # Flatten signature for TFLite conversion
            signature_for_conversion = tree.flatten(input_struct)

            # Use adapted model and flat list signature for conversion
            model_to_convert = adapted_model

        # Store original model reference for later use
        original_model = self.model

        # Temporarily replace self.model with the model to convert
        self.model = model_to_convert

        try:
            # Convert the model to TFLite.
            tflite_model = self._convert_to_tflite(signature_for_conversion)
        finally:
            # Restore original model
            self.model = original_model

        # Save the TFLite model to the specified file path.
        if not filepath.endswith(".tflite"):
            raise ValueError(
                f"The LiteRT export requires the filepath to end with "
                f"'.tflite'. Got: {filepath}"
            )

        with open(filepath, "wb") as f:
            f.write(tflite_model)

        return filepath

    def _create_nested_inputs_adapter(self, input_signature_struct):
        """Create an adapter model that converts flat list inputs to nested
        structure.

        This adapter allows models expecting nested inputs (dicts, lists, etc.)
        to be exported to TFLite format (which only supports positional/list
        inputs).

        Args:
            input_signature_struct: Nested structure of InputSpecs (dict, list,
                etc.)

        Returns:
            A Functional model that accepts flat list inputs and converts to
            nested
        """
        # Get flat paths to preserve names and print input mapping
        paths_and_specs = tree.flatten_with_path(input_signature_struct)
        paths = [".".join(str(e) for e in p) for p, v in paths_and_specs]
        io_utils.print_msg(f"Creating adapter for inputs: {paths}")

        # Create Input layers for TFLite (flat list-based)
        input_layers = []
        for path, spec in paths_and_specs:
            # Extract the input name from spec or path
            name = (
                spec.name
                if hasattr(spec, "name") and spec.name
                else (str(path[-1]) if path else "input")
            )

            input_layer = layers.Input(
                shape=spec.shape[1:],  # Remove batch dimension
                dtype=spec.dtype,
                name=name,
            )
            input_layers.append(input_layer)

        # Reconstruct the nested structure from flat list
        inputs_structure = tree.pack_sequence_as(
            input_signature_struct, input_layers
        )

        # Call the original model with nested inputs
        outputs = self.model(inputs_structure)

        # Build as Functional model (flat list inputs -> nested -> model ->
        # output)
        adapted_model = models.Model(inputs=input_layers, outputs=outputs)

        # Preserve the original model's variables
        adapted_model._variables = self.model.variables
        adapted_model._trainable_variables = self.model.trainable_variables
        adapted_model._non_trainable_variables = (
            self.model.non_trainable_variables
        )

        return adapted_model

    def _convert_to_tflite(self, input_signature):
        """Converts the Keras model to TFLite format.

        Returns:
            A bytes object containing the serialized TFLite model.
        """
        # Try direct conversion first for all models
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            # Keras 3 only supports resource variables
            converter.experimental_enable_resource_variables = True

            # Apply any additional converter settings from kwargs
            self._apply_converter_kwargs(converter)

            tflite_model = converter.convert()

            return tflite_model

        except Exception as e:
            # If direct conversion fails, raise the error with helpful message
            raise RuntimeError(
                f"Direct TFLite conversion failed. This may be due to model "
                f"complexity or unsupported operations. Error: {e}"
            ) from e

    def _apply_converter_kwargs(self, converter):
        """Apply additional converter settings from kwargs.

        Args:
            converter: tf.lite.TFLiteConverter instance to configure

        Raises:
            ValueError: If any kwarg is not a valid converter attribute
        """
        for attr, value in self.kwargs.items():
            if attr == "target_spec" and isinstance(value, dict):
                # Handle nested target_spec settings
                for spec_key, spec_value in value.items():
                    if hasattr(converter.target_spec, spec_key):
                        setattr(converter.target_spec, spec_key, spec_value)
                    else:
                        raise ValueError(
                            f"Unknown target_spec attribute '{spec_key}'"
                        )
            elif hasattr(converter, attr):
                setattr(converter, attr, value)
            else:
                raise ValueError(f"Unknown converter attribute '{attr}'")


def export_litert_via_torch(
    model, filepath, input_signature=None, verbose=None, **kwargs
):
    """Export Keras model to LiteRT via PyTorch backend."""
    try:
        import litert_torch
        import torch
    except ImportError:
        raise ImportError(
            "To export to LiteRT with the PyTorch backend, "
            "you must install the `litert-torch` package. "
            "Install via: pip install litert-torch"
        )

    from keras.src.export.export_utils import convert_spec_to_tensor

    original_devices = {}
    _move_model_to_cpu(model, original_devices, torch)

    from keras.src.backend.torch.core import device_scope

    with device_scope("cpu"):
        _register_litert_decompositions(torch, litert_torch)

        if input_signature is None:
            input_signature = get_input_signature(model)

        sample_inputs = tree.map_structure(
            lambda x: convert_spec_to_tensor(x, replace_none_number=1),
            input_signature,
        )
        sample_inputs = tree.map_structure(
            lambda t: t.cpu() if hasattr(t, "cpu") else t,
            sample_inputs,
        )
        sample_inputs = tuple(sample_inputs)

        if hasattr(model, "eval"):
            model.eval()

        litert_torch_kwargs = _prepare_litert_kwargs(kwargs, litert_torch)

        try:
            try:
                edge_model = litert_torch.convert(
                    model, sample_inputs, **litert_torch_kwargs
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to convert PyTorch model to LiteRT. "
                    f"Common causes: unsupported operations, dynamic shapes, "
                    f"or complex control flow. Original error: {e}"
                ) from e

            edge_model.export(filepath)
        finally:
            _restore_model_devices(model, original_devices, torch)

    if verbose:
        io_utils.print_msg(f"Saved LiteRT model to {filepath}")

    return filepath


def _prepare_litert_kwargs(kwargs, litert_torch):
    """Prepare litert_torch conversion kwargs from user-provided arguments."""
    litert_torch_kwargs = {}

    valid_litert_torch_args = {
        "strict_export",
        "quant_config",
        "dynamic_shapes",
        "_ai_edge_converter_flags",
        "_saved_model_dir",
    }
    for k, v in kwargs.items():
        if k in valid_litert_torch_args:
            litert_torch_kwargs[k] = v

    if "optimizations" in kwargs and "quant_config" not in litert_torch_kwargs:
        quant_cfg = _create_quant_config_from_optimizations(
            kwargs["optimizations"], litert_torch
        )
        if quant_cfg is not None:
            litert_torch_kwargs["quant_config"] = quant_cfg

    return litert_torch_kwargs


def _move_model_to_cpu(model, original_devices, torch):
    """Move all model tensors to CPU for portable export."""
    try:
        for v in model.variables:
            if hasattr(v, "value") and hasattr(v.value, "data"):
                dev = str(v.value.device)
                if dev != "cpu":
                    original_devices[("var", v.path)] = dev
                    v.value.data = v.value.data.to("cpu")
    except (AttributeError, TypeError):
        pass

    try:
        for name, p in model.named_parameters():
            if p.device.type != "cpu":
                original_devices[("param", name)] = str(p.device)
                p.data = p.data.to("cpu")
        for name, b in model.named_buffers():
            if b.device.type != "cpu":
                original_devices[("buffer", name)] = str(b.device)
                b.data = b.data.to("cpu")
    except (AttributeError, RuntimeError):
        pass

    try:
        for layer in model._flatten_layers():
            for attr_name in list(vars(layer)):
                obj = getattr(layer, attr_name, None)
                if (
                    isinstance(obj, torch.Tensor)
                    and not isinstance(obj, torch.nn.Parameter)
                    and obj.device.type != "cpu"
                ):
                    key = ("attr", f"{layer.name}.{attr_name}")
                    original_devices[key] = str(obj.device)
                    setattr(layer, attr_name, obj.to("cpu"))
    except (AttributeError, RuntimeError):
        pass


def _create_quant_config_from_optimizations(optimizations, litert_torch):
    """Translate TFLite optimizations to litert_torch QuantConfig."""
    if not optimizations:
        return None

    try:
        from litert_torch.quantize.pt2e_quantizer import PT2EQuantizer
        from litert_torch.quantize.pt2e_quantizer import (
            get_symmetric_quantization_config,
        )
        from litert_torch.quantize.quant_config import QuantConfig
    except ImportError:
        io_utils.print_msg(
            "Warning: litert_torch quantization modules not available. "
            "Skipping quantization."
        )
        return None

    try:
        import tensorflow as tf

        optimize_default = tf.lite.Optimize.DEFAULT
        optimize_size = getattr(tf.lite.Optimize, "OPTIMIZE_FOR_SIZE", None)
        optimize_latency = getattr(
            tf.lite.Optimize, "OPTIMIZE_FOR_LATENCY", None
        )
    except (ImportError, AttributeError):
        return None

    has_default = optimize_default in optimizations
    has_size = optimize_size and optimize_size in optimizations
    has_latency = optimize_latency and optimize_latency in optimizations

    if has_default or has_size or has_latency:
        is_dynamic = has_default and not (has_size or has_latency)
        is_per_channel = has_latency or has_size

        quant_config_obj = get_symmetric_quantization_config(
            is_per_channel=is_per_channel,
            is_dynamic=is_dynamic,
            is_qat=False,
        )

        quantizer = PT2EQuantizer()
        quantizer.set_global(quant_config_obj)

        return QuantConfig(pt2e_quantizer=quantizer)

    return None


def _restore_model_devices(model, original_devices, torch):
    """Restore model tensors to their original devices after export."""
    if not original_devices:
        return

    try:
        for v in model.variables:
            key = ("var", v.path)
            if key in original_devices:
                v.value.data = v.value.data.to(original_devices[key])
    except (AttributeError, TypeError):
        pass

    try:
        for name, p in model.named_parameters():
            key = ("param", name)
            if key in original_devices:
                p.data = p.data.to(original_devices[key])
        for name, b in model.named_buffers():
            key = ("buffer", name)
            if key in original_devices:
                b.data = b.data.to(original_devices[key])
    except (AttributeError, RuntimeError):
        pass

    try:
        for layer in model._flatten_layers():
            for attr_name in list(vars(layer)):
                key = ("attr", f"{layer.name}.{attr_name}")
                if key in original_devices:
                    obj = getattr(layer, attr_name, None)
                    if isinstance(obj, torch.Tensor):
                        setattr(
                            layer,
                            attr_name,
                            obj.to(original_devices[key]),
                        )
    except (AttributeError, RuntimeError):
        pass


def _register_litert_decompositions(torch, litert_torch):
    """Register decompositions for operations unsupported by litert_torch."""
    from litert_torch.fx_infra import decomp as litert_decomp

    pre_convert = litert_decomp.pre_convert_decomp()

    mps_sdpa = getattr(
        torch.ops.aten,
        "_scaled_dot_product_attention_math_for_mps",
        None,
    )
    if mps_sdpa is not None:
        mps_sdpa_default = getattr(mps_sdpa, "default", None)
        if mps_sdpa_default is not None and mps_sdpa_default not in pre_convert:
            non_mps_sdpa = getattr(
                torch.ops.aten._scaled_dot_product_attention_math,
                "default",
                None,
            )
            if non_mps_sdpa is not None:
                core_decomps = torch._decomp.core_aten_decompositions()
                if non_mps_sdpa in core_decomps:
                    litert_decomp.add_pre_convert_decomp(
                        mps_sdpa_default, core_decomps[non_mps_sdpa]
                    )

    mean_dim_op = torch.ops.aten.mean.dim
    if mean_dim_op not in pre_convert:

        def _mean_dim_with_dtype(self, dim, keepdim=False, *, dtype=None):
            if dtype is not None:
                self = self.to(dtype)
            curr_dim = dim
            if curr_dim is None:
                curr_dim = list(range(self.ndim))
            elif isinstance(curr_dim, int):
                curr_dim = [curr_dim]
            count = 1
            for d in curr_dim:
                count *= self.shape[d]
            return (
                torch.ops.aten.sum.dim_IntList(self, curr_dim, keepdim=keepdim)
                / count
            )

        litert_decomp.add_pre_convert_decomp(mean_dim_op, _mean_dim_with_dtype)

    repeat_interleave_op = getattr(
        torch.ops.aten.repeat_interleave, "Tensor", None
    )
    if (
        repeat_interleave_op is not None
        and repeat_interleave_op not in pre_convert
    ):

        def _repeat_interleave_decomp(repeats, output_size=None):
            if output_size is None:
                output_size = torch.ops.aten.sum.default(repeats)
            boundaries = torch.ops.aten.cumsum.default(
                repeats, dim=0, dtype=repeats.dtype
            )
            out_indices = torch.ops.aten.arange.start_step(
                0, output_size, 1, dtype=repeats.dtype, device=repeats.device
            )
            return torch.ops.aten.searchsorted.default(
                boundaries, out_indices, right=False
            )

        litert_decomp.add_pre_convert_decomp(
            repeat_interleave_op, _repeat_interleave_decomp
        )

    repeat_interleave_self_int = getattr(
        torch.ops.aten.repeat_interleave, "self_int", None
    )
    if (
        repeat_interleave_self_int is not None
        and repeat_interleave_self_int not in pre_convert
    ):

        def _repeat_interleave_self_int_decomp(
            self, repeats, dim=None, *, output_size=None
        ):
            if dim is None:
                self = self.flatten()
                dim = 0
            if dim < 0:
                dim = self.ndim + dim
            x = self.unsqueeze(dim + 1)
            expand_shape = [-1] * x.ndim
            expand_shape[dim + 1] = repeats
            x = x.expand(expand_shape)
            shape = list(self.shape)
            shape[dim] = shape[dim] * repeats
            return x.reshape(shape)

        litert_decomp.add_pre_convert_decomp(
            repeat_interleave_self_int, _repeat_interleave_self_int_decomp
        )
