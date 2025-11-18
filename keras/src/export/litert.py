import logging
import os
import traceback

from keras.src import backend
from keras.src import tree
from keras.src.export.export_utils import get_input_signature
from keras.src.export.export_utils import make_input_spec
from keras.src.export.export_utils import make_tf_tensor_spec
from keras.src.utils import io_utils
from keras.src.utils import summary_utils
from keras.src.utils.module_utils import litert
from keras.src.utils.module_utils import tensorflow as tf


def export_litert(
    model,
    filepath,
    input_signature=None,
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

    def _infer_dict_input_signature(self):
        """Infer input signature from a model with dict inputs.

        This reads the actual shapes and dtypes from model._inputs_struct.

        Returns:
            dict or None: Dictionary mapping input names to InputSpec, or None
        """
        # Check _inputs_struct first (preserves dict structure)
        if hasattr(self.model, "_inputs_struct") and isinstance(
            self.model._inputs_struct, dict
        ):
            return {
                name: make_input_spec(inp)
                for name, inp in self.model._inputs_struct.items()
            }

        # Fall back to model.inputs if it's a dict
        if hasattr(self.model, "inputs") and isinstance(
            self.model.inputs, dict
        ):
            return {
                name: make_input_spec(inp)
                for name, inp in self.model.inputs.items()
            }

        return None

    def export(self, filepath):
        """Exports the Keras model to a TFLite file.

        Args:
            filepath: Output path for the exported model

        Returns:
            Path to exported model
        """
        # 1. Resolve / infer input signature
        if self.input_signature is None:
            # Try dict-specific inference first (for models with dict inputs)
            dict_signature = self._infer_dict_input_signature()
            if dict_signature is not None:
                self.input_signature = dict_signature
            else:
                # Fall back to standard inference
                self.input_signature = get_input_signature(self.model)

        # 3. Handle dictionary inputs by creating an adapter
        # Check if we have dict inputs that need adaptation
        has_dict_inputs = isinstance(self.input_signature, dict)

        if has_dict_inputs:
            # Create adapter model that converts list to dict
            adapted_model = self._create_dict_adapter(self.input_signature)

            # Convert dict signature to list for TFLite conversion
            # The adapter will handle the dict->list conversion
            input_signature_list = list(self.input_signature.values())

            # Use adapted model and list signature for conversion
            model_to_convert = adapted_model
            signature_for_conversion = input_signature_list
        else:
            # No dict inputs - use model as-is
            model_to_convert = self.model
            signature_for_conversion = self.input_signature

        # Store original model reference for later use
        original_model = self.model

        # Temporarily replace self.model with the model to convert
        self.model = model_to_convert

        try:
            # 4. Convert the model to TFLite.
            tflite_model = self._convert_to_tflite(signature_for_conversion)
        finally:
            # Restore original model
            self.model = original_model

        # 4. Save the initial TFLite model to the specified file path.
        if not filepath.endswith(".tflite"):
            raise ValueError(
                "The LiteRT export requires the filepath to end with "
                "'.tflite'. Got: {filepath}"
            )

        with open(filepath, "wb") as f:
            f.write(tflite_model)

        return filepath

    def _create_dict_adapter(self, input_signature_dict):
        """Create an adapter model that converts list inputs to dict inputs.

        This adapter allows models expecting dictionary inputs to be exported
        to TFLite format (which only supports positional/list inputs).

        Args:
            input_signature_dict: Dictionary mapping input names to InputSpec

        Returns:
            A Functional model that accepts list inputs and converts to dict
        """
        io_utils.print_msg(
            f"Creating adapter for dictionary inputs: "
            f"{list(input_signature_dict.keys())}"
        )

        input_keys = list(input_signature_dict.keys())

        # Create Input layers for TFLite (list-based)
        input_layers = []
        for name in input_keys:
            spec = input_signature_dict[name]
            input_layer = tf.keras.layers.Input(
                shape=spec.shape[1:],  # Remove batch dimension
                dtype=spec.dtype,
                name=name,
            )
            input_layers.append(input_layer)

        # Create dict from list inputs
        inputs_dict = {
            name: layer for name, layer in zip(input_keys, input_layers)
        }

        # Call the original model with dict inputs
        outputs = self.model(inputs_dict)

        # Build as Functional model (list inputs -> dict -> model -> output)
        adapted_model = tf.keras.Model(inputs=input_layers, outputs=outputs)

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
        current_backend = backend.backend()
        
        # JAX backend requires jax2tf conversion
        if current_backend == "jax":
            if self.verbose:
                io_utils.print_msg(
                    "JAX backend detected. Using jax2tf conversion path..."
                )
            return self._convert_jax_model(input_signature)
        
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

        except Exception:
            return self._convert_with_wrapper(input_signature)

    def _convert_jax_model(self, input_signature):
        """Converts a JAX backend model to TFLite using jax2tf.

        Args:
            input_signature: Input signature specification

        Returns:
            A bytes object containing the serialized TFLite model.
        
        Raises:
            RuntimeError: If model is too large for jax2tf conversion
        """
        try:
            # Import jax2tf for JAX to TensorFlow conversion
            from jax.experimental import jax2tf
        except ImportError:
            raise ImportError(
                "jax2tf is required for JAX backend LiteRT export. "
                "Please install JAX with: pip install jax jaxlib"
            )

        # Estimate model size and warn if it may exceed protobuf limit
        num_params = summary_utils.count_params(self.model.trainable_variables)
        if self.verbose:
            io_utils.print_msg(
                f"Model has {num_params:,} trainable parameters"
            )
        
        # Warn if model is large (>200M params typically hits 2GB limit)
        if num_params > 200_000_000:
            io_utils.print_msg(
                f"WARNING: Large model detected ({num_params:,} parameters). "
                f"JAX backend export may fail due to protobuf 2GB limit. "
                f"Consider using TensorFlow backend for models >200M parameters."
            )

        if self.verbose:
            io_utils.print_msg(
                "Converting JAX model to TensorFlow using jax2tf..."
            )

        # Prepare input signature
        if not isinstance(input_signature, (list, tuple)):
            input_signature = [input_signature]

        from keras.src.export.export_utils import make_tf_tensor_spec
        
        # Convert input signature to tensor specs, handling structures
        tensor_specs = []
        for spec in input_signature:
            if isinstance(spec, (list, tuple)):
                # Handle list/tuple of specs
                tensor_specs.extend([make_tf_tensor_spec(s) for s in spec])
            elif isinstance(spec, dict):
                # Handle dict of specs - flatten to list
                tensor_specs.extend([make_tf_tensor_spec(s) for s in spec.values()])
            else:
                # Single spec
                tensor_specs.append(make_tf_tensor_spec(spec))

        # Create a wrapper function that calls the model
        def model_fn(*args):
            """Wrapper function for JAX model."""
            if len(args) == 1:
                return self.model(args[0])
            else:
                return self.model(list(args))
        
        # Build polymorphic_shapes for dynamic batch dimension
        # Format: 'b, ...' where 'b' is the batch dimension variable
        polymorphic_shapes = []
        for spec in tensor_specs:
            shape = spec.shape
            if shape.rank is None or shape.rank == 0:
                polymorphic_shapes.append(None)
            else:
                # Create shape string with 'b' for batch dimension and concrete sizes for others
                shape_str = ', '.join(
                    'b' if (i == 0 and dim is None) else str(dim) if dim is not None else '_'
                    for i, dim in enumerate(shape.as_list())
                )
                polymorphic_shapes.append(shape_str)
        
        # Convert the JAX function to TensorFlow using jax2tf
        if self.verbose:
            io_utils.print_msg("Converting JAX function using jax2tf...")
            io_utils.print_msg(f"Polymorphic shapes: {polymorphic_shapes}")
        
        tf_fn = jax2tf.convert(
            model_fn,
            enable_xla=False,
            polymorphic_shapes=polymorphic_shapes,
        )
        
        # Wrap in tf.Module for proper variable tracking
        class JAXModelWrapper(tf.Module):
            """Wrapper for jax2tf converted function."""
            
            def __init__(self, tf_fn, model):
                super().__init__()
                self._tf_fn = tf_fn
                
                # Track all variables from the Keras model
                with self.name_scope:
                    for i, var in enumerate(model.variables):
                        setattr(self, f"model_var_{i}", var)

            @tf.function
            def __call__(self, *args):
                """Entry point for the exported model."""
                return self._tf_fn(*args)

        wrapper = JAXModelWrapper(tf_fn, self.model)

        # Get concrete function
        concrete_func = wrapper.__call__.get_concrete_function(*tensor_specs)

        if self.verbose:
            io_utils.print_msg(
                "Converting concrete function to TFLite format..."
            )

        # Try conversion with different strategies
        conversion_strategies = [
            {
                "experimental_enable_resource_variables": False,
                "name": "without resource variables",
            },
            {
                "experimental_enable_resource_variables": True,
                "name": "with resource variables",
            },
        ]

        last_error = None
        for strategy in conversion_strategies:
            try:
                converter = tf.lite.TFLiteConverter.from_concrete_functions(
                    [concrete_func], trackable_obj=wrapper
                )
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS,
                ]
                converter.experimental_enable_resource_variables = strategy[
                    "experimental_enable_resource_variables"
                ]

                # Apply any additional converter settings from kwargs
                self._apply_converter_kwargs(converter)

                if self.verbose:
                    io_utils.print_msg(
                        f"Trying conversion {strategy['name']}..."
                    )

                tflite_model = converter.convert()

                if self.verbose:
                    io_utils.print_msg(
                        f"JAX model conversion successful {strategy['name']}!"
                    )

                return tflite_model

            except Exception as e:
                last_error = e
                if self.verbose:
                    io_utils.print_msg(
                        f"Conversion failed {strategy['name']}: {e}"
                    )
                
                # Check if this is a protobuf size error
                error_msg = str(e).lower()
                if any(
                    indicator in error_msg
                    for indicator in ["2gb", "size limit", "too large", "exceeds"]
                ):
                    raise RuntimeError(
                        f"Model is too large for JAX backend export. "
                        f"The model has {num_params:,} parameters. "
                        f"JAX backend export is limited by the protobuf 2GB size limit. "
                        f"For large models (>200M parameters), please use the TensorFlow backend instead. "
                        f"Original error: {e}"
                    )
                continue

        # If all strategies fail, raise the last error with helpful message
        raise RuntimeError(
            f"All conversion strategies failed for JAX model. "
            f"If this is a large model ({num_params:,} parameters), "
            f"the protobuf 2GB limit may be the issue. "
            f"Consider using TensorFlow backend for large models. "
            f"Last error: {last_error}"
        )

    def _convert_with_wrapper(self, input_signature):
        """Converts the model to TFLite using SavedModel as intermediate.

        This fallback method is used when direct Keras conversion fails.
        It uses TensorFlow's SavedModel format as an intermediate step.

        Returns:
            A bytes object containing the serialized TFLite model.
        """
        # Normalize input_signature to list format for concrete function
        if isinstance(input_signature, dict):
            # For multi-input models with dict signature, convert to
            # ordered list
            if hasattr(self.model, "inputs") and len(self.model.inputs) > 1:
                input_signature_list = []
                for input_layer in self.model.inputs:
                    input_name = input_layer.name
                    if input_name not in input_signature:
                        raise ValueError(
                            f"Missing input '{input_name}' in input_signature. "
                            f"Model expects inputs: "
                            f"{[inp.name for inp in self.model.inputs]}, "
                            f"but input_signature only has: "
                            f"{list(input_signature.keys())}"
                        )
                    input_signature_list.append(input_signature[input_name])
                input_signature = input_signature_list
            else:
                # Single-input model with dict signature
                input_signature = [input_signature]
        elif not isinstance(input_signature, (list, tuple)):
            input_signature = [input_signature]

        # Convert to TensorSpec
        tensor_specs = [make_tf_tensor_spec(spec) for spec in input_signature]

        # Get concrete function from the model
        @tf.function
        def model_fn(*args):
            return self.model(*args)

        concrete_func = model_fn.get_concrete_function(*tensor_specs)

        # Convert using concrete function
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [concrete_func], self.model
        )
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
