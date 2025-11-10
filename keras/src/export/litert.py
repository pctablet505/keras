import logging
import os
import traceback

from keras.src import backend
from keras.src import tree
from keras.src.utils import io_utils
from keras.src.utils import summary_utils
from keras.src.utils.module_utils import litert
from keras.src.utils.module_utils import tensorflow as tf


def export_litert(
    model,
    filepath,
    verbose=True,
    input_signature=None,
    aot_compile_targets=None,
    **kwargs,
):
    """Export the model as a LiteRT artifact for inference.

    Args:
        model: The Keras model to export.
        filepath: The path to save the exported artifact.
        verbose: `bool`. Whether to print a message during export. Defaults to
            `None`, which uses the default value set by different backends and
            formats.
        input_signature: Optional input signature specification. If
            `None`, it will be inferred.
        aot_compile_targets: Optional list of LiteRT targets for AOT
        compilation.
        **kwargs: Additional keyword arguments passed to the exporter.
    """

    exporter = LiteRTExporter(
        model=model,
        input_signature=input_signature,
        verbose=verbose,
        aot_compile_targets=aot_compile_targets,
        **kwargs,
    )
    exporter.export(filepath)
    if verbose:
        io_utils.print_msg(f"Saved artifact at '{filepath}'.")


class LiteRTExporter:
    """Exporter for the LiteRT (TFLite) format.

    This class handles the conversion of Keras models for LiteRT runtime and
    generates a `.tflite` model file. For efficient inference on mobile and
    embedded devices, it creates a single callable signature based on the
    model's `call()` method and supports optional Ahead-of-Time (AOT)
    compilation for specific hardware targets.
    """

    def __init__(
        self,
        model,
        input_signature=None,
        verbose=False,
        aot_compile_targets=None,
        **kwargs,
    ):
        """Initialize the LiteRT exporter.

        Args:
            model: The Keras model to export
            input_signature: Input signature specification
            verbose: Whether to print progress messages during export.
            aot_compile_targets: List of LiteRT targets for AOT compilation
            **kwargs: Additional export parameters
        """
        self.model = model
        self.input_signature = input_signature
        self.verbose = verbose
        self.aot_compile_targets = aot_compile_targets
        self.kwargs = kwargs

    def export(self, filepath):
        """Exports the Keras model to a TFLite file and optionally performs AOT
        compilation.

        Args:
            filepath: Output path for the exported model

        Returns:
            Path to exported model or compiled models if AOT compilation is
            performed
        """
        if self.verbose:
            io_utils.print_msg("Starting LiteRT export...")

        # 1. Ensure the model is built by calling it if necessary
        self._ensure_model_built()

        # 2. Resolve / infer input signature
        if self.input_signature is None:
            if self.verbose:
                io_utils.print_msg("Inferring input signature from model.")
            from keras.src.export.export_utils import get_input_signature

            self.input_signature = get_input_signature(self.model)

        # 3. Convert the model to TFLite.
        tflite_model = self._convert_to_tflite(self.input_signature)

        if self.verbose:
            # Calculate model size from the serialized bytes
            final_size_mb = len(tflite_model) / (1024 * 1024)
            io_utils.print_msg(
                f"TFLite model converted successfully. Size: "
                f"{final_size_mb:.2f} MB"
            )

        # 4. Save the initial TFLite model to the specified file path.
        assert filepath.endswith(".tflite"), (
            "The LiteRT export requires the filepath to end with '.tflite'. "
            f"Got: {filepath}"
        )

        with open(filepath, "wb") as f:
            f.write(tflite_model)

        if self.verbose:
            io_utils.print_msg(f"TFLite model saved to {filepath}")

        # 5. Perform AOT compilation if targets are specified and LiteRT is
        # available
        compiled_models = None
        if self.aot_compile_targets and litert.available:
            if self.verbose:
                io_utils.print_msg(
                    "Performing AOT compilation for LiteRT targets..."
                )
            compiled_models = self._aot_compile(filepath)
        elif self.aot_compile_targets and not litert.available:
            logging.warning(
                "AOT compilation requested but LiteRT is not available. "
                "Skipping AOT compilation."
            )

        if self.verbose:
            io_utils.print_msg(
                f"LiteRT export completed. Base model: {filepath}"
            )
            if compiled_models:
                io_utils.print_msg(
                    f"AOT compiled models: {len(compiled_models.models)} "
                    "variants"
                )

        return compiled_models if compiled_models else filepath

    def _ensure_model_built(self):
        """
        Ensures the model is built before conversion.

        For models that are not yet built, this attempts to build them
        using the input signature or model.inputs.
        """
        if self.model.built:
            return

        if self.verbose:
            io_utils.print_msg("Building model before conversion...")

        try:
            # Try to build using input_signature if available
            if self.input_signature:
                input_shapes = tree.map_structure(
                    lambda spec: spec.shape, self.input_signature
                )
                self.model.build(input_shapes)
            # Fall back to model.inputs for Functional/Sequential models
            elif hasattr(self.model, "inputs") and self.model.inputs:
                input_shapes = [inp.shape for inp in self.model.inputs]
                if len(input_shapes) == 1:
                    self.model.build(input_shapes[0])
                else:
                    self.model.build(input_shapes)
            else:
                raise ValueError(
                    "Cannot export model to the litert format as the "
                    "input_signature could not be inferred. Either pass an "
                    "`input_signature` to `model.export()` or ensure that the "
                    "model is already built (called once on real inputs)."
                )

            if self.verbose:
                io_utils.print_msg("Model built successfully.")

        except Exception as e:
            if self.verbose:
                io_utils.print_msg(f"Error building model: {e}")
            raise ValueError(
                f"Failed to build model: {e}. Please ensure the model is "
                "properly defined or provide an input_signature."
            )

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
        
        # TensorFlow backend can use direct conversion
        is_sequential = isinstance(self.model, tf.keras.Sequential)

        # Try direct conversion first for TensorFlow backend
        try:
            if self.verbose:
                model_type = "Sequential" if is_sequential else "Functional"
                io_utils.print_msg(
                    f"{model_type} model detected. Trying direct conversion..."
                )

            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter.experimental_enable_resource_variables = False

            # Apply any additional converter settings from kwargs
            self._apply_converter_kwargs(converter)

            tflite_model = converter.convert()

            if self.verbose:
                io_utils.print_msg("Direct conversion successful.")
            return tflite_model

        except Exception as direct_error:
            if self.verbose:
                model_type = "Sequential" if is_sequential else "Functional"
                io_utils.print_msg(
                    f"Direct conversion failed for {model_type} model: "
                    f"{direct_error}"
                )
                io_utils.print_msg(
                    "Falling back to wrapper-based conversion..."
                )

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
        """Converts the model to TFLite using the tf.Module wrapper.

        Returns:
            A bytes object containing the serialized TFLite model.
        """

        # Define the wrapper class dynamically to avoid module-level
        # tf.Module inheritance
        class KerasModelWrapper(tf.Module):
            """
            A tf.Module wrapper for a Keras model.

            This wrapper is designed to be a clean, serializable interface
            for TFLite conversion. It holds the Keras model and exposes a
            single `__call__` method that is decorated with `tf.function`.
            Crucially, it also ensures all variables from the Keras model
            are tracked by the SavedModel format, which is key to including
            them in the final TFLite model.
            """

            def __init__(self, model):
                super().__init__()
                # Store the model reference in a way that TensorFlow won't
                # try to track it. This prevents the _DictWrapper error during
                # SavedModel serialization
                object.__setattr__(self, "_model", model)

                # Track all variables from the Keras model using proper
                # tf.Module methods. This ensures proper variable handling for
                # stateful layers like BatchNorm
                with self.name_scope:
                    for i, var in enumerate(model.variables):
                        # Use a different attribute name to avoid conflicts with
                        # tf.Module's variables property
                        setattr(self, f"model_var_{i}", var)

            @tf.function
            def __call__(self, *args, **kwargs):
                """The single entry point for the exported model."""
                # Handle both single and multi-input cases
                if args and not kwargs:
                    # Called with positional arguments
                    if len(args) == 1:
                        return self._model(args[0])
                    else:
                        return self._model(*args)
                elif kwargs and not args:
                    # Called with keyword arguments
                    if len(kwargs) == 1 and "inputs" in kwargs:
                        # Single input case
                        return self._model(kwargs["inputs"])
                    else:
                        # Multi-input case - convert to list/dict format
                        # expected by model
                        if (
                            hasattr(self._model, "inputs")
                            and len(self._model.inputs) > 1
                        ):
                            # Multi-input functional model
                            input_list = []
                            missing_inputs = []
                            for input_layer in self._model.inputs:
                                input_name = input_layer.name
                                if input_name in kwargs:
                                    input_list.append(kwargs[input_name])
                                else:
                                    missing_inputs.append(input_name)

                            if missing_inputs:
                                available = list(kwargs.keys())
                                raise ValueError(
                                    f"Missing required inputs for multi-input "
                                    f"model: {missing_inputs}. "
                                    f"Available kwargs: {available}. "
                                    f"Please provide all inputs by name."
                                )

                            return self._model(input_list)
                        else:
                            # Single input model called with named arguments
                            return self._model(list(kwargs.values())[0])
                else:
                    # Fallback to original call
                    return self._model(*args, **kwargs)

        # 1. Wrap the Keras model in our clean tf.Module.
        wrapper = KerasModelWrapper(self.model)

        # 2. Get a concrete function from the wrapper.
        if not isinstance(input_signature, (list, tuple)):
            input_signature = [input_signature]

        from keras.src.export.export_utils import make_tf_tensor_spec

        tensor_specs = [make_tf_tensor_spec(spec) for spec in input_signature]

        # Pass tensor specs as positional arguments to get the concrete
        # function.
        concrete_func = wrapper.__call__.get_concrete_function(*tensor_specs)

        # 3. Convert from the concrete function.
        if self.verbose:
            io_utils.print_msg(
                "Converting concrete function to TFLite format..."
            )

        # Try multiple conversion strategies for better inference compatibility
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
                        f"Conversion successful {strategy['name']}!"
                    )

                return tflite_model

            except Exception as e:
                if self.verbose:
                    io_utils.print_msg(
                        f"Conversion failed {strategy['name']}: {e}"
                    )
                continue

        # If all strategies fail, raise the last error
        raise RuntimeError(
            "All conversion strategies failed for wrapper-based conversion"
        )

    def _apply_converter_kwargs(self, converter):
        """Apply additional converter settings from kwargs.

        This method applies any TFLite converter settings passed via kwargs
        to the converter object. Common settings include:
        - optimizations: List of optimization options
          (e.g., [tf.lite.Optimize.DEFAULT])
        - representative_dataset: Dataset generator for quantization
        - target_spec: Additional target specification settings
        - inference_input_type: Input type for inference (e.g., tf.int8)
        - inference_output_type: Output type for inference (e.g., tf.int8)

        Args:
            converter: tf.lite.TFLiteConverter instance to configure
        """
        if not self.kwargs:
            return

        for key, value in self.kwargs.items():
            if hasattr(converter, key):
                setattr(converter, key, value)
                if self.verbose:
                    io_utils.print_msg(f"Applied converter setting: {key}")
            elif key == "target_spec" and isinstance(value, dict):
                # Handle nested target_spec settings
                for spec_key, spec_value in value.items():
                    if hasattr(converter.target_spec, spec_key):
                        setattr(converter.target_spec, spec_key, spec_value)
                        if self.verbose:
                            io_utils.print_msg(
                                f"Applied target_spec setting: {spec_key}"
                            )
            elif self.verbose:
                io_utils.print_msg(
                    f"Warning: Unknown converter setting '{key}' - ignoring"
                )

    def _aot_compile(self, tflite_filepath):
        """Performs AOT compilation using LiteRT."""
        if not litert.available:
            raise RuntimeError("LiteRT is not available for AOT compilation")

        try:
            # Create a LiteRT model from the TFLite file
            litert_model = litert.python.aot.core.types.Model.create_from_path(
                tflite_filepath
            )

            # Determine output directory
            base_dir = os.path.dirname(tflite_filepath)
            model_name = os.path.splitext(os.path.basename(tflite_filepath))[0]
            output_dir = os.path.join(base_dir, f"{model_name}_compiled")

            if self.verbose:
                io_utils.print_msg(
                    f"AOT compiling for targets: {self.aot_compile_targets}"
                )
                io_utils.print_msg(f"Output directory: {output_dir}")

            # Perform AOT compilation
            result = litert.python.aot.aot_compile(
                input_model=litert_model,
                output_dir=output_dir,
                target=self.aot_compile_targets,
                keep_going=True,  # Continue even if some targets fail
            )

            if self.verbose:
                io_utils.print_msg(
                    f"AOT compilation completed: {len(result.models)} "
                    f"successful, {len(result.failed_backends)} failed"
                )
                if result.failed_backends:
                    for backend, error in result.failed_backends:
                        io_utils.print_msg(
                            f"  Failed: {backend.id()} - {error}"
                        )

                # Print compilation report if available
                try:
                    report = result.compilation_report()
                    if report:
                        io_utils.print_msg("Compilation Report:")
                        io_utils.print_msg(report)
                except Exception:
                    pass

            return result

        except Exception as e:
            if self.verbose:
                io_utils.print_msg(f"AOT compilation failed: {e}")
                io_utils.print_msg(traceback.format_exc())
            raise RuntimeError(f"AOT compilation failed: {e}")

    def _get_available_litert_targets(self):
        """Get available LiteRT targets for AOT compilation."""
        if not litert.available:
            return []

        try:
            # Get all registered targets
            targets = (
                litert.python.aot.vendors.import_vendor.AllRegisteredTarget()
            )
            return targets if isinstance(targets, list) else [targets]
        except Exception as e:
            if self.verbose:
                io_utils.print_msg(f"Failed to get available targets: {e}")
            return []

    @classmethod
    def export_with_aot(
        cls, model, filepath, targets=None, verbose=True, **kwargs
    ):
        """
        Convenience method to export a Keras model with AOT compilation.

        Args:
            model: Keras model to export
            filepath: Output file path
            targets: List of LiteRT targets for AOT compilation (e.g.,
            ['qualcomm', 'mediatek'])
            verbose: Whether to print verbose output
            **kwargs: Additional arguments for the exporter

        Returns:
            CompilationResult if AOT compilation is performed, otherwise the
            filepath
        """
        exporter = cls(
            model=model, verbose=verbose, aot_compile_targets=targets, **kwargs
        )
        return exporter.export(filepath)

    @classmethod
    def get_available_targets(cls):
        """Get list of available LiteRT AOT compilation targets."""
        if not litert.available:
            return []

        dummy_exporter = cls(model=None)
        return dummy_exporter._get_available_litert_targets()
