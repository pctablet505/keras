import warnings

from keras.src import backend
from keras.src import tree
from keras.src.export.export_utils import convert_spec_to_tensor
from keras.src.export.export_utils import get_input_signature
from keras.src.utils import io_utils


def export_torch(
    model,
    filepath,
    input_signature=None,
    verbose=None,
    **kwargs,
):
    """Export the model as a PyTorch ExportedProgram (`.pt2`) artifact.

    Uses `torch.export.export` to capture the model's computation graph.
    The exported artifact can be loaded via `torch.export.load()` or
    converted to LiteRT using `litert-torch-nightly`.

    Args:
        model: The Keras model to export.
        filepath: `str` or `pathlib.Path` object. Path to save the
            exported artifact. Must end with `.pt2`.
        input_signature: Optional input signature. If `None`, inferred
            from the model.
        verbose: `bool`. Whether to print a message after export.
            Defaults to `True`.
        **kwargs: Additional keyword arguments passed to
            `torch.export.export` (e.g., `strict`).

    Example:

    ```python
    model.export("path/to/model.pt2", format="torch")
    import torch
    loaded_program = torch.export.load("path/to/model.pt2")
    output = loaded_program.module()(torch.randn(1, 10))
    ```
    """
    exporter = TorchExporter(
        model=model,
        input_signature=input_signature,
        **kwargs,
    )
    exporter.export(filepath)
    actual_verbose = verbose if verbose is not None else True
    if actual_verbose:
        io_utils.print_msg(f"Saved PyTorch ExportedProgram at '{filepath}'.")


class TorchExporter:
    """Exporter for the PyTorch ExportedProgram (`.pt2`) format."""

    def __init__(
        self,
        model,
        input_signature=None,
        **kwargs,
    ):
        if backend.backend() != "torch":
            raise RuntimeError(
                "`export_torch` is only compatible with the PyTorch backend. "
                f"Current backend: '{backend.backend()}'."
            )

        self.model = model
        self.input_signature = input_signature
        self.kwargs = kwargs

    def export(self, filepath):
        import torch

        filepath = str(filepath)
        if not filepath.endswith(".pt2"):
            raise ValueError(
                "The PyTorch export requires the filepath to end with "
                f"'.pt2'. Got: {filepath}"
            )

        if self.input_signature is None:
            self.input_signature = get_input_signature(self.model)

        sample_inputs = tree.map_structure(
            lambda x: convert_spec_to_tensor(x, replace_none_number=1),
            self.input_signature,
        )
        sample_inputs = tuple(sample_inputs)

        device = self._get_model_device()
        if device is not None:
            sample_inputs = tuple(
                t.to(device) if hasattr(t, "to") else t for t in sample_inputs
            )

        if hasattr(self.model, "eval"):
            self.model.eval()

        export_kwargs = self._get_export_kwargs()

        with warnings.catch_warnings():
            # Suppress Keras internal warnings about submodule registration
            warnings.filterwarnings(
                "ignore",
                message=r".*not properly registered as a submodule.*",
            )

            try:
                exported_program = torch.export.export(
                    self.model,
                    sample_inputs,
                    **export_kwargs,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to export model to PyTorch format. "
                    f"Common causes: unsupported operations, data-dependent "
                    f"control flow, or dynamic shapes. Original error: {e}"
                ) from e

        torch.export.save(exported_program, filepath)
        return filepath

    def _get_model_device(self):
        try:
            for v in self.model.variables:
                if hasattr(v, "value") and hasattr(v.value, "device"):
                    return v.value.device
        except (AttributeError, TypeError):
            pass
        return None

    def _get_export_kwargs(self):
        export_kwargs = {}
        if "strict" in self.kwargs:
            export_kwargs["strict"] = self.kwargs["strict"]
        else:
            # Default to non-strict mode for broader Keras model compatibility.
            export_kwargs["strict"] = False
        return export_kwargs
