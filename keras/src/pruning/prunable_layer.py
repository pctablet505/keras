"""A mixin class to make a layer prunable."""

from keras.src import ops


class PrunableLayer:
    """A mixin to make a layer prunable.

    This class is designed to be mixed in with other Keras layers
    (e.g., Dense, Conv2D) to add pruning capabilities. It holds the
    pruning mask as a non-trainable weight and applies it to the
    layer's kernel during the forward pass.
    """

    def build(self, *args, **kwargs):
        """Create the pruning mask and store the original kernel."""
        super().build(*args, **kwargs)
        if not hasattr(self, "kernel"):
            raise ValueError(
                "PrunableLayer can only be used with layers that have a "
                "'kernel' attribute."
            )

        # This will be the trainable, unpruned weight
        self.unpruned_kernel = self.kernel

        self.pruning_mask = self.add_weight(
            name="pruning_mask",
            shape=self.kernel.shape,
            initializer="ones",
            trainable=False,
            dtype="bool",
        )

    @property
    def kernel(self):
        """Return the masked kernel. This is what the layer will use."""
        return self.unpruned_kernel * ops.cast(
            self.pruning_mask, self.unpruned_kernel.dtype
        )

    @kernel.setter
    def kernel(self, value):
        # The kernel is now managed by `unpruned_kernel` and `pruning_mask`.
        # This setter prevents external code from overwriting the property.
        pass
