"""Keras callbacks for pruning."""

import keras
from keras.src.pruning.pruning_schedule import PruningSchedule
from keras.src.pruning.core import get_pruning_mask, should_prune_layer
from keras.src.pruning.prunable_layer import PrunableLayer


class PruningCallback(keras.callbacks.Callback):
    """Callback to update pruning masks during training."""

    def __init__(
        self,
        pruning_schedule,
        update_frequency="epoch",
        pruning_method="l1",
        **kwargs,
    ):
        super().__init__()
        if not isinstance(pruning_schedule, PruningSchedule):
            raise ValueError(
                "pruning_schedule must be an instance of PruningSchedule."
            )
        self.pruning_schedule = pruning_schedule
        if update_frequency not in ["epoch", "batch"]:
            raise ValueError(
                "update_frequency must be either 'epoch' or 'batch'."
            )
        self.update_frequency = update_frequency
        self.pruning_method = pruning_method
        self.kwargs = kwargs
        self.step = 0

    def _update_masks(self):
        """Update the pruning masks for all prunable layers."""
        sparsity = self.pruning_schedule(self.step)
        print(f"\nStep {self.step}: Updating pruning masks to sparsity {sparsity:.4f}")
        for layer in self.model.layers:
            if isinstance(layer, PrunableLayer) and should_prune_layer(layer):
                mask = get_pruning_mask(
                    layer,
                    sparsity,
                    method=self.pruning_method,
                    model=self.model,
                    **self.kwargs,
                )
                layer.pruning_mask.assign(mask)
        self.step += 1

    def on_epoch_begin(self, epoch, logs=None):
        if self.update_frequency == "epoch":
            self._update_masks()

    def on_train_batch_begin(self, batch, logs=None):
        if self.update_frequency == "batch":
            self._update_masks()

    def on_train_begin(self, logs=None):
        # Ensure step is reset at the beginning of training
        self.step = 0
