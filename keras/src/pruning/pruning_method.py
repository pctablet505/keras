"""Pruning method classes for different pruning algorithms."""

import abc

from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export



@keras_export("keras.pruning.PruningMethod")
class PruningMethod(abc.ABC):
    """Abstract base class for pruning methods."""

    _all_gradients_cache = {}

    @abc.abstractmethod
    def compute_mask(self, weights, sparsity_ratio, **kwargs):
        pass

    def apply_mask(self, weights, mask):
        return weights * ops.cast(mask, weights.dtype)

    def _find_target_weight_variable(self, model, weights_tensor):
        """Find the target weight variable in the model that matches the given weights."""
        trainable_weights = model.trainable_variables
        
        # First try exact shape and identity matching
        for weight in trainable_weights:
            if weight is weights_tensor:
                return weight
                
        # Then try shape and value matching
        for weight in trainable_weights:
            if ops.shape(weight) == ops.shape(weights_tensor):
                weight_val = (
                    weight.value()
                    if hasattr(weight, "value") and callable(weight.value)
                    else weight
                )
                # Use more lenient comparison
                diff = backend.convert_to_numpy(
                    ops.mean(ops.abs(weight_val - weights_tensor))
                )
                if diff < 1e-4:  # More lenient threshold
                    return weight
                    
        # Debug: Print shapes to help identify the issue
        print(f"Target shape: {ops.shape(weights_tensor)}")
        print(f"Available shapes: {[ops.shape(w) for w in trainable_weights]}")
        
        raise ValueError(
            f"Could not find target weight variable with shape "
            f"{ops.shape(weights_tensor)} in {len(trainable_weights)} trainable variables."
        )


    


@keras_export("keras.pruning.L1Pruning")
class L1Pruning(PruningMethod):
    """L1 norm-based pruning method."""

    def __init__(self, structured=False):
        self.structured = structured

    def compute_mask(self, weights, sparsity_ratio, **kwargs):
        if sparsity_ratio <= 0:
            return ops.ones_like(weights, dtype="bool")
        if sparsity_ratio >= 1:
            return ops.zeros_like(weights, dtype="bool")

        if self.structured:
            return self._compute_structured_mask(weights, sparsity_ratio)
        else:
            return self._compute_unstructured_mask(weights, sparsity_ratio)

    def _compute_unstructured_mask(self, weights, sparsity_ratio):
        l1_weights = ops.abs(weights)
        flat_weights = ops.reshape(l1_weights, [-1])
        total_size = int(backend.convert_to_numpy(ops.size(flat_weights)))
        k = int(sparsity_ratio * total_size)
        if k == 0:
            return ops.ones_like(weights, dtype="bool")
        sorted_weights = ops.sort(flat_weights)
        threshold = sorted_weights[k]
        return l1_weights > threshold

    def _compute_structured_mask(self, weights, sparsity_ratio):
        if len(ops.shape(weights)) == 2:  # Dense
            l1_norms = ops.sum(ops.abs(weights), axis=0)
        elif len(ops.shape(weights)) == 4:  # Conv2D
            l1_norms = ops.sum(ops.abs(weights), axis=(0, 1, 2))
        else:
            return self._compute_unstructured_mask(weights, sparsity_ratio)

        flat_norms = ops.reshape(l1_norms, [-1])
        total_size = int(backend.convert_to_numpy(ops.size(flat_norms)))
        k = int(sparsity_ratio * total_size)
        if k == 0:
            return ops.ones_like(weights, dtype="bool")
        sorted_norms = ops.sort(flat_norms)
        threshold = sorted_norms[k]
        channel_mask = l1_norms > threshold

        if len(ops.shape(weights)) == 2:
            return ops.broadcast_to(channel_mask[None, :], ops.shape(weights))
        elif len(ops.shape(weights)) == 4:
            return ops.broadcast_to(
                channel_mask[None, None, None, :], ops.shape(weights)
            )
        return self._compute_unstructured_mask(weights, sparsity_ratio)


@keras_export("keras.pruning.SaliencyPruning")
class SaliencyPruning(PruningMethod):
    """Gradient-based saliency pruning method."""

    def compute_mask(self, weights, sparsity_ratio, **kwargs):
        weights_tensor = (
            weights.value()
            if hasattr(weights, "value") and callable(weights.value)
            else weights
        )
        if sparsity_ratio <= 0:
            return ops.ones_like(weights_tensor, dtype="bool")
        if sparsity_ratio >= 1:
            return ops.zeros_like(weights_tensor, dtype="bool")

        model = kwargs.get("model")
        loss_fn = kwargs.get("loss_fn")
        dataset = kwargs.get("dataset")
        loss_fn = _validate_gradient_method_requirements(
            "SaliencyPruning", model, dataset, loss_fn
        )
        kwargs["loss_fn"] = loss_fn
        
        # Clear cache to avoid stale gradients
        self._all_gradients_cache.clear()
        
        saliency_scores = self.calculate_scores(weights_tensor, **kwargs)

        flat_scores = ops.reshape(saliency_scores, [-1])
        num_params = ops.size(flat_scores)
        num_to_prune = ops.cast(
            ops.cast(num_params, "float32") * sparsity_ratio, "int32"
        )

        if num_to_prune >= num_params:
            return ops.zeros_like(weights_tensor, dtype="bool")
        if num_to_prune == 0:
            return ops.ones_like(weights_tensor, dtype="bool")

        # Find the indices of the weights with the smallest scores to prune.
        # We find the top 'k' of the negated scores.
        _, bottom_k_indices = ops.top_k(-flat_scores, k=num_to_prune)

        # Create a mask that keeps all weights by default.
        mask_flat = ops.ones(shape=(num_params,), dtype="bool")
        # Set the weights to be pruned to False.
        updates = ops.zeros(shape=(num_to_prune,), dtype="bool")
        indices = ops.expand_dims(bottom_k_indices, axis=1)

        mask_flat = ops.scatter_update(mask_flat, indices, updates)
        return ops.reshape(mask_flat, ops.shape(saliency_scores))

    def calculate_scores(self, weights, **kwargs):
        """Calculate saliency scores: |weight * gradient|."""
        gradients = self._compute_gradients(weights, **kwargs)
        return ops.abs(weights * gradients)


@keras_export("keras.pruning.TaylorPruning")
class TaylorPruning(PruningMethod):
    """Second-order Taylor expansion based pruning method."""

    def __init__(self, second_order_weight=0.1, use_fisher_approximation=True):
        self.second_order_weight = second_order_weight
        self.use_fisher_approximation = use_fisher_approximation

    def compute_mask(self, weights, sparsity_ratio, **kwargs):
        weights_tensor = (
            weights.value()
            if hasattr(weights, "value") and callable(weights.value)
            else weights
        )
        if sparsity_ratio <= 0:
            return ops.ones_like(weights_tensor, dtype="bool")
        if sparsity_ratio >= 1:
            return ops.zeros_like(weights_tensor, dtype="bool")

        model = kwargs.get("model")
        loss_fn = kwargs.get("loss_fn")
        dataset = kwargs.get("dataset")
        loss_fn = _validate_gradient_method_requirements(
            "TaylorPruning", model, dataset, loss_fn
        )
        kwargs["loss_fn"] = loss_fn
        taylor_scores = self._compute_taylor_scores(weights, **kwargs)

        flat_scores = ops.reshape(taylor_scores, [-1])
        num_params = ops.size(flat_scores)
        num_to_prune = ops.cast(
            ops.cast(num_params, "float32") * sparsity_ratio, "int32"
        )

        if num_to_prune >= num_params:
            return ops.zeros_like(weights_tensor, dtype="bool")
        if num_to_prune == 0:
            return ops.ones_like(weights_tensor, dtype="bool")

        # Find the indices of the weights with the smallest scores to prune.
        _, bottom_k_indices = ops.top_k(-flat_scores, k=num_to_prune)

        # Create a mask that keeps all weights by default.
        mask_flat = ops.ones(shape=(num_params,), dtype="bool")
        # Set the weights to be pruned to False.
        updates = ops.zeros(shape=(num_to_prune,), dtype="bool")
        indices = ops.expand_dims(bottom_k_indices, axis=1)

        mask_flat = ops.scatter_update(mask_flat, indices, updates)
        return ops.reshape(mask_flat, ops.shape(taylor_scores))

    def _compute_taylor_scores(self, weights, **kwargs):
        """Efficient Taylor scores computation."""
        weights_tensor_val = (
            weights.value()
            if hasattr(weights, "value") and callable(weights.value)
            else weights
        )
        gradients, squared_gradients = self._compute_gradient_statistics(
            weights_tensor_val, **kwargs
        )
        first_order_term = ops.abs(gradients * weights_tensor_val)
        second_order_term = self.second_order_weight * ops.abs(
            squared_gradients * ops.square(weights_tensor_val)
        )
        return first_order_term + second_order_term


@keras_export("keras.pruning.RandomPruning")
class RandomPruning(PruningMethod):
    """Random pruning method.

    Randomly prunes weights regardless of their values.
    Mainly useful for research/comparison purposes.
    """

    def __init__(self, seed=None):
        """Initialize random pruning.

        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed

    def compute_mask(self, weights, sparsity_ratio, **kwargs):
        """Compute random pruning mask."""
        if sparsity_ratio <= 0:
            return ops.ones_like(weights, dtype="bool")
        if sparsity_ratio >= 1:
            return ops.zeros_like(weights, dtype="bool")

        # Generate random values and threshold
        if self.seed is not None:
            # Use deterministic random generation if seed provided
            random_vals = ops.random.uniform(
                ops.shape(weights), seed=self.seed, dtype=weights.dtype
            )
        else:
            random_vals = ops.random.uniform(
                ops.shape(weights), dtype=weights.dtype
            )

        # Keep weights where random value > sparsity_ratio
        mask = random_vals > sparsity_ratio
        return mask


@keras_export("keras.pruning.LnPruning")
class LnPruning(L1Pruning):
    """Ln norm-based pruning method.

    Prunes weights with smallest Ln norm magnitude.
    Supports both unstructured and structured pruning.
    """

    def __init__(self, n=2, structured=False):
        """Initialize Ln pruning.

        Args:
            n: Norm order (e.g., 1 for L1, 2 for L2, etc.).
            structured: If True, prune entire channels/filters.
        """
        super().__init__(structured)
        self.n = n

    def _compute_unstructured_mask(self, weights, sparsity_ratio):
        """Unstructured Ln pruning."""
        if self.n == 1:
            ln_weights = ops.abs(weights)
        elif self.n == 2:
            ln_weights = ops.abs(weights)  # For ranking, sqrt not needed
        else:
            ln_weights = ops.power(ops.abs(weights), self.n)

        flat_weights = ops.reshape(ln_weights, [-1])
        total_size = int(backend.convert_to_numpy(ops.size(flat_weights)))
        k = int(sparsity_ratio * total_size)
        if k == 0:
            return ops.ones_like(weights, dtype="bool")

        sorted_weights = ops.sort(flat_weights)
        threshold = sorted_weights[k]

        mask = ln_weights > threshold
        return mask

    def _compute_structured_mask(self, weights, sparsity_ratio):
        """Structured Ln pruning."""
        if len(ops.shape(weights)) == 2:  # Dense layer
            if self.n == 1:
                ln_norms = ops.sum(ops.abs(weights), axis=0)
            elif self.n == 2:
                ln_norms = ops.sqrt(ops.sum(ops.square(weights), axis=0))
            else:
                ln_norms = ops.power(
                    ops.sum(ops.power(ops.abs(weights), self.n), axis=0),
                    1.0 / self.n,
                )
        elif len(ops.shape(weights)) == 4:  # Conv2D layer
            if self.n == 1:
                ln_norms = ops.sum(ops.abs(weights), axis=(0, 1, 2))
            elif self.n == 2:
                ln_norms = ops.sqrt(
                    ops.sum(ops.square(weights), axis=(0, 1, 2))
                )
            else:
                ln_norms = ops.power(
                    ops.sum(
                        ops.power(ops.abs(weights), self.n), axis=(0, 1, 2)
                    ),
                    1.0 / self.n,
                )
        else:
            return self._compute_unstructured_mask(weights, sparsity_ratio)

        flat_norms = ops.reshape(ln_norms, [-1])
        total_size = int(backend.convert_to_numpy(ops.size(flat_norms)))
        k = int(sparsity_ratio * total_size)
        if k == 0:
            return ops.ones_like(weights, dtype="bool")

        sorted_norms = ops.sort(flat_norms)
        threshold = sorted_norms[k]

        channel_mask = ln_norms > threshold

        # Broadcast to weight tensor shape
        if len(ops.shape(weights)) == 2:
            mask = ops.broadcast_to(channel_mask[None, :], ops.shape(weights))
        elif len(ops.shape(weights)) == 4:
            mask = ops.broadcast_to(
                channel_mask[None, None, None, :], ops.shape(weights)
            )

        return mask


@keras_export("keras.pruning.StructuredPruning")
class StructuredPruning(LnPruning):
    def __init__(self, axis=-1):
        super().__init__(n=2, structured=True)
        self.axis = axis
