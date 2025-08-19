"""Pruning method classes for different pruning algorithms."""

import abc
import tensorflow as tf

from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.trainers.data_adapters import data_adapter_utils



# To verify GPU usage, you can add the following lines at the start of your
# script:
# import tensorflow as tf
# tf.debugging.set_log_device_placement(True)


def _validate_gradient_method_requirements(method_name, model, dataset):
    """Minimal validation for gradient-based pruning methods."""
    if model is None:
        raise ValueError(
            f"{method_name} requires 'model' parameter. Pass model through "
            "model.prune() kwargs."
        )
    if not hasattr(model, "compiled") or not model.compiled:
        raise ValueError(
            f"{method_name} requires a compiled model. Please call "
            "`model.compile()` before pruning."
        )
    if dataset is None:
        raise ValueError(
            f"{method_name} requires 'dataset' parameter. Pass dataset as "
            "tuple (x, y) through model.prune() kwargs."
        )


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

    def _compute_gradients(self, weights, **kwargs):
        """Compute per-variable averaged gradients over `dataset`.

        Returns the gradient tensor for the target weight `weights`.
        Expects `kwargs` to contain `model` and `dataset`.
        """
        model = kwargs.get("model")
        dataset = kwargs.get("dataset")

        _validate_gradient_method_requirements(
            "Gradient-based pruning", model, dataset
        )

        trainable_variables = tuple(model.trainable_variables)
        cache_key = (id(model), id(dataset), "grads")

        if cache_key not in self._all_gradients_cache:
            start_time = tf.timestamp()
            tf.print("[DEBUG] Starting gradient computation...")
            if isinstance(dataset, tuple):
                x_data, y_data = dataset
                # Move data to GPU upfront to avoid per-batch transfer overhead.
                with tf.device("/GPU:0"):
                    x_data_gpu = tf.constant(x_data)
                    y_data_gpu = tf.constant(y_data)
                tf_dataset = (
                    tf.data.Dataset.from_tensor_slices((x_data_gpu, y_data_gpu))
                    .batch(32)
                    .prefetch(tf.data.AUTOTUNE)
                )
            else:
                # For large datasets, prefetch to GPU.
                tf_dataset = dataset.prefetch(tf.data.AUTOTUNE)

            # Use tf.data.Dataset.reduce for in-graph accumulation
            @tf.function
            def reduce_func(state, batch):
                *sum_grads, num_batches = state
                x, y, sample_weight = (
                    data_adapter_utils.unpack_x_y_sample_weight(batch)
                )

                with tf.GradientTape() as tape:
                    preds = model(x, training=True)
                    loss = model._compute_loss(
                        x=x,
                        y=y,
                        y_pred=preds,
                        sample_weight=sample_weight,
                        training=True,
                    )
                    if getattr(model, "optimizer", None) and hasattr(
                        model.optimizer, "scale_loss"
                    ):
                        loss = model.optimizer.scale_loss(loss)

                grads = tape.gradient(loss, trainable_variables)
                if getattr(model, "optimizer", None) and hasattr(
                    model.optimizer, "unscale_gradients"
                ):
                    grads = model.optimizer.unscale_gradients(grads)

                sum_grads = [
                    s + g if g is not None else s
                    for s, g in zip(sum_grads, grads)
                ]
                num_batches += 1
                return tuple(sum_grads) + (num_batches,)

            initial_state = tuple(
                [ops.zeros_like(v) for v in trainable_variables]
            ) + (tf.constant(0, dtype=tf.int64),)

            reduce_start_time = tf.timestamp()
            tf.print("[DEBUG] Starting tf.data.Dataset.reduce for grads...")
            final_state = tf_dataset.reduce(initial_state, reduce_func)
            reduce_end_time = tf.timestamp()
            tf.print(
                "[DEBUG] Finished tf.data.Dataset.reduce for grads. Time taken:",
                reduce_end_time - reduce_start_time,
                "seconds.",
            )

            sum_grads = final_state[:-1]
            num_batches = final_state[-1]

            if num_batches > 0:
                avg_grads = [
                    tf.math.divide_no_nan(g, tf.cast(num_batches, g.dtype))
                    for g in sum_grads
                ]
            else:
                avg_grads = sum_grads

            self._all_gradients_cache[cache_key] = list(avg_grads)
            end_time = tf.timestamp()
            tf.print(
                "[DEBUG] Finished gradient computation. Total time:",
                end_time - start_time,
                "seconds.",
            )

        all_gradients = self._all_gradients_cache[cache_key]
        target_weight_var = self._find_target_weight_variable(model, weights)
        target_index = -1
        for i, v in enumerate(model.trainable_variables):
            if v is target_weight_var:
                target_index = i
                break
        if target_index == -1:
            raise ValueError(
                "Target variable not found in model.trainable_variables"
            )
        return all_gradients[target_index]

    def _compute_gradient_statistics(self, weights, **kwargs):
        """Compute (E[g], E[g^2]) per trainable variable over the dataset.

        Returns the pair of tensors for the requested weight.
        """
        model = kwargs.get("model")
        dataset = kwargs.get("dataset")

        _validate_gradient_method_requirements(
            "Gradient-based pruning", model, dataset
        )

        trainable_variables = tuple(model.trainable_variables)
        cache_key = (id(model), id(dataset), "stats")

        if cache_key not in self._all_gradients_cache:
            start_time = tf.timestamp()
            tf.print("[DEBUG] Starting gradient statistics computation...")
            if isinstance(dataset, tuple):
                x_data, y_data = dataset
                # Move data to GPU upfront to avoid per-batch transfer overhead.
                with tf.device("/GPU:0"):
                    x_data_gpu = tf.constant(x_data)
                    y_data_gpu = tf.constant(y_data)
                tf_dataset = (
                    tf.data.Dataset.from_tensor_slices((x_data_gpu, y_data_gpu))
                    .batch(64)
                    .prefetch(tf.data.AUTOTUNE)
                )
            else:
                # For large datasets, prefetch to GPU.
                tf_dataset = dataset.prefetch(tf.data.AUTOTUNE)

            # Use tf.data.Dataset.reduce for in-graph accumulation
            @tf.function
            def reduce_func(state, batch):
                num_vars = len(trainable_variables)
                sum_grads = state[:num_vars]
                sum_sq_grads = state[num_vars : 2 * num_vars]
                num_batches = state[-1]

                x, y, sample_weight = (
                    data_adapter_utils.unpack_x_y_sample_weight(batch)
                )

                with tf.GradientTape() as tape:
                    preds = model(x, training=True)
                    loss = model._compute_loss(
                        x=x,
                        y=y,
                        y_pred=preds,
                        sample_weight=sample_weight,
                        training=True,
                    )
                    if getattr(model, "optimizer", None) and hasattr(
                        model.optimizer, "scale_loss"
                    ):
                        loss = model.optimizer.scale_loss(loss)

                grads = tape.gradient(loss, trainable_variables)
                if getattr(model, "optimizer", None) and hasattr(
                    model.optimizer, "unscale_gradients"
                ):
                    grads = model.optimizer.unscale_gradients(grads)

                sum_grads = [
                    s + g if g is not None else s
                    for s, g in zip(sum_grads, grads)
                ]
                sum_sq_grads = [
                    s + tf.square(g) if g is not None else s
                    for s, g in zip(sum_sq_grads, grads)
                ]
                num_batches += 1
                return tuple(sum_grads) + tuple(sum_sq_grads) + (num_batches,)

            num_vars = len(trainable_variables)
            initial_state = (
                tuple([ops.zeros_like(v) for v in trainable_variables])
                + tuple([ops.zeros_like(v) for v in trainable_variables])
                + (tf.constant(0, dtype=tf.int64),)
            )

            reduce_start_time = tf.timestamp()
            tf.print("[DEBUG] Starting tf.data.Dataset.reduce for stats...")
            final_state = tf_dataset.reduce(initial_state, reduce_func)
            reduce_end_time = tf.timestamp()
            tf.print(
                "[DEBUG] Finished tf.data.Dataset.reduce for stats. Time taken:",
                reduce_end_time - reduce_start_time,
                "seconds.",
            )

            sum_grads = final_state[:num_vars]
            sum_sq_grads = final_state[num_vars : 2 * num_vars]
            num_batches = final_state[-1]

            if num_batches > 0:
                avg_grads = [
                    tf.math.divide_no_nan(g, tf.cast(num_batches, g.dtype))
                    for g in sum_grads
                ]
                avg_sq_grads = [
                    tf.math.divide_no_nan(g, tf.cast(num_batches, g.dtype))
                    for g in sum_sq_grads
                ]
                stats = (avg_grads, avg_sq_grads)
            else:
                stats = (list(sum_grads), list(sum_sq_grads))

            self._all_gradients_cache[cache_key] = stats
            end_time = tf.timestamp()
            tf.print(
                "[DEBUG] Finished gradient statistics computation. Total time:",
                end_time - start_time,
                "seconds.",
            )

        all_gradients, all_squared_gradients = self._all_gradients_cache[
            cache_key
        ]
        target_weight_var = self._find_target_weight_variable(model, weights)
        target_index = -1
        for i, v in enumerate(model.trainable_variables):
            if v is target_weight_var:
                target_index = i
                break
        if target_index == -1:
            raise ValueError(
                "Target variable not found in model.trainable_variables"
            )
        return (
            all_gradients[target_index],
            all_squared_gradients[target_index],
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
        overall_start_time = tf.timestamp()
        tf.print("[DEBUG] SaliencyPruning.compute_mask started.")
        weights_tensor = (
            weights.value()
            if hasattr(weights, "value") and callable(weights.value)
            else weights
        )
        if sparsity_ratio <= 0:
            return ops.ones_like(weights_tensor, dtype="bool")
        if sparsity_ratio >= 1:
            return ops.zeros_like(weights_tensor, dtype="bool")

        score_start_time = tf.timestamp()
        tf.print("[DEBUG] Calculating saliency scores...")
        saliency_scores = self.calculate_scores(weights, **kwargs)
        score_end_time = tf.timestamp()
        tf.print(
            "[DEBUG] Saliency score calculation took:",
            score_end_time - score_start_time,
            "seconds.",
        )

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

        overall_end_time = tf.timestamp()
        tf.print(
            "[DEBUG] SaliencyPruning.compute_mask finished. Total time:",
            overall_end_time - overall_start_time,
            "seconds.",
        )
        return ops.reshape(mask_flat, ops.shape(saliency_scores))

    def calculate_scores(self, weights, **kwargs):
        """Calculate saliency scores: |weight * gradient|."""
        gradients = self._compute_gradients(weights, **kwargs)
        weights_tensor = (
            weights.value()
            if hasattr(weights, "value") and callable(weights.value)
            else weights
        )
        return ops.abs(weights_tensor * gradients)


@keras_export("keras.pruning.TaylorPruning")
class TaylorPruning(PruningMethod):
    """Second-order Taylor expansion based pruning method."""

    def __init__(self, second_order_weight=0.1, use_fisher_approximation=True):
        self.second_order_weight = second_order_weight
        self.use_fisher_approximation = use_fisher_approximation

    def compute_mask(self, weights, sparsity_ratio, **kwargs):
        overall_start_time = tf.timestamp()
        tf.print("[DEBUG] TaylorPruning.compute_mask started.")
        weights_tensor = (
            weights.value()
            if hasattr(weights, "value") and callable(weights.value)
            else weights
        )
        if sparsity_ratio <= 0:
            return ops.ones_like(weights_tensor, dtype="bool")
        if sparsity_ratio >= 1:
            return ops.zeros_like(weights_tensor, dtype="bool")

        score_start_time = tf.timestamp()
        tf.print("[DEBUG] Calculating taylor scores...")
        taylor_scores = self._compute_taylor_scores(weights, **kwargs)
        score_end_time = tf.timestamp()
        tf.print(
            "[DEBUG] Taylor score calculation took:",
            score_end_time - score_start_time,
            "seconds.",
        )

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

        overall_end_time = tf.timestamp()
        tf.print(
            "[DEBUG] TaylorPruning.compute_mask finished. Total time:",
            overall_end_time - overall_start_time,
            "seconds.",
        )
        return ops.reshape(mask_flat, ops.shape(taylor_scores))

    def _compute_taylor_scores(self, weights, **kwargs):
        """Efficient Taylor scores computation."""
        weights_tensor_val = (
            weights.value()
            if hasattr(weights, "value") and callable(weights.value)
            else weights
        )
        gradients, squared_gradients = self._compute_gradient_statistics(
            weights, **kwargs
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
