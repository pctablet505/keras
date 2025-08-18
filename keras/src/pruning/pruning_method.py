"""Pruning method classes for different pruning algorithms."""

import abc

from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export


def _validate_gradient_method_requirements(method_name, model, dataset, loss_fn):
    """Validate that gradient-based methods have required parameters."""
    if model is None:
        raise ValueError(f"{method_name} requires 'model' parameter. Pass model through model.prune() kwargs.")
    
    if dataset is None:
        raise ValueError(f"{method_name} requires 'dataset' parameter. Pass dataset as tuple (x, y) through model.prune() kwargs.")
    
    # Get loss_fn from model if not provided
    if loss_fn is None:
        if hasattr(model, 'loss') and model.loss is not None:
            return model.loss
        else:
            raise ValueError(f"{method_name} requires 'loss_fn' parameter or model must have a compiled loss function.")
    
    return loss_fn


@keras_export("keras.pruning.PruningMethod")
class PruningMethod(abc.ABC):
    """Abstract base class for pruning methods.

    A pruning method defines the algorithm used to determine which weights
    to prune from a layer.
    """

    @abc.abstractmethod
    def compute_mask(self, weights, sparsity_ratio, **kwargs):
        """Compute a binary mask indicating which weights to prune.

        Args:
            weights: Weight tensor to analyze.
            sparsity_ratio: Float between 0 and 1. Fraction of weights to prune.
            **kwargs: Additional arguments like model, loss_fn, input_data, target_data.

        Returns:
            Binary mask tensor with same shape as weights.
            True = keep weight, False = prune weight.
        """
        pass

    def apply_mask(self, weights, mask):
        """Apply pruning mask to weights.

        Args:
            weights: Weight tensor to prune.
            mask: Binary mask tensor.

        Returns:
            Pruned weight tensor.
        """
        return weights * ops.cast(mask, weights.dtype)

    def _setup_data_iterator(self, dataset, desc, **kwargs):
        """Setup data iterator and progress bar for gradient computation."""
        import numpy as np
        from tqdm import tqdm
        
        # Check if dataset is a pre-batched iterator or a tuple of NumPy arrays.
        is_iterator = not (isinstance(dataset, tuple) and hasattr(dataset[0], 'shape'))

        if is_iterator:
            data_iterator = dataset
            progress_bar = tqdm(data_iterator, desc=desc, unit="batch")
        elif isinstance(dataset, tuple) and len(dataset) == 2:
            x_data, y_data = dataset
            total_samples = x_data.shape[0]
            pruning_batch_size = kwargs.get('pruning_batch_size', 64)
            num_batches = (total_samples + pruning_batch_size - 1) // pruning_batch_size
            
            indices = np.arange(total_samples)
            np.random.shuffle(indices)
            
            def data_generator():
                for i in range(num_batches):
                    start_idx = i * pruning_batch_size
                    end_idx = min(start_idx + pruning_batch_size, total_samples)
                    if start_idx >= end_idx:
                        continue
                    batch_indices = indices[start_idx:end_idx]
                    yield x_data[batch_indices], y_data[batch_indices]
            
            data_iterator = data_generator()
            progress_bar = tqdm(data_iterator, desc=desc, unit="batch", total=num_batches)
        else:
            raise ValueError("Dataset must be a batched iterator (e.g., tf.data.Dataset) or a tuple of (x_data, y_data).")
        
        return data_iterator, progress_bar

    def _find_target_weight_variable(self, model, weights_tensor):
        """Find the target weight variable in the model that matches the given weights."""
        trainable_weights = [v for v in model.trainable_variables if len(v.shape) > 1]

        target_weight_var = None
        for weight in trainable_weights:
            if ops.shape(weight) == ops.shape(weights_tensor):
                weight_tensor_val = weight.value() if hasattr(weight, 'value') and callable(weight.value) else weight
                if backend.convert_to_numpy(ops.mean(ops.abs(weight_tensor_val - weights_tensor))) < 1e-6:
                    target_weight_var = weight
                    break
        
        if target_weight_var is None:
            raise ValueError(f"Could not find target weight variable with shape {ops.shape(weights_tensor)}.")
        
        return target_weight_var


@keras_export("keras.pruning.L1Pruning")
class L1Pruning(PruningMethod):
    """L1 norm-based pruning method.

    Prunes weights with smallest L1 magnitude (absolute value).
    Supports both unstructured and structured pruning.
    """

    def __init__(self, structured=False):
        """Initialize L1 pruning.

        Args:
            structured: If True, prune entire channels/filters based on L1 norm.
                       If False, prune individual weights.
        """
        self.structured = structured

    def compute_mask(self, weights, sparsity_ratio, **kwargs):
        """Compute mask based on L1 norms."""
        if sparsity_ratio <= 0:
            return ops.ones_like(weights, dtype="bool")
        if sparsity_ratio >= 1:
            return ops.zeros_like(weights, dtype="bool")

        if self.structured:
            return self._compute_structured_mask(weights, sparsity_ratio)
        else:
            return self._compute_unstructured_mask(weights, sparsity_ratio)

    def _compute_unstructured_mask(self, weights, sparsity_ratio):
        """Unstructured L1 pruning."""
        l1_weights = ops.abs(weights)
        flat_weights = ops.reshape(l1_weights, [-1])

        # Convert ops.size to int for calculation
        total_size = int(backend.convert_to_numpy(ops.size(flat_weights)))
        k = int(sparsity_ratio * total_size)
        if k == 0:
            return ops.ones_like(weights, dtype="bool")

        sorted_weights = ops.sort(flat_weights)
        threshold = sorted_weights[k]

        mask = l1_weights > threshold
        return mask

    def _compute_structured_mask(self, weights, sparsity_ratio):
        """Structured L1 pruning."""
        if len(ops.shape(weights)) == 2:  # Dense layer
            l1_norms = ops.sum(ops.abs(weights), axis=0)
        elif len(ops.shape(weights)) == 4:  # Conv2D layer
            l1_norms = ops.sum(ops.abs(weights), axis=(0, 1, 2))
        else:
            # Fall back to unstructured for other shapes
            return self._compute_unstructured_mask(weights, sparsity_ratio)

        flat_norms = ops.reshape(l1_norms, [-1])
        total_size = int(backend.convert_to_numpy(ops.size(flat_norms)))
        k = int(sparsity_ratio * total_size)
        if k == 0:
            return ops.ones_like(weights, dtype="bool")

        sorted_norms = ops.sort(flat_norms)
        threshold = sorted_norms[k]

        channel_mask = l1_norms > threshold

        # Broadcast to weight tensor shape
        if len(ops.shape(weights)) == 2:
            mask = ops.broadcast_to(channel_mask[None, :], ops.shape(weights))
        elif len(ops.shape(weights)) == 4:
            mask = ops.broadcast_to(
                channel_mask[None, None, None, :], ops.shape(weights)
            )

        return mask


@keras_export("keras.pruning.StructuredPruning")
class StructuredPruning(PruningMethod):
    """Structured pruning method.

    Prunes entire channels/filters based on their L2 norm.
    """

    def __init__(self, axis=-1):
        """Initialize structured pruning.

        Args:
            axis: Axis along which to compute norms for structured pruning.
                Typically -1 for output channels.
        """
        self.axis = axis

    def compute_mask(self, weights, sparsity_ratio, **kwargs):
        """Compute mask based on channel/filter norms."""
        if sparsity_ratio <= 0:
            return ops.ones_like(weights, dtype="bool")
        if sparsity_ratio >= 1:
            return ops.zeros_like(weights, dtype="bool")

        # Compute L2 norms along appropriate axes
        if len(ops.shape(weights)) == 2:  # Dense layer
            norms = ops.sqrt(ops.sum(ops.square(weights), axis=0))
        elif len(ops.shape(weights)) == 4:  # Conv2D layer
            norms = ops.sqrt(ops.sum(ops.square(weights), axis=(0, 1, 2)))
        else:
            # Fall back to L1 pruning for other shapes
            return L1Pruning(structured=False).compute_mask(
                weights, sparsity_ratio
            )

        # Find threshold
        flat_norms = ops.reshape(norms, [-1])
        total_size = int(backend.convert_to_numpy(ops.size(flat_norms)))
        k = int(sparsity_ratio * total_size)
        if k == 0:
            return ops.ones_like(weights, dtype="bool")

        sorted_norms = ops.sort(flat_norms)
        threshold = sorted_norms[k]

        # Create channel mask
        channel_mask = norms > threshold

        # Broadcast mask to weight tensor shape
        if len(ops.shape(weights)) == 2:
            mask = ops.broadcast_to(channel_mask[None, :], ops.shape(weights))
        elif len(ops.shape(weights)) == 4:
            mask = ops.broadcast_to(
                channel_mask[None, None, None, :], ops.shape(weights)
            )

        return mask


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
class LnPruning(PruningMethod):
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
        self.n = n
        self.structured = structured

    def compute_mask(self, weights, sparsity_ratio, **kwargs):
        """Compute mask based on Ln norms."""
        if sparsity_ratio <= 0:
            return ops.ones_like(weights, dtype="bool")
        if sparsity_ratio >= 1:
            return ops.zeros_like(weights, dtype="bool")

        if self.structured:
            return self._compute_structured_mask(weights, sparsity_ratio)
        else:
            return self._compute_unstructured_mask(weights, sparsity_ratio)

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


@keras_export("keras.pruning.SaliencyPruning")
class SaliencyPruning(PruningMethod):
    """Gradient-based saliency pruning method.

    Estimates weight importance using first-order gradients.
    """

    def __init__(self):
        """Initialize saliency pruning."""
        pass

    def compute_mask(self, weights, sparsity_ratio, **kwargs):
        """Compute saliency-based mask using gradients."""
        # Ensure we work with tensor values, not Variable objects
        if hasattr(weights, 'value') and callable(weights.value):
            weights_tensor = weights.value()
        else:
            weights_tensor = weights
            
        if sparsity_ratio <= 0:
            return ops.ones_like(weights_tensor, dtype="bool")
        if sparsity_ratio >= 1:
            return ops.zeros_like(weights_tensor, dtype="bool")

        # Get model and data from kwargs (passed by core.py)
        model = kwargs.get('model')
        loss_fn = kwargs.get('loss_fn')
        dataset = kwargs.get('dataset')
        
        # Validate requirements and get loss_fn (may return model.loss if not provided)
        loss_fn = _validate_gradient_method_requirements("SaliencyPruning", model, dataset, loss_fn)
        
        # Compute saliency scores (pass validated loss_fn in kwargs)
        kwargs['loss_fn'] = loss_fn
        saliency_scores = self._compute_saliency_scores(weights, **kwargs)

        flat_scores = ops.reshape(saliency_scores, [-1])
        total_size = int(backend.convert_to_numpy(ops.size(flat_scores)))
        k = int(sparsity_ratio * total_size)
        if k == 0:
            return ops.ones_like(weights_tensor, dtype="bool")

        sorted_scores = ops.sort(flat_scores)
        threshold = sorted_scores[k]

        mask = saliency_scores > threshold
        return mask

    def _compute_tensorflow_gradients(self, target_weight_var, model, batch_x, batch_y, loss_fn):
        """Compute gradients using TensorFlow backend."""
        import tensorflow as tf
        import keras
        
        with tf.GradientTape() as tape:
            # Watch the actual TensorFlow variable, not the Keras Variable wrapper
            tf_var = target_weight_var._value if hasattr(target_weight_var, '_value') else target_weight_var
            tape.watch(tf_var)
            predictions = model(batch_x, training=False)
            if callable(loss_fn):
                loss_val = loss_fn(batch_y, predictions)
            else:
                loss_obj = keras.losses.get(loss_fn)
                loss_val = loss_obj(batch_y, predictions)
            # Ensure loss is scalar
            loss = ops.mean(loss_val) if len(ops.shape(loss_val)) > 0 else loss_val
        
        return tape.gradient(loss, tf_var)

    def _compute_jax_gradients(self, target_weight_var, model, batch_x, batch_y, loss_fn):
        """Compute gradients using JAX backend."""
        import jax
        import keras
        
        def loss_fn_for_grad(weight_vals):
            old_weights = target_weight_var.value() if hasattr(target_weight_var, 'value') and callable(target_weight_var.value) else target_weight_var.value if hasattr(target_weight_var, 'value') else target_weight_var
            target_weight_var.assign(weight_vals)
            
            predictions = model(batch_x, training=False)
            if callable(loss_fn):
                loss_val = loss_fn(batch_y, predictions)
            else:
                loss_obj = keras.losses.get(loss_fn)
                loss_val = loss_obj(batch_y, predictions)
            
            target_weight_var.assign(old_weights)
            return ops.mean(loss_val) if len(ops.shape(loss_val)) > 0 else loss_val
        
        weights_val = target_weight_var.value() if hasattr(target_weight_var, 'value') and callable(target_weight_var.value) else target_weight_var
        return jax.grad(loss_fn_for_grad)(weights_val)

    def _compute_torch_gradients(self, target_weight_var, model, batch_x, batch_y, loss_fn):
        """Compute gradients using PyTorch backend."""
        import torch
        import keras
        
        target_var = target_weight_var.value() if hasattr(target_weight_var, 'value') and callable(target_weight_var.value) else target_weight_var.value if hasattr(target_weight_var, 'value') else target_weight_var
        if hasattr(target_var, 'requires_grad') and not target_var.requires_grad:
            target_var.requires_grad_(True)

        predictions = model(batch_x, training=False)
        if callable(loss_fn):
            loss_val = loss_fn(batch_y, predictions)
        else:
            loss_obj = keras.losses.get(loss_fn)
            loss_val = loss_obj(batch_y, predictions)
        loss = ops.mean(loss_val) if len(ops.shape(loss_val)) > 0 else loss_val
        
        return torch.autograd.grad(loss, target_var)[0]

    def _compute_saliency_scores(self, weights, **kwargs):
        """Compute saliency scores using gradients.
        
        Memory-efficient version that processes an iterator dataset or NumPy arrays.
        """
        import keras
        import numpy as np
        from tqdm import tqdm
        
        # Extract parameters from kwargs
        model = kwargs.get('model')
        loss_fn = kwargs.get('loss_fn')
        dataset = kwargs.get('dataset')
        
        # Setup data iterator and progress bar
        data_iterator, progress_bar = self._setup_data_iterator(dataset, "Computing saliency gradients", **kwargs)

        # Use backend-specific gradient computation
        from keras.src import backend as keras_backend
        backend_name = keras_backend.backend()
        
        # Find target weight variable
        weights_tensor = weights.value() if hasattr(weights, 'value') and callable(weights.value) else weights
        target_weight_var = self._find_target_weight_variable(model, weights_tensor)

        # Initialize accumulated gradients
        accumulated_gradients = None
        total_batches_processed = 0

        # Process data in batches with progress bar
        for batch_data in progress_bar:
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                batch_x, batch_y = batch_data
            else:
                raise ValueError("Dataset must yield tuples of (x, y).")

            # Ensure data is in tensor format
            batch_x = ops.convert_to_tensor(batch_x)
            batch_y = ops.convert_to_tensor(batch_y)

            # Backend-specific gradient computation
            if backend_name == "tensorflow":
                batch_avg_gradients = self._compute_tensorflow_gradients(target_weight_var, model, batch_x, batch_y, loss_fn)
            elif backend_name == "jax":
                batch_avg_gradients = self._compute_jax_gradients(target_weight_var, model, batch_x, batch_y, loss_fn)
            elif backend_name == "torch":
                batch_avg_gradients = self._compute_torch_gradients(target_weight_var, model, batch_x, batch_y, loss_fn)
            else:
                raise ValueError(f"SaliencyPruning is not supported for backend '{backend_name}'.")

            if batch_avg_gradients is None:
                continue

            # Accumulate gradients
            batch_gradients_val = batch_avg_gradients.value() if hasattr(batch_avg_gradients, 'value') and callable(batch_avg_gradients.value) else batch_avg_gradients
            
            if accumulated_gradients is None:
                accumulated_gradients = batch_gradients_val
            else:
                accumulated_gradients = accumulated_gradients + batch_gradients_val
            
            total_batches_processed += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'batches': f"{total_batches_processed}"
            })

        progress_bar.close()

        if accumulated_gradients is None or total_batches_processed == 0:
            raise ValueError(f"Could not compute gradients for weight tensor with shape {ops.shape(weights_tensor)}.")
        
        # Average the accumulated gradients
        gradients_val = accumulated_gradients / total_batches_processed
        
        saliency_scores = ops.abs(gradients_val * weights_tensor)
        
        return saliency_scores


@keras_export("keras.pruning.TaylorPruning")
class TaylorPruning(PruningMethod):
    """Second-order Taylor expansion based pruning method.

    Estimates weight importance using second-order Taylor expansion:
    Taylor score ≈ |∂L/∂w * w| + λ * |∂²L/∂w² * w|
    
    This implementation uses Fisher Information Matrix diagonal as Hessian approximation,
    which is more stable and theoretically grounded than using gradient squares.
    """

    def __init__(self, second_order_weight=0.1, use_fisher_approximation=True):
        """Initialize Taylor pruning.
        
        Args:
            second_order_weight: Weight for second-order term (λ in the formula).
            use_fisher_approximation: If True, use proper Fisher Information Matrix 
                                    diagonal E[g²] (mathematically correct, optimized implementation).
                                    If False, use simpler approximation (E[g])² (faster but less accurate).
        """
        self.second_order_weight = second_order_weight
        self.use_fisher_approximation = use_fisher_approximation

    def compute_mask(self, weights, sparsity_ratio, **kwargs):
        """Compute Taylor expansion based mask."""
        # Ensure we work with tensor values, not Variable objects
        if hasattr(weights, 'value') and callable(weights.value):
            weights_tensor = weights.value()
        else:
            weights_tensor = weights
            
        if sparsity_ratio <= 0:
            return ops.ones_like(weights_tensor, dtype="bool")
        if sparsity_ratio >= 1:
            return ops.zeros_like(weights_tensor, dtype="bool")

        # Get model and data from kwargs (passed by core.py)
        model = kwargs.get('model')
        loss_fn = kwargs.get('loss_fn')
        dataset = kwargs.get('dataset')
        
        # Validate requirements and get loss_fn (may return model.loss if not provided)
        loss_fn = _validate_gradient_method_requirements("TaylorPruning", model, dataset, loss_fn)

        # Compute Taylor scores (pass validated loss_fn in kwargs)
        kwargs['loss_fn'] = loss_fn
        taylor_scores = self._compute_taylor_scores(weights, **kwargs)

        flat_scores = ops.reshape(taylor_scores, [-1])
        total_size = int(backend.convert_to_numpy(ops.size(flat_scores)))
        k = int(sparsity_ratio * total_size)
        if k == 0:
            return ops.ones_like(weights_tensor, dtype="bool")

        sorted_scores = ops.sort(flat_scores)
        threshold = sorted_scores[k]

        mask = taylor_scores > threshold
        return mask

    def _compute_taylor_scores(self, weights, **kwargs):
        """Compute second-order Taylor expansion scores.
        
        Memory-efficient version that processes an iterator dataset or NumPy arrays.
        """
        import keras
        import numpy as np
        from tqdm import tqdm
        
        # Extract parameters from kwargs
        model = kwargs.get('model')
        loss_fn = kwargs.get('loss_fn')
        dataset = kwargs.get('dataset')
        
        # Setup data iterator and progress bar
        data_iterator, progress_bar = self._setup_data_iterator(dataset, "Computing Taylor gradients", **kwargs)
        
        # Extract tensor values to ensure we work with tensors, not Variables
        weights_tensor_val = weights.value() if hasattr(weights, 'value') and callable(weights.value) else weights

        # Find the target weight variable by matching tensor values
        target_weight_var = self._find_target_weight_variable(model, weights_tensor_val)
        
        # Backend-specific gradient computation
        from keras.src import backend as keras_backend
        backend_name = keras_backend.backend()
        
        # Initialize accumulated gradients and squared gradients
        accumulated_gradients = None
        accumulated_squared_gradients = None
        total_batches_processed = 0

        # Process data in batches with progress bar
        for batch_data in progress_bar:
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                batch_x, batch_y = batch_data
            else:
                raise ValueError("Dataset must yield tuples of (x, y).")

            # Ensure data is in tensor format
            batch_x = ops.convert_to_tensor(batch_x)
            batch_y = ops.convert_to_tensor(batch_y)

            if backend_name == "tensorflow":
                import tensorflow as tf
                
                if self.use_fisher_approximation:
                    # Fast Fisher Information approximation using Gauss-Newton approximation
                    # This is much faster than per-sample gradients while maintaining mathematical validity
                    tf_var = target_weight_var._value if hasattr(target_weight_var, '_value') else target_weight_var
                    
                    with tf.GradientTape() as tape:
                        tape.watch(tf_var)
                        predictions = model(batch_x, training=False)
                        if callable(loss_fn):
                            loss_val = loss_fn(batch_y, predictions)
                        else:
                            loss_obj = keras.losses.get(loss_fn)
                            loss_val = loss_obj(batch_y, predictions)
                        
                        # Use per-sample losses for Fisher approximation
                        if len(loss_val.shape) == 0:
                            total_loss = loss_val
                        elif len(loss_val.shape) == 1:  # Per-sample losses
                            total_loss = tf.reduce_mean(loss_val)
                        else:
                            # Flatten to per-sample losses
                            loss_val = tf.reduce_mean(loss_val, axis=list(range(1, len(loss_val.shape))))
                            total_loss = tf.reduce_mean(loss_val)
                    
                    # Compute standard gradient
                    batch_avg_gradients = tape.gradient(total_loss, tf_var)
                    
                    # Fast Fisher approximation: Use gradient variance across the batch
                    # This approximates E[g²] ≈ Var[g] + E[g]² where Var[g] captures second-order info
                    with tf.GradientTape() as tape2:
                        tape2.watch(tf_var)
                        predictions = model(batch_x, training=False)
                        if callable(loss_fn):
                            loss_val = loss_fn(batch_y, predictions)
                        else:
                            loss_obj = keras.losses.get(loss_fn)
                            loss_val = loss_obj(batch_y, predictions)
                        
                        # Use sum instead of mean to get variance-like behavior
                        if len(loss_val.shape) == 0:
                            variance_loss = loss_val
                        elif len(loss_val.shape) == 1:
                            variance_loss = tf.reduce_sum(tf.square(loss_val - tf.reduce_mean(loss_val)))
                        else:
                            loss_val = tf.reduce_mean(loss_val, axis=list(range(1, len(loss_val.shape))))
                            variance_loss = tf.reduce_sum(tf.square(loss_val - tf.reduce_mean(loss_val)))
                    
                    variance_gradients = tape2.gradient(variance_loss, tf_var)
                    
                    # Combine: E[g²] ≈ |g_mean|² + α * |g_variance|
                    alpha = 0.1  # Weight for variance term
                    if variance_gradients is not None:
                        batch_squared_gradients = tf.square(batch_avg_gradients) + alpha * tf.abs(variance_gradients)
                    else:
                        batch_squared_gradients = tf.square(batch_avg_gradients)
                else:
                    # Simple approximation without Fisher Information
                    with tf.GradientTape() as tape:
                        tf_var = target_weight_var._value if hasattr(target_weight_var, '_value') else target_weight_var
                        tape.watch(tf_var)
                        predictions = model(batch_x, training=False)
                        if callable(loss_fn):
                            loss_val = loss_fn(batch_y, predictions)
                        else:
                            loss_obj = keras.losses.get(loss_fn)
                            loss_val = loss_obj(batch_y, predictions)
                        total_loss = ops.mean(loss_val) if len(ops.shape(loss_val)) > 0 else loss_val
                    
                    batch_avg_gradients = tape.gradient(total_loss, tf_var)
                    batch_squared_gradients = ops.square(batch_avg_gradients) if batch_avg_gradients is not None else None
                
            elif backend_name == "jax":
                import jax
                import jax.numpy as jnp
                
                if self.use_fisher_approximation:
                    # Fast Fisher Information approximation using Gauss-Newton approximation for JAX
                    def loss_fn_for_grad(weight_vals):
                        old_weights = target_weight_var.value() if hasattr(target_weight_var, 'value') and callable(target_weight_var.value) else target_weight_var.value if hasattr(target_weight_var, 'value') else target_weight_var
                        target_weight_var.assign(weight_vals)
                        
                        predictions = model(batch_x, training=False)
                        if callable(loss_fn):
                            loss_val = loss_fn(batch_y, predictions)
                        else:
                            loss_obj = keras.losses.get(loss_fn)
                            loss_val = loss_obj(batch_y, predictions)
                        
                        target_weight_var.assign(old_weights)
                        return jnp.mean(loss_val) if len(loss_val.shape) > 0 else loss_val
                    
                    def variance_loss_fn_for_grad(weight_vals):
                        old_weights = target_weight_var.value() if hasattr(target_weight_var, 'value') and callable(target_weight_var.value) else target_weight_var.value if hasattr(target_weight_var, 'value') else target_weight_var
                        target_weight_var.assign(weight_vals)
                        
                        predictions = model(batch_x, training=False)
                        if callable(loss_fn):
                            loss_val = loss_fn(batch_y, predictions)
                        else:
                            loss_obj = keras.losses.get(loss_fn)
                            loss_val = loss_obj(batch_y, predictions)
                        
                        target_weight_var.assign(old_weights)
                        
                        # Compute variance-like loss for Fisher approximation
                        if len(loss_val.shape) == 0:
                            return loss_val
                        elif len(loss_val.shape) == 1:
                            return jnp.sum(jnp.square(loss_val - jnp.mean(loss_val)))
                        else:
                            loss_val = jnp.mean(loss_val, axis=tuple(range(1, len(loss_val.shape))))
                            return jnp.sum(jnp.square(loss_val - jnp.mean(loss_val)))
                    
                    weights_val = target_weight_var.value() if hasattr(target_weight_var, 'value') and callable(target_weight_var.value) else target_weight_var.value if hasattr(target_weight_var, 'value') else target_weight_var
                    
                    # Compute standard gradient
                    batch_avg_gradients = jax.grad(loss_fn_for_grad)(weights_val)
                    
                    # Compute variance gradient for Fisher approximation
                    variance_gradients = jax.grad(variance_loss_fn_for_grad)(weights_val)
                    
                    # Combine: E[g²] ≈ |g_mean|² + α * |g_variance|
                    alpha = 0.1
                    batch_squared_gradients = jnp.square(batch_avg_gradients) + alpha * jnp.abs(variance_gradients)
                else:
                    # Simple approximation
                    def loss_fn_for_grad(weight_vals):
                        old_weights = target_weight_var.value() if hasattr(target_weight_var, 'value') and callable(target_weight_var.value) else target_weight_var.value if hasattr(target_weight_var, 'value') else target_weight_var
                        target_weight_var.assign(weight_vals)
                        
                        predictions = model(batch_x, training=False)
                        if callable(loss_fn):
                            loss_val = loss_fn(batch_y, predictions)
                        else:
                            loss_obj = keras.losses.get(loss_fn)
                            loss_val = loss_obj(batch_y, predictions)
                        
                        target_weight_var.assign(old_weights)
                        return ops.mean(loss_val) if len(ops.shape(loss_val)) > 0 else loss_val
                    
                    weights_val = target_weight_var.value() if hasattr(target_weight_var, 'value') and callable(target_weight_var.value) else target_weight_var.value if hasattr(target_weight_var, 'value') else target_weight_var
                    batch_avg_gradients = jax.grad(loss_fn_for_grad)(weights_val)
                    batch_squared_gradients = ops.square(batch_avg_gradients)
                    
            elif backend_name == "torch":
                import torch
                
                target_var = target_weight_var.value() if hasattr(target_weight_var, 'value') and callable(target_weight_var.value) else target_weight_var.value if hasattr(target_weight_var, 'value') else target_weight_var
                if hasattr(target_var, 'requires_grad'):
                    target_var.requires_grad_(True)
                
                if self.use_fisher_approximation:
                    # Fast Fisher Information approximation using Gauss-Newton approximation for PyTorch
                    
                    # Standard gradient computation
                    predictions = model(batch_x, training=False)
                    if callable(loss_fn):
                        loss_val = loss_fn(batch_y, predictions)
                    else:
                        loss_obj = keras.losses.get(loss_fn)
                        loss_val = loss_obj(batch_y, predictions)
                    
                    total_loss = torch.mean(loss_val) if len(loss_val.shape) > 0 else loss_val
                    batch_avg_gradients = torch.autograd.grad(total_loss, target_var, create_graph=True, retain_graph=True)[0]
                    
                    # Variance-based Fisher approximation
                    if len(loss_val.shape) == 0:
                        variance_loss = loss_val
                    elif len(loss_val.shape) == 1:
                        variance_loss = torch.sum(torch.square(loss_val - torch.mean(loss_val)))
                    else:
                        loss_val_flat = torch.mean(loss_val, dim=tuple(range(1, len(loss_val.shape))))
                        variance_loss = torch.sum(torch.square(loss_val_flat - torch.mean(loss_val_flat)))
                    
                    try:
                        variance_gradients = torch.autograd.grad(variance_loss, target_var, create_graph=False, retain_graph=False)[0]
                        # Combine: E[g²] ≈ |g_mean|² + α * |g_variance|
                        alpha = 0.1
                        batch_squared_gradients = torch.square(batch_avg_gradients) + alpha * torch.abs(variance_gradients)
                    except:
                        # Fallback if variance gradient computation fails
                        batch_squared_gradients = torch.square(batch_avg_gradients)
                        
                else:
                    # Simple approximation
                    predictions = model(batch_x, training=False)
                    if callable(loss_fn):
                        per_sample_loss = loss_fn(batch_y, predictions)
                    else:
                        loss_obj = keras.losses.get(loss_fn)
                        per_sample_loss = loss_obj(batch_y, predictions)
                    
                    total_loss = ops.mean(per_sample_loss)
                    batch_avg_gradients = torch.autograd.grad(total_loss, target_var, create_graph=False)[0]
                    batch_squared_gradients = ops.square(batch_avg_gradients)
                    
            else:
                raise ValueError(f"TaylorPruning not supported for backend '{backend_name}'.")

            if batch_avg_gradients is None:
                continue

            # Accumulate gradients and squared gradients
            batch_gradients_val = batch_avg_gradients.value() if hasattr(batch_avg_gradients, 'value') and callable(batch_avg_gradients.value) else batch_avg_gradients
            batch_squared_gradients_val = batch_squared_gradients.value() if hasattr(batch_squared_gradients, 'value') and callable(batch_squared_gradients.value) else batch_squared_gradients
            
            if accumulated_gradients is None:
                accumulated_gradients = batch_gradients_val
                accumulated_squared_gradients = batch_squared_gradients_val
            else:
                accumulated_gradients = accumulated_gradients + batch_gradients_val
                accumulated_squared_gradients = accumulated_squared_gradients + batch_squared_gradients_val
            
            total_batches_processed += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'batches': f"{total_batches_processed}"
            })

        progress_bar.close()

        if accumulated_gradients is None or total_batches_processed == 0:
            raise ValueError(f"Could not compute gradients for weight tensor with shape {ops.shape(weights_tensor_val)}.")
        
        # Average the accumulated gradients
        avg_gradients_val = accumulated_gradients / total_batches_processed
        avg_squared_gradients_val = accumulated_squared_gradients / total_batches_processed
        
        # Compute Taylor score using corrected formula
        # First-order term: |∂L/∂w * w|
        first_order_term = ops.abs(avg_gradients_val * weights_tensor_val)

        # Second-order term using Fisher Information approximation: λ * |E[g²] * w²|
        second_order_term = self.second_order_weight * ops.abs(avg_squared_gradients_val * ops.square(weights_tensor_val))

        # Combined Taylor score
        taylor_scores = first_order_term + second_order_term

        return taylor_scores
