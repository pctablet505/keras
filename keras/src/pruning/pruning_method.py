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

        # Compute saliency scores (|weight * gradient|)
        saliency_scores = self._compute_saliency_scores(weights, model, loss_fn, dataset) # Pass original weights

        flat_scores = ops.reshape(saliency_scores, [-1])
        total_size = int(backend.convert_to_numpy(ops.size(flat_scores)))
        k = int(sparsity_ratio * total_size)
        if k == 0:
            return ops.ones_like(weights_tensor, dtype="bool")

        sorted_scores = ops.sort(flat_scores)
        threshold = sorted_scores[k]

        mask = saliency_scores > threshold
        return mask

    def _compute_saliency_scores(self, weights, model, loss_fn, dataset):
        """Compute saliency scores using gradients.
        
        Saliency score = |gradient * weight| for each weight.
        This estimates how much the loss would change if we set that weight to zero.
        """
        import keras
        import numpy as np
        
        # Extract input and target data from dataset
        if isinstance(dataset, tuple) and len(dataset) == 2:
            x_data, y_data = dataset
        else:
            raise ValueError("Dataset must be a tuple (x_data, y_data) for saliency computation.")
        
        # Process data in smaller batches to avoid OOM
        # Limit batch size to avoid GPU memory issues
        if hasattr(x_data, 'shape') and len(x_data.shape) > 0:
            total_samples = x_data.shape[0]
            max_batch_size = min(32, total_samples)  # Use small batches to avoid OOM
            
            # Take a representative sample if dataset is very large
            if total_samples > max_batch_size:
                # Use random sampling for better gradient estimation
                indices = np.random.choice(total_samples, max_batch_size, replace=False)
                x_data = x_data[indices]
                y_data = y_data[indices]
        
        # Convert to tensors after sampling
        x_data = ops.convert_to_tensor(x_data)
        y_data = ops.convert_to_tensor(y_data)
        
        # Use backend-specific gradient computation for efficiency and accuracy
        from keras.src import backend as keras_backend
        backend_name = keras_backend.backend()
        
        trainable_weights = [v for v in model.trainable_variables if len(v.shape) > 1]
        weights_tensor = weights.value() if hasattr(weights, 'value') and callable(weights.value) else weights

        if backend_name == "tensorflow":
            import tensorflow as tf
            
            def compute_loss():
                predictions = model(x_data, training=False)
                if callable(loss_fn):
                    loss = loss_fn(y_data, predictions)
                else:
                    loss_obj = keras.losses.get(loss_fn)
                    loss = loss_obj(y_data, predictions)
                return ops.mean(loss) if len(ops.shape(loss)) > 0 else loss
            
            watch_vars = [v.value() if hasattr(v, 'value') and callable(v.value) else v.value if hasattr(v, 'value') else v for v in trainable_weights]
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(watch_vars)
                loss = compute_loss()
            
            all_gradients = tape.gradient(loss, watch_vars)
            
            target_gradients = None
            for i, weight in enumerate(trainable_weights):
                if ops.shape(weight) == ops.shape(weights_tensor):
                    weight_tensor_val = weight.value() if hasattr(weight, 'value') and callable(weight.value) else weight
                    if backend.convert_to_numpy(ops.mean(ops.abs(weight_tensor_val - weights_tensor))) < 1e-6:
                        target_gradients = all_gradients[i]
                        break
            
            if target_gradients is None:
                 raise ValueError(f"Could not find gradients for weight tensor with shape {ops.shape(weights_tensor)} in TensorFlow backend.")
            gradients = target_gradients

        elif backend_name == "jax":
            import jax
            
            def get_loss(weight_values):
                original_weights = [w.value() if hasattr(w, 'value') and callable(w.value) else w.value if hasattr(w, 'value') else w for w in trainable_weights]
                for var, new_w in zip(trainable_weights, weight_values):
                    var.assign(new_w)
                
                predictions = model(x_data, training=False)
                if callable(loss_fn):
                    loss = loss_fn(y_data, predictions)
                else:
                    loss_obj = keras.losses.get(loss_fn)
                    loss = loss_obj(y_data, predictions)
                
                for var, old_w in zip(trainable_weights, original_weights):
                    var.assign(old_w)
                    
                return ops.mean(loss) if len(ops.shape(loss)) > 0 else loss
            
            current_weights = [w.value() if hasattr(w, 'value') and callable(w.value) else w.value if hasattr(w, 'value') else w for w in trainable_weights]
            all_gradients = jax.grad(get_loss)(current_weights)
            
            target_gradients = None
            for i, weight_var in enumerate(trainable_weights):
                if ops.shape(weight_var) == ops.shape(weights_tensor):
                    weight_tensor_val = weight_var.value() if hasattr(weight_var, 'value') and callable(weight_var.value) else weight_var
                    if ops.mean(ops.abs(weight_tensor_val - weights_tensor)) < 1e-6:
                        target_gradients = all_gradients[i]
                        break
            
            if target_gradients is None:
                raise ValueError(f"Could not find gradients for weight tensor with shape {ops.shape(weights_tensor)} in JAX backend.")
            gradients = target_gradients

        elif backend_name == "torch":
            import torch
            
            torch_weights = []
            for var in trainable_weights:
                tensor = var.value() if hasattr(var, 'value') and callable(var.value) else var.value if hasattr(var, 'value') else var
                if hasattr(tensor, 'requires_grad') and not tensor.requires_grad:
                    tensor.requires_grad_(True)
                torch_weights.append(tensor)

            def compute_loss():
                predictions = model(x_data, training=False)
                if callable(loss_fn):
                    loss = loss_fn(y_data, predictions)
                else:
                    loss_obj = keras.losses.get(loss_fn)
                    loss = loss_obj(y_data, predictions)
                return ops.mean(loss) if len(ops.shape(loss)) > 0 else loss
            
            loss = compute_loss()
            all_gradients = torch.autograd.grad(loss, torch_weights, allow_unused=True)
            
            target_gradients = None
            for i, weight_var in enumerate(trainable_weights):
                if ops.shape(weight_var) == ops.shape(weights_tensor) and all_gradients[i] is not None:
                    weight_tensor_val = weight_var.value() if hasattr(weight_var, 'value') and callable(weight_var.value) else weight_var
                    if ops.mean(ops.abs(weight_tensor_val - weights_tensor)) < 1e-6:
                        target_gradients = all_gradients[i]
                        break
            
            if target_gradients is None:
                raise ValueError(f"Could not find gradients for weight tensor with shape {ops.shape(weights_tensor)} in PyTorch backend.")
            gradients = target_gradients
                
        else:
            raise ValueError(f"SaliencyPruning is not supported for backend '{backend_name}'.")
        
        gradients_val = gradients.value() if hasattr(gradients, 'value') and callable(gradients.value) else gradients
            
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
            use_fisher_approximation: If True, use Fisher Information approximation for Hessian.
                                    If False, fall back to simpler gradient-based approximation.
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

        # Compute Taylor scores
        taylor_scores = self._compute_taylor_scores(weights, model, loss_fn, dataset)

        flat_scores = ops.reshape(taylor_scores, [-1])
        total_size = int(backend.convert_to_numpy(ops.size(flat_scores)))
        k = int(sparsity_ratio * total_size)
        if k == 0:
            return ops.ones_like(weights_tensor, dtype="bool")

        sorted_scores = ops.sort(flat_scores)
        threshold = sorted_scores[k]

        mask = taylor_scores > threshold
        return mask

    def _compute_taylor_scores(self, weights, model, loss_fn, dataset):
        """Compute second-order Taylor expansion scores.
        
        Taylor score = |∂L/∂w * w| + λ * |H_ii * w|
        where H_ii is the diagonal of the Hessian matrix.
        
        For computational efficiency, we approximate H_ii using:
        1. Fisher Information Matrix: H_ii ≈ E[g_i²] where g_i = ∂L/∂w_i
        2. Or simple gradient magnitude: H_ii ≈ |g_i|
        """
        import keras
        import numpy as np
        
        # Extract input and target data from dataset
        if isinstance(dataset, tuple) and len(dataset) == 2:
            x_data, y_data = dataset
        else:
            raise ValueError("Dataset must be a tuple (x_data, y_data) for Taylor computation.")
        
        # Process data in smaller batches to avoid OOM
        if hasattr(x_data, 'shape') and len(x_data.shape) > 0:
            total_samples = x_data.shape[0]
            max_batch_size = min(32, total_samples)  # Use small batches to avoid OOM
            
            if total_samples > max_batch_size:
                indices = np.random.choice(total_samples, max_batch_size, replace=False)
                x_data = x_data[indices]
                y_data = y_data[indices]
        
        # Convert to tensors
        x_data = ops.convert_to_tensor(x_data)
        y_data = ops.convert_to_tensor(y_data)
        
        # Extract tensor values to ensure we work with tensors, not Variables
        weights_tensor_val = weights.value() if hasattr(weights, 'value') and callable(weights.value) else weights

        # Find the target weight variable by matching tensor values
        target_weight_var = None
        for var in model.trainable_variables:
            if hasattr(var, 'shape') and len(var.shape) > 1:  # Skip bias terms
                if ops.shape(var) == ops.shape(weights_tensor_val):
                    try:
                        var_tensor = var.value() if hasattr(var, 'value') and callable(var.value) else var
                        weight_diff = ops.mean(ops.abs(var_tensor - weights_tensor_val))
                        if backend.convert_to_numpy(weight_diff) < 1e-6:
                            target_weight_var = var
                            break
                    except:
                        target_weight_var = var
                        break
        
        if target_weight_var is None:
            raise ValueError(f"Could not find weight variable with shape {ops.shape(weights_tensor_val)}")
        
        # Backend-specific gradient computation
        from keras.src import backend as keras_backend
        backend_name = keras_backend.backend()
        
        def compute_loss():
            predictions = model(x_data, training=False)
            if callable(loss_fn):
                loss = loss_fn(y_data, predictions)
            else:
                loss_obj = keras.losses.get(loss_fn)
                loss = loss_obj(y_data, predictions)
            return ops.mean(loss) if len(ops.shape(loss)) > 0 else loss
        
        if backend_name == "tensorflow":
            import tensorflow as tf
            
            # Compute gradients
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                watch_var = target_weight_var.value() if hasattr(target_weight_var, 'value') and callable(target_weight_var.value) else target_weight_var.value if hasattr(target_weight_var, 'value') else target_weight_var
                tape.watch(watch_var)
                loss = compute_loss()
            
            gradients = tape.gradient(loss, watch_var)
            if gradients is None:
                raise ValueError("No gradients computed in TensorFlow backend.")
            
        elif backend_name == "jax":
            import jax
            
            def loss_fn_for_grad(weight_vals):
                old_weights = target_weight_var.value() if hasattr(target_weight_var, 'value') and callable(target_weight_var.value) else target_weight_var.value if hasattr(target_weight_var, 'value') else target_weight_var
                target_weight_var.assign(weight_vals)
                loss_val = compute_loss()
                target_weight_var.assign(old_weights)
                return loss_val
            
            gradients = jax.grad(loss_fn_for_grad)(weights_tensor_val)
            
        elif backend_name == "torch":
            import torch
            
            torch_var = target_weight_var.value() if hasattr(target_weight_var, 'value') and callable(target_weight_var.value) else target_weight_var
            if hasattr(torch_var, 'requires_grad'):
                torch_var.requires_grad_(True)
            
            loss = compute_loss()
            gradients = torch.autograd.grad(loss, torch_var, create_graph=False)[0]
            
        else:
            raise ValueError(f"TaylorPruning not supported for backend '{backend_name}'.")
        
        # Extract gradient values
        gradients_val = gradients.value() if hasattr(gradients, 'value') and callable(gradients.value) else gradients
        
        # Compute first-order term: |∂L/∂w * w|
        first_order_term = ops.abs(gradients_val * weights_tensor_val)
        
        # Compute second-order approximation
        if self.use_fisher_approximation:
            # Fisher Information approximation: Use squared gradients as Hessian diagonal approximation
            # This is more theoretically sound than using |gradient|
            hessian_diag_approx = ops.square(gradients_val) + 1e-8
        else:
            # Simple approximation: Use gradient magnitude
            hessian_diag_approx = ops.abs(gradients_val) + 1e-8
        
        # Second-order term: |H_ii * w^2|
        second_order_term = ops.abs(
            hessian_diag_approx * ops.square(weights_tensor_val)
        )
        
        # Combine terms with proper normalization to avoid scale issues
        # Normalize each term to have similar magnitude
        first_mean = ops.mean(first_order_term) + 1e-8
        second_mean = ops.mean(second_order_term) + 1e-8
        
        # Scale second term to be comparable to first term
        normalized_second_term = (second_order_term / second_mean) * first_mean
        
        # Final Taylor score
        taylor_scores = first_order_term + self.second_order_weight * normalized_second_term
        
        return taylor_scores
