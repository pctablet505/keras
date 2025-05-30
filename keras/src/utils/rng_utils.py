import random

import numpy as np

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state
from keras.src.utils.module_utils import tensorflow as tf

GLOBAL_RANDOM_SEED = "global_random_seed"


@keras_export("keras.utils.set_random_seed")
def set_random_seed(seed):
    """Sets all random seeds (Python, NumPy, and backend framework, e.g. TF).

    You can use this utility to make almost any Keras program fully
    deterministic. Some limitations apply in cases where network communications
    are involved (e.g. parameter server distribution), which creates additional
    sources of randomness, or when certain non-deterministic cuDNN ops are
    involved.

    Calling this utility is equivalent to the following:

    ```python
    import random
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    import tensorflow as tf  # Only if TF is installed
    tf.random.set_seed(seed)

    import torch  # Only if the backend is 'torch'
    torch.manual_seed(seed)
    ```

    Note that the TensorFlow seed is set even if you're not using TensorFlow
    as your backend framework, since many workflows leverage `tf.data`
    pipelines (which feature random shuffling). Likewise many workflows
    might leverage NumPy APIs.

    Arguments:
        seed: Integer, the random seed to use.
    """
    if not isinstance(seed, int):
        raise ValueError(
            "Expected `seed` argument to be an integer. "
            f"Received: seed={seed} (of type {type(seed)})"
        )

    # Store seed in global state so we can query it if set.
    global_state.set_global_attribute(GLOBAL_RANDOM_SEED, seed)
    random.seed(seed)
    np.random.seed(seed)
    if tf.available:
        tf.random.set_seed(seed)
    if backend.backend() == "torch":
        import torch

        torch.manual_seed(seed)


def get_random_seed():
    """Returns the explicit integer random seed if set.

    If the seed has been explicitly set via `set_random_seed`, then
    returns the seed.  Otherwise, returns `None`.
    """
    return global_state.get_global_attribute(GLOBAL_RANDOM_SEED)
