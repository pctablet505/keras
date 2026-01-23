from keras.src import backend
from keras.src import layers
from keras.src.api_export import keras_export
from keras.src.applications import imagenet_utils
from keras.src.models import Functional
from keras.src.ops import operation_utils
from keras.src.utils import file_utils


def _conv_block(x, filters, name, kernel_size=3, activation="relu"):
    """Apply two convolutional layers with batch normalization.

    Args:
        x: Input tensor.
        filters: Number of filters for the convolution layers.
        name: Name prefix for the layers.
        kernel_size: Size of the convolutional kernel. Defaults to 3.
        activation: Activation function. Defaults to "relu".

    Returns:
        Output tensor after applying two conv layers.
    """
    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        use_bias=False,
        name=f"{name}_conv1",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation(activation, name=f"{name}_relu1")(x)

    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        use_bias=False,
        name=f"{name}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation(activation, name=f"{name}_relu2")(x)

    return x


def _encoder_block(x, filters, name, pool=True):
    """Encoder block with optional pooling.

    Args:
        x: Input tensor.
        filters: Number of filters.
        name: Name prefix for the layers.
        pool: Whether to apply max pooling. Defaults to True.

    Returns:
        Tuple of (pooled tensor, skip connection tensor).
    """
    skip = _conv_block(x, filters, name)

    if pool:
        x = layers.MaxPooling2D((2, 2), name=f"{name}_pool")(skip)
        return x, skip
    else:
        return skip


def _decoder_block(x, skip, filters, name):
    """Decoder block with upsampling and skip connections.

    Args:
        x: Input tensor.
        skip: Skip connection tensor from encoder.
        filters: Number of filters.
        name: Name prefix for the layers.

    Returns:
        Output tensor after upsampling and concatenation.
    """
    x = layers.Conv2DTranspose(
        filters,
        (2, 2),
        strides=(2, 2),
        padding="same",
        name=f"{name}_upsample",
    )(x)
    # Concatenate along channel axis (axis=1 for channels_first, axis=-1 for channels_last)
    if backend.image_data_format() == "channels_first":
        concat_axis = 1
    else:
        concat_axis = -1
    x = layers.Concatenate(axis=concat_axis, name=f"{name}_concat")([x, skip])
    x = _conv_block(x, filters, name)

    return x


@keras_export(["keras.applications.unet.UNet", "keras.applications.UNet"])
def UNet(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1,
    classifier_activation="sigmoid",
    name="unet",
    depth=5,
    base_filters=64,
):
    """Instantiates the U-Net architecture.

    Reference:
    - [U-Net: Convolutional Networks for Biomedical Image Segmentation](
    https://arxiv.org/abs/1505.04597) (MICCAI 2015)

    U-Net is a convolutional neural network architecture for fast and precise
    segmentation of images. It is particularly popular in medical image
    segmentation tasks.

    The default input size for this model is 256x256.

    Args:
        include_top: Whether to include the output layer at the top of the
            network. If `False`, the model outputs the bottleneck features.
        weights: One of `None` (random initialization) or the path to the
            weights file to be loaded. Note: ImageNet weights are not
            available for U-Net.
        input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: Optional shape tuple, only to be specified if
            `include_top` is `False` (otherwise the input shape has to be
            `(256, 256, 3)` (with `"channels_last"` data format) or
            `(3, 256, 256)` (with `"channels_first"` data format). It should
            have exactly 3 input channels, and width and height should be
            divisible by 2^(depth-1). E.g. `(128, 128, 3)` would be one valid
            value.
        pooling: Optional pooling mode for feature extraction when
            `include_top` is `False`.
            - `None` means that the output of the model will be the 4D tensor
                output of the last decoder block.
            - `avg` means that global average pooling will be applied to the
                output of the last decoder block, and thus the output of the
                model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        classes: Optional number of classes or output channels to segment,
            only to be specified if `include_top` is `True`. Defaults to 1
            for binary segmentation.
        classifier_activation: A `str` or callable. The activation function to
            use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top"
            layer. Defaults to `"sigmoid"`.
        name: The name of the model (string).
        depth: The depth of the U-Net (number of encoder/decoder blocks).
            Defaults to 5 (as in the original paper).
        base_filters: Number of filters in the first encoder block. Each
            subsequent encoder block doubles the number of filters. Defaults
            to 64.

    Returns:
        A `Model` instance.
    """
    if weights is not None and not file_utils.exists(weights):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), "
            "or the path to the weights file to be loaded. "
            f"Received: weights={weights}"
        )

    # Determine proper input shape
    if input_shape is None:
        default_size = 256
    else:
        if backend.image_data_format() == "channels_last":
            rows = input_shape[0] if len(input_shape) >= 1 else None
            cols = input_shape[1] if len(input_shape) >= 2 else None
        else:
            rows = input_shape[1] if len(input_shape) >= 2 else None
            cols = input_shape[2] if len(input_shape) >= 3 else None

        if rows and cols:
            default_size = rows
        else:
            default_size = 256

    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=2 ** (depth - 1),
        data_format=backend.image_data_format(),
        require_flatten=False,
        weights=weights,
    )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Encoder
    x = img_input
    skips = []

    for i in range(depth - 1):
        filters = base_filters * (2**i)
        x, skip = _encoder_block(x, filters, f"encoder{i + 1}", pool=True)
        skips.append(skip)

    # Bottleneck
    filters = base_filters * (2 ** (depth - 1))
    x = _encoder_block(x, filters, f"encoder{depth}", pool=False)

    # Decoder
    for i in range(depth - 2, -1, -1):
        filters = base_filters * (2**i)
        x = _decoder_block(x, skips[i], filters, f"decoder{i + 1}")

    if include_top:
        # Output layer
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Conv2D(
            classes,
            (1, 1),
            activation=classifier_activation,
            padding="same",
            name="output",
        )(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = operation_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Functional(inputs, x, name=name)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model


@keras_export("keras.applications.unet.preprocess_input")
def preprocess_input(x, data_format=None):
    """Preprocesses a tensor or Numpy array encoding a batch of images.

    Args:
        x: Input Numpy or symbolic tensor, 3D or 4D.
        data_format: Data format of the image tensor/array.

    Returns:
        Preprocessed tensor or Numpy array.
    """
    return imagenet_utils.preprocess_input(
        x, data_format=data_format, mode="torch"
    )


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode="",
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TORCH,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC,
)
