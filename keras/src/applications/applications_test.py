import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.applications import convnext
from keras.src.applications import densenet
from keras.src.applications import efficientnet
from keras.src.applications import efficientnet_v2
from keras.src.applications import inception_resnet_v2
from keras.src.applications import inception_v3
from keras.src.applications import mobilenet
from keras.src.applications import mobilenet_v2
from keras.src.applications import mobilenet_v3
from keras.src.applications import nasnet
from keras.src.applications import resnet
from keras.src.applications import resnet_v2
from keras.src.applications import unet
from keras.src.applications import vgg16
from keras.src.applications import vgg19
from keras.src.applications import xception
from keras.src.layers import Conv2D
from keras.src.layers import Input
from keras.src.saving import serialization_lib
from keras.src.utils import file_utils
from keras.src.utils import image_utils

try:
    import PIL
except ImportError:
    PIL = None

MODEL_LIST = [
    # vgg
    (vgg16.VGG16, 512, vgg16),
    (vgg19.VGG19, 512, vgg19),
    # xception
    (xception.Xception, 2048, xception),
    # inception
    (inception_v3.InceptionV3, 2048, inception_v3),
    (inception_resnet_v2.InceptionResNetV2, 1536, inception_resnet_v2),
    # mobilenet
    (mobilenet.MobileNet, 1024, mobilenet),
    (mobilenet_v2.MobileNetV2, 1280, mobilenet_v2),
    (mobilenet_v3.MobileNetV3Small, 576, mobilenet_v3),
    (mobilenet_v3.MobileNetV3Large, 960, mobilenet_v3),
    # efficientnet
    (efficientnet.EfficientNetB0, 1280, efficientnet),
    (efficientnet.EfficientNetB1, 1280, efficientnet),
    (efficientnet.EfficientNetB2, 1408, efficientnet),
    (efficientnet.EfficientNetB3, 1536, efficientnet),
    (efficientnet.EfficientNetB4, 1792, efficientnet),
    (efficientnet.EfficientNetB5, 2048, efficientnet),
    (efficientnet.EfficientNetB6, 2304, efficientnet),
    (efficientnet.EfficientNetB7, 2560, efficientnet),
    (efficientnet_v2.EfficientNetV2B0, 1280, efficientnet_v2),
    (efficientnet_v2.EfficientNetV2B1, 1280, efficientnet_v2),
    (efficientnet_v2.EfficientNetV2B2, 1408, efficientnet_v2),
    (efficientnet_v2.EfficientNetV2B3, 1536, efficientnet_v2),
    (efficientnet_v2.EfficientNetV2S, 1280, efficientnet_v2),
    (efficientnet_v2.EfficientNetV2M, 1280, efficientnet_v2),
    (efficientnet_v2.EfficientNetV2L, 1280, efficientnet_v2),
    # densenet
    (densenet.DenseNet121, 1024, densenet),
    (densenet.DenseNet169, 1664, densenet),
    (densenet.DenseNet201, 1920, densenet),
    # convnext
    (convnext.ConvNeXtTiny, 768, convnext),
    (convnext.ConvNeXtSmall, 768, convnext),
    (convnext.ConvNeXtBase, 1024, convnext),
    (convnext.ConvNeXtLarge, 1536, convnext),
    (convnext.ConvNeXtXLarge, 2048, convnext),
    # nasnet
    (nasnet.NASNetMobile, 1056, nasnet),
    (nasnet.NASNetLarge, 4032, nasnet),
    # resnet
    (resnet.ResNet50, 2048, resnet),
    (resnet.ResNet101, 2048, resnet),
    (resnet.ResNet152, 2048, resnet),
    (resnet_v2.ResNet50V2, 2048, resnet_v2),
    (resnet_v2.ResNet101V2, 2048, resnet_v2),
    (resnet_v2.ResNet152V2, 2048, resnet_v2),
]
MODELS_UNSUPPORTED_CHANNELS_FIRST = ["ConvNeXt", "DenseNet", "NASNet"]

# Add names for `named_parameters`, and add each data format for each model
test_parameters = [
    (
        "{}_{}".format(model[0].__name__, image_data_format),
        *model,
        image_data_format,
    )
    for image_data_format in ["channels_first", "channels_last"]
    for model in MODEL_LIST
]


def _get_elephant(target_size):
    # For models that don't include a Flatten step,
    # the default is to accept variable-size inputs
    # even when loading ImageNet weights (since it is possible).
    # In this case, default to 299x299.
    TEST_IMAGE_PATH = (
        "https://storage.googleapis.com/tensorflow/"
        "keras-applications/tests/elephant.jpg"
    )

    if target_size[0] is None:
        target_size = (299, 299)
    test_image = file_utils.get_file("elephant.jpg", TEST_IMAGE_PATH)
    img = image_utils.load_img(test_image, target_size=tuple(target_size))
    x = image_utils.img_to_array(img)
    return np.expand_dims(x, axis=0)


@pytest.mark.skipif(
    os.environ.get("SKIP_APPLICATIONS_TESTS"),
    reason="Env variable set to skip.",
)
@pytest.mark.requires_trainable_backend
class ApplicationsTest(testing.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_image_data_format = backend.image_data_format()

    @classmethod
    def tearDownClass(cls):
        backend.set_image_data_format(cls.original_image_data_format)

    def skip_if_invalid_image_data_format_for_model(
        self, app, image_data_format
    ):
        does_not_support_channels_first = any(
            [
                unsupported_name.lower() in app.__name__.lower()
                for unsupported_name in MODELS_UNSUPPORTED_CHANNELS_FIRST
            ]
        )
        if (
            image_data_format == "channels_first"
            and does_not_support_channels_first
        ):
            self.skipTest(
                "{} does not support channels first".format(app.__name__)
            )

    @parameterized.named_parameters(test_parameters)
    def test_application_notop_variable_input_channels(
        self, app, last_dim, _, image_data_format
    ):
        if app == nasnet.NASNetMobile and backend.backend() == "torch":
            self.skipTest(
                "NASNetMobile pretrained incorrect with torch backend."
            )
        self.skip_if_invalid_image_data_format_for_model(app, image_data_format)
        backend.set_image_data_format(image_data_format)

        # Test compatibility with 1 channel
        if image_data_format == "channels_first":
            input_shape = (1, None, None)
            correct_output_shape = [None, last_dim, None, None]
        else:
            input_shape = (None, None, 1)
            correct_output_shape = [None, None, None, last_dim]

        model = app(weights=None, include_top=False, input_shape=input_shape)
        output_shape = list(model.outputs[0].shape)
        self.assertEqual(output_shape, correct_output_shape)

        # Test compatibility with 4 channels
        if image_data_format == "channels_first":
            input_shape = (4, None, None)
        else:
            input_shape = (None, None, 4)
        model = app(weights=None, include_top=False, input_shape=input_shape)
        output_shape = list(model.outputs[0].shape)
        self.assertEqual(output_shape, correct_output_shape)

    @parameterized.named_parameters(test_parameters)
    @pytest.mark.skipif(PIL is None, reason="Requires PIL.")
    def test_application_base(self, app, _, app_module, image_data_format):
        import tensorflow as tf

        if app == nasnet.NASNetMobile and backend.backend() == "torch":
            self.skipTest(
                "NASNetMobile pretrained incorrect with torch backend."
            )
        if (
            image_data_format == "channels_first"
            and len(tf.config.list_physical_devices("GPU")) == 0
            and backend.backend() == "tensorflow"
        ):
            self.skipTest(
                "Conv2D doesn't support channels_first using CPU with "
                "tensorflow backend"
            )
        self.skip_if_invalid_image_data_format_for_model(app, image_data_format)
        backend.set_image_data_format(image_data_format)

        # Can be instantiated with default arguments
        model = app(weights="imagenet")

        # Can run a correct inference on a test image
        if image_data_format == "channels_first":
            shape = model.input_shape[2:4]
        else:
            shape = model.input_shape[1:3]
        x = _get_elephant(shape)

        x = app_module.preprocess_input(x)
        preds = model.predict(x)
        names = [p[1] for p in app_module.decode_predictions(preds)[0]]
        # Test correct label is in top 3 (weak correctness test).
        self.assertIn("African_elephant", names[:3])

        # Can be serialized and deserialized
        config = serialization_lib.serialize_keras_object(model)
        reconstructed_model = serialization_lib.deserialize_keras_object(config)
        self.assertEqual(len(model.weights), len(reconstructed_model.weights))

    @parameterized.named_parameters(test_parameters)
    def test_application_notop_custom_input_shape(
        self, app, last_dim, _, image_data_format
    ):
        if app == nasnet.NASNetMobile and backend.backend() == "torch":
            self.skipTest(
                "NASNetMobile pretrained incorrect with torch backend."
            )
        self.skip_if_invalid_image_data_format_for_model(app, image_data_format)
        backend.set_image_data_format(image_data_format)

        if image_data_format == "channels_first":
            input_shape = (3, 123, 123)
            last_dim_axis = 1
        else:
            input_shape = (123, 123, 3)
            last_dim_axis = -1
        model = app(weights=None, include_top=False, input_shape=input_shape)
        output_shape = list(model.outputs[0].shape)
        self.assertEqual(output_shape[last_dim_axis], last_dim)

    @parameterized.named_parameters(test_parameters)
    def test_application_notop_custom_input_tensor(
        self, app, last_dim, _, image_data_format
    ):
        if app == nasnet.NASNetMobile and backend.backend() == "torch":
            self.skipTest(
                "NASNetMobile pretrained incorrect with torch backend."
            )
        self.skip_if_invalid_image_data_format_for_model(app, image_data_format)
        backend.set_image_data_format(image_data_format)

        if image_data_format == "channels_first":
            input_shape = (4, 123, 123)
            last_dim_axis = 1
        else:
            input_shape = (123, 123, 4)
            last_dim_axis = -1

        inputs_custom = Input(shape=input_shape, name="custom_input")
        inputs_custom = Conv2D(3, (2, 2), padding="valid", strides=(2, 2))(
            inputs_custom
        )
        model = app(weights=None, include_top=False, input_tensor=inputs_custom)
        output_shape = list(model.outputs[0].shape)
        self.assertEqual(output_shape[last_dim_axis], last_dim)

    @parameterized.named_parameters(test_parameters)
    def test_application_pooling(self, app, last_dim, _, image_data_format):
        if app == nasnet.NASNetMobile and backend.backend() == "torch":
            self.skipTest(
                "NASNetMobile pretrained incorrect with torch backend."
            )
        self.skip_if_invalid_image_data_format_for_model(app, image_data_format)
        backend.set_image_data_format(image_data_format)

        model = app(weights=None, include_top=False, pooling="max")
        output_shape = list(model.outputs[0].shape)
        self.assertEqual(output_shape, [None, last_dim])

    @parameterized.named_parameters(test_parameters)
    def test_application_classifier_activation(self, app, *_):
        if app == nasnet.NASNetMobile and backend.backend() == "torch":
            self.skipTest(
                "NASNetMobile pretrained incorrect with torch backend."
            )

        model = app(
            weights=None, include_top=True, classifier_activation="softmax"
        )
        last_layer_act = model.layers[-1].activation.__name__
        self.assertEqual(last_layer_act, "softmax")


@pytest.mark.skipif(
    os.environ.get("SKIP_APPLICATIONS_TESTS"),
    reason="Env variable set to skip.",
)
@pytest.mark.requires_trainable_backend
class UNetTest(testing.TestCase):
    """Test suite specifically for U-Net model."""

    @classmethod
    def setUpClass(cls):
        cls.original_image_data_format = backend.image_data_format()

    @classmethod
    def tearDownClass(cls):
        backend.set_image_data_format(cls.original_image_data_format)

    def test_unet_basic_instantiation(self):
        """Test basic UNet instantiation with default parameters."""
        model = unet.UNet(weights=None, include_top=True)
        self.assertIsNotNone(model)
        # UNet preserves spatial dimensions, so with default None input size
        # the output spatial dims are also None
        self.assertEqual(
            model.output_shape[-1], 2
        )  # 2 classes for binary segmentation

    def test_unet_custom_input_shape(self):
        """Test UNet with custom input shape."""
        backend.set_image_data_format("channels_last")
        model = unet.UNet(
            weights=None, include_top=True, input_shape=(128, 128, 3)
        )
        self.assertEqual(model.output_shape, (None, 128, 128, 2))

    def test_unet_custom_classes(self):
        """Test UNet with custom number of output classes."""
        backend.set_image_data_format("channels_last")
        model = unet.UNet(
            weights=None, include_top=True, classes=3, input_shape=(128, 128, 3)
        )
        self.assertEqual(model.output_shape, (None, 128, 128, 3))

    def test_unet_notop(self):
        """Test UNet without top layer."""
        backend.set_image_data_format("channels_last")
        model = unet.UNet(
            weights=None, include_top=False, input_shape=(128, 128, 3)
        )
        # Output should be from last decoder block, not a single channel
        self.assertEqual(model.output_shape[0], None)
        self.assertEqual(model.output_shape[1:3], (128, 128))

    def test_unet_notop_with_pooling(self):
        """Test UNet without top but with pooling."""
        backend.set_image_data_format("channels_last")
        model = unet.UNet(
            weights=None,
            include_top=False,
            input_shape=(128, 128, 3),
            pooling="avg",
        )
        # With pooling, output should be 2D
        self.assertEqual(len(model.output_shape), 2)
        self.assertEqual(model.output_shape[0], None)

    def test_unet_custom_depth(self):
        """Test UNet with custom depth."""
        backend.set_image_data_format("channels_last")
        model = unet.UNet(
            weights=None,
            include_top=True,
            input_shape=(128, 128, 3),
            depth=4,
        )
        self.assertIsNotNone(model)

    def test_unet_custom_base_filters(self):
        """Test UNet with custom base filters."""
        backend.set_image_data_format("channels_last")
        model = unet.UNet(
            weights=None,
            include_top=True,
            input_shape=(128, 128, 3),
            base_filters=32,
        )
        self.assertIsNotNone(model)

    def test_unet_channels_first(self):
        """Test UNet with channels_first data format."""
        backend.set_image_data_format("channels_first")
        model = unet.UNet(
            weights=None, include_top=True, input_shape=(3, 128, 128)
        )
        self.assertEqual(model.output_shape, (None, 2, 128, 128))
        backend.set_image_data_format("channels_last")

    def test_unet_preprocess_input(self):
        """Test UNet preprocess_input function."""
        x = np.random.random((1, 128, 128, 3)) * 255  # Scale to [0, 255] range
        processed = unet.preprocess_input(x)
        self.assertEqual(processed.shape, x.shape)
        # Check that values are scaled to [0, 1]
        self.assertTrue(np.all(processed >= 0))
        self.assertTrue(np.all(processed <= 1))

    def test_unet_serialization(self):
        """Test UNet model serialization."""
        backend.set_image_data_format("channels_last")
        model = unet.UNet(
            weights=None, include_top=True, input_shape=(128, 128, 3)
        )
        config = serialization_lib.serialize_keras_object(model)
        reconstructed_model = serialization_lib.deserialize_keras_object(config)
        self.assertEqual(len(model.weights), len(reconstructed_model.weights))

    def test_unet_training_inference(self):
        """Test UNet training and inference with different batch sizes."""
        backend.set_image_data_format("channels_last")

        # Create model with smaller input for faster testing
        model = unet.UNet(
            weights=None, include_top=True, input_shape=(64, 64, 3), classes=2
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Generate dummy training data - smaller dataset
        batch_sizes = [2, 4]  # Reduced batch sizes to test
        num_samples = 4  # Reduced samples

        for batch_size in batch_sizes:
            print(f"\nTesting with batch size {batch_size}")

            # Create dummy data with smaller size
            x_train = np.random.random((num_samples, 64, 64, 3)).astype(
                np.float32
            )
            y_train = np.random.randint(0, 2, (num_samples, 64, 64)).astype(
                np.int32
            )  # Integer labels for sparse_categorical_crossentropy

            # Train for 2 epochs (first is slow due to graph tracing)
            history = model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=2,  # Need at least 2 epochs
                verbose=0,
            )

            # Verify training worked
            self.assertIn("loss", history.history)
            self.assertIn("accuracy", history.history)

            # Test inference
            x_test = np.random.random((batch_size, 64, 64, 3)).astype(
                np.float32
            )
            predictions = model.predict(
                x_test, verbose=0, batch_size=batch_size
            )

            # Verify predictions
            self.assertEqual(predictions.shape, (batch_size, 64, 64, 2))
            self.assertTrue(
                np.all(predictions >= 0)
                and np.all(predictions <= 1)
                and np.allclose(np.sum(predictions, axis=-1), 1.0)
            )  # Softmax output - probabilities sum to 1

            print(
                f"Batch size {batch_size}: loss decreased from "
                f"{history.history['loss'][0]:.4f} to "
                f"{history.history['loss'][-1]:.4f}"
            )

    def test_unet_training_different_classes(self):
        """Test UNet training with multiple classes."""
        backend.set_image_data_format("channels_last")

        # Test with 3 classes
        model = unet.UNet(
            weights=None, include_top=True, input_shape=(128, 128, 3), classes=3
        )
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Generate dummy data
        num_samples = 8
        x_train = np.random.random((num_samples, 128, 128, 3)).astype(
            np.float32
        )
        y_train = np.random.random((num_samples, 128, 128, 3)).astype(
            np.float32
        )  # Random multiclass targets

        # Train
        history = model.fit(x_train, y_train, batch_size=2, epochs=2, verbose=0)

        # Verify
        self.assertIn("loss", history.history)
        self.assertLess(history.history["loss"][-1], history.history["loss"][0])

        # Test inference
        predictions = model.predict(x_train[:2], verbose=0)
        self.assertEqual(predictions.shape, (2, 128, 128, 3))

    def test_unet_inference_loops(self):
        """Test UNet inference with multiple loops and different input sizes."""
        backend.set_image_data_format("channels_last")

        # Test different input sizes that are powers of 2
        input_sizes = [(64, 64), (128, 128), (256, 256)]

        for height, width in input_sizes:
            print(f"\nTesting inference with input size {height}x{width}")

            # Create model for this size (since UNet preserves spatial dims)
            test_model = unet.UNet(
                weights=None, include_top=True, input_shape=(height, width, 3)
            )

            # Run multiple inference loops
            for loop in range(3):
                x = np.random.random((2, height, width, 3)).astype(np.float32)
                y = test_model.predict(x, verbose=0)

                self.assertEqual(y.shape, (2, height, width, 2))
                print(f"Loop {loop + 1}: output shape {y.shape}")

    def test_unet_training_validation(self):
        """Test UNet training with validation and metrics monitoring."""
        backend.set_image_data_format("channels_last")

        model = unet.UNet(
            weights=None, include_top=True, input_shape=(64, 64, 3), classes=2
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],  # Reduced metrics for speed
        )

        # Generate smaller dataset
        num_samples = 8  # Much smaller dataset
        x_data = np.random.random((num_samples, 64, 64, 3)).astype(np.float32)
        y_data = np.random.randint(0, 2, (num_samples, 64, 64)).astype(
            np.int32
        )  # Integer labels for sparse_categorical_crossentropy

        # Train with validation - at least 2 epochs for proper testing
        history = model.fit(
            x_data,
            y_data,
            batch_size=4,
            epochs=3,  # At least 2 epochs, 3 for safety
            verbose=0,
            validation_split=0.3,
        )

        # Verify essential metrics are present
        required_metrics = ["loss", "accuracy"]
        for metric in required_metrics:
            self.assertIn(metric, history.history)
            self.assertIn(f"val_{metric}", history.history)

        # Verify loss decreased
        self.assertLess(history.history["loss"][-1], history.history["loss"][0])

        loss_final = history.history["loss"][-1]
        val_loss_final = history.history["val_loss"][-1]
        print(
            f"Training completed: final loss {loss_final:.4f}, "
            f"val_loss {val_loss_final:.4f}"
        )
