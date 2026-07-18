"""
efficientnet.py

Transfer learning model using EfficientNetB0.
"""

import tensorflow as tf


def build_efficientnet(num_classes: int):
    """
    Builds an EfficientNetB0 transfer learning model.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.Model: Compiled EfficientNetB0 model.
    """

    # Load pretrained EfficientNetB0 without the classification head
    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )

    # Freeze pretrained layers
    base_model.trainable = False

    # Model architecture
    inputs = tf.keras.Input(shape=(224, 224, 3))

    # EfficientNet preprocessing
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)

    # Feature extraction
    x = base_model(x, training=False)

    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Regularization
    x = tf.keras.layers.Dropout(0.3)(x)

    # Classification layer
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model