"""
efficientnet.py

Transfer learning model using EfficientNetB0.
"""

import tensorflow as tf


def build_efficientnet(num_classes: int):

    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )

    # Freeze most layers
    base_model.trainable = True

    # Freeze all except the last 20 layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = tf.keras.applications.efficientnet.preprocess_input(inputs)

    x = base_model(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model