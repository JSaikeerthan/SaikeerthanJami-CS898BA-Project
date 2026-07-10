"""
cnn.py

Baseline CNN model.
"""

import tensorflow as tf


def build_cnn(num_classes: int):

    model = tf.keras.Sequential(

        [

            tf.keras.layers.Rescaling(1.0 / 255),

            tf.keras.layers.Conv2D(
                32,
                3,
                activation="relu"
            ),

            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(
                64,
                3,
                activation="relu"
            ),

            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(
                128,
                3,
                activation="relu"
            ),

            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(
                256,
                activation="relu"
            ),

            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(
                num_classes,
                activation="softmax"
            ),

        ]
    )

    model.compile(

        optimizer="adam",

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"],

    )

    return model