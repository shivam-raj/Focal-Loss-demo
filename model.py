from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow.keras as keras
import tensorflow as tf


def get_model(input_shape):
    base_model=keras.applications.Xception(input_shape=input_shape,include_top=False,weights='imagenet')
    base_model.trainable=False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = keras.layers.Dense(3,activation='sigmoid')
    fc1=keras.layers.Dense(128,activation='relu')
    fc2=keras.layers.Dense(512,activation='relu')
    model=keras.models.Sequential([
        base_model,
        global_average_layer,
        fc1,
        fc2,
        prediction_layer
    ])
    return model