from src.net.models.base import BaseModel
import tensorflow as tf

class SimpleCNN(tf.keras.Model):
    def __init__(self):
        super(SimpleCNN).__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28)),
        self.dense = tf.keras.layers.Dense(128, activation='relu'),
        self.out = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense(x)
        return self.out(x)