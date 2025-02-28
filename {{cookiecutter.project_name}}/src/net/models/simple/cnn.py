import tensorflow as tf

class SimpleCNN(tf.keras.Model):
    def __init__(self, in_shape: tuple = (28, 28, 1), out_channels: int = 10):
        super().__init__()
        self.in_shape = in_shape
        self.out_channels = out_channels
        self.net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=in_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(out_channels, activation='softmax'),
        ])

    def call(self, x):
        return self.net(x)