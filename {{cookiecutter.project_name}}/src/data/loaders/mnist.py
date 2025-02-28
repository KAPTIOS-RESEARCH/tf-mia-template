import tensorflow as tf

class MNISTLoader:
    def __init__(self, input_size=(28, 28), batch_size=8, debug=False):
        self.input_size = input_size
        self.batch_size = batch_size
        self.debug = debug
        
        # Load MNIST dataset
        (x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
        
        # Normalize and expand dims for channels
        x_train, x_val = x_train / 255.0, x_val / 255.0
        x_train, x_val = x_train[..., tf.newaxis], x_val[..., tf.newaxis]
        
        # Convert to TensorFlow datasets
        self.train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        
        if self.debug:
            self.train_dataset = self.train_dataset.take(100)
            self.val_dataset = self.val_dataset.take(100)

    def preprocess(self, image, label):
        image = tf.image.resize(image, self.input_size)  # Resize if needed
        return image, label
    
    def train(self):
        return (self.train_dataset
                .map(self.preprocess)
                .shuffle(1000)
                .batch(self.batch_size)
                .prefetch(tf.data.AUTOTUNE))
    
    def val(self):
        return (self.val_dataset
                .map(self.preprocess)
                .batch(self.batch_size)
                .prefetch(tf.data.AUTOTUNE))
    
    def test(self):
        return self.val()