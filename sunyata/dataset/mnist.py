import keras
import numpy as np


def one_hot(indices, num_classes, dtype):
    assert indices.ndim == 1
    assert isinstance(num_classes, int)
    assert 0 < num_classes
    x = np.zeros((len(indices), num_classes), dtype)
    x[np.arange(len(indices)), indices] = 1
    return x


def scale_pixels(x):
    return (x / 255 - 0.5) * 2


def load_mnist(dtype):
    (x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, 1).astype(dtype)
    x_train = scale_pixels(x_train)
    y_train = one_hot(y_train, 10, dtype)
    x_val = np.expand_dims(x_val, 1).astype(dtype)
    x_val = scale_pixels(x_val)
    y_val = one_hot(y_val, 10, dtype)
    return (x_train, y_train), (x_val, y_val)
