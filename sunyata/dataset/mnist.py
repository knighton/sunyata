import gzip
import numpy as np
import os
import pickle

from .base import download, get_dataset_dir


_NAME = 'mnist'
_URL = 'https://s3.amazonaws.com/img-datasets/mnist.pkl.gz'


def _one_hot(indices, num_classes, dtype):
    assert indices.ndim == 1
    assert isinstance(num_classes, int)
    assert 0 < num_classes
    x = np.zeros((len(indices), num_classes), dtype)
    x[np.arange(len(indices)), indices] = 1
    return x


def _scale_pixels(x):
    return (x / 255 - 0.5) * 2


def load_mnist(dtype, verbose=2):
    dataset_dir = get_dataset_dir(_NAME)
    local = os.path.join(dataset_dir, os.path.basename(_URL))
    if not os.path.exists(local):
        download(_URL, local, verbose)
    (x_train, y_train), (x_val, y_val) = \
        pickle.load(gzip.open(local), encoding='latin1')
    x_train = np.expand_dims(x_train, 1).astype(dtype)
    x_train = _scale_pixels(x_train)
    y_train = _one_hot(y_train, 10, dtype)
    x_val = np.expand_dims(x_val, 1).astype(dtype)
    x_val = _scale_pixels(x_val)
    y_val = _one_hot(y_val, 10, dtype)
    return (x_train, y_train), (x_val, y_val)
