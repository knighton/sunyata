import gzip
import numpy as np
import os
import pickle

from .base import download, get_dataset_dir, kwargs_only, scale_pixels, \
    to_one_hot


_DATASET_NAME = 'mnist'
_URL = 'https://s3.amazonaws.com/img-datasets/mnist.pkl.gz'


def _transform_x(x, scale, dtype):
    x = np.expand_dims(x, 1).astype(dtype)
    if scale:
        x = scale_pixels(x)
    return x


def _transform_y(y, one_hot, dtype):
    if one_hot:
        y = to_one_hot(y, 10, dtype)
    else:
        y = y.astype(dtype)
    return y


@kwargs_only
def load_mnist(dataset_name=_DATASET_NAME, one_hot=True, scale=True,
               x_dtype='float32', y_dtype='float32', url=_URL, verbose=2):
    dataset_dir = get_dataset_dir(dataset_name)
    local = os.path.join(dataset_dir, os.path.basename(url))
    if not os.path.exists(local):
        download(url, local, verbose)
    (x_train, y_train), (x_val, y_val) = \
        pickle.load(gzip.open(local), encoding='latin1')
    x_train = _transform_x(x_train, scale, x_dtype)
    y_train = _transform_y(y_train, one_hot, y_dtype)
    x_val = _transform_x(x_val, scale, x_dtype)
    y_val = _transform_y(y_val, one_hot, y_dtype)
    return (x_train, y_train), (x_val, y_val)
