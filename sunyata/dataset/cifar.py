import numpy as np
import os
import pickle
import tarfile
from tqdm import tqdm

from .base import download, get_dataset_dir, kwargs_only, scale_pixels, \
    to_one_hot


_DATASET_NAME = 'cifar'
_CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
_CIFAR100_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'


def _transform_x(x, scale, dtype):
    x = x.reshape(-1, 3, 32, 32).astype(dtype)
    if scale:
        x = scale_pixels(x)
    return x


def _transform_y(y, one_hot, num_classes, dtype):
    if one_hot:
        y = np.array(y, 'int32')
        y = to_one_hot(y, num_classes, dtype)
    else:
        y = np.array(y, dtype)
    return y


def _load_cifar10_data(tar, one_hot, scale, x_dtype, y_dtype, verbose):
    if verbose == 2:
        bar = tqdm(total=5, leave=False)
    xx = []
    yy = []
    for info in tar.getmembers():
        if not info.isreg():
            continue
        if not info.path.startswith('cifar-10-batches-py/data_batch_'):
            continue
        data = tar.extractfile(info).read()
        obj = pickle.loads(data, encoding='bytes')
        x = obj[b'data']
        x = _transform_x(x, scale, x_dtype)
        y = obj[b'labels']
        y = _transform_y(y, one_hot, 10, y_dtype)
        xx.append(x)
        yy.append(y)
        if verbose == 2:
            bar.update(1)
    if verbose == 2:
        bar.close()
    x = np.vstack(xx)
    y = np.vstack(yy)
    return x, y


def _load_cifar10_class_names(tar):
    path = 'cifar-10-batches-py/batches.meta'
    data = tar.extractfile(path).read()
    obj = pickle.loads(data, encoding='bytes')
    labels = obj[b'label_names']
    return list(map(lambda s: s.decode('utf-8'), labels))


@kwargs_only
def load_cifar10(dataset_name=_DATASET_NAME, one_hot=True, scale=True,
                 url=_CIFAR10_URL, verbose=2, x_dtype='float32',
                 y_dtype='float32'):
    dataset_dir = get_dataset_dir(dataset_name)
    local = os.path.join(dataset_dir, os.path.basename(url))
    if not os.path.exists(local):
        download(url, local, verbose)
    tar = tarfile.open(local, 'r:gz')
    x, y = _load_cifar10_data(tar, one_hot, scale, x_dtype, y_dtype, verbose)
    class_names = _load_cifar10_class_names(tar)
    tar.close()
    return x, y, class_names


def _load_cifar100_split(tar, classes, one_hot, scale, x_dtype, y_dtype, split):
    path = 'cifar-100-python/%s' % split
    data = tar.extractfile(path).read()
    obj = pickle.loads(data, encoding='bytes')
    x = obj[b'data']
    x = _transform_x(x, scale, x_dtype)
    if classes == 20:
        key = b'coarse_labels'
    elif classes == 100:
        key = b'fine_labels'
    else:
        assert False
    y = obj[key]
    y = _transform_y(y, one_hot, classes, y_dtype)
    return x, y


def _load_cifar100_class_names(tar, classes):
    info = tar.getmember('cifar-100-python/meta')
    data = tar.extractfile(info).read()
    obj = pickle.loads(data, encoding='bytes')
    if classes == 20:
        key = b'coarse_label_names'
    elif classes == 100:
        key = b'fine_label_names'
    else:
        assert False
    labels = obj[key]
    return list(map(lambda s: s.decode('utf-8'), labels))


@kwargs_only
def load_cifar100(classes=100, dataset_name=_DATASET_NAME, one_hot=True,
                  scale=True, url=_CIFAR100_URL, verbose=2, x_dtype='float32',
                  y_dtype='float32'):
    dataset_dir = get_dataset_dir(dataset_name)
    local = os.path.join(dataset_dir, os.path.basename(url))
    if not os.path.exists(local):
        download(url, local, verbose)
    tar = tarfile.open(local, 'r:gz')
    train = _load_cifar100_split(
        tar, classes, one_hot, scale, x_dtype, y_dtype, 'train')
    val = _load_cifar100_split(
        tar, classes, one_hot, scale, x_dtype, y_dtype, 'test')
    class_names = _load_cifar100_class_names(tar, classes)
    tar.close()
    return train, val, class_names
