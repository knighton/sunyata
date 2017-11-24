from glob import glob
import numpy as np
import os
import tarfile
from time import time
from tqdm import tqdm

from .base import download, get_dataset_dir, kwargs_only
from ..transform import *  # noqa


_DATASET_NAME = 'imdb'
_URL = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
_TGZ_SUBDIR = 'aclImdb'
_PROC_SUBDIR = 'proc'


def _extract(dataset_dir, tgz_basename, verbose):
    local = os.path.join(dataset_dir, tgz_basename)
    if verbose:
        print('Extracting...')
        t0 = time()
    tar = tarfile.open(local, 'r:gz')
    tar.extractall(dataset_dir)
    if verbose:
        t = time() - t0
        print('...took %.3f sec.' % t)


def _combine(pos, neg):
    pos = list(map(lambda s: (s, 1), pos))
    neg = list(map(lambda s: (s, 0), neg))
    data = pos + neg
    np.random.shuffle(data)
    return data


def _preprocess_split_polarity(tgz_dir, proc_dir, split, polarity, verbose):
    if verbose:
        print('Preprocessing %s %s files...' % (polarity, split))
    texts = []
    pattern = os.path.join(tgz_dir, split, polarity, '*')
    each_filename = glob(pattern)
    if verbose <= 2:
        each_filename = tqdm(each_filename, leave=False)
    for filename in each_filename:
        text = open(filename).read()
        texts.append(text)
    if not os.path.exists(proc_dir):
        os.mkdir(proc_dir)
    count_basename = '%s_%s_count.txt' % (split, polarity)
    count_filename = os.path.join(proc_dir, count_basename)
    with open(count_filename, 'wb') as count_file:
        text = str(len(texts)).encode('utf-8')
        count_file.write(text)
    data_basename = '%s_%s_data.txt' % (split, polarity)
    data_filename = os.path.join(proc_dir, data_basename)
    with open(data_filename, 'wb') as data_file:
        for text in texts:
            line = (text + '\n').encode('utf-8')
            data_file.write(line)


def _preprocess(tgz_dir, proc_dir, verbose):
    for split in ['train', 'test']:
        for polarity in ['pos', 'neg']:
            _preprocess_split_polarity(
                tgz_dir, proc_dir, split, polarity, verbose)
            _preprocess_split_polarity(
                tgz_dir, proc_dir, split, polarity, verbose)


def _load_split_polarity(proc_dir, split, polarity, verbose):
    count_basename = '%s_%s_count.txt' % (split, polarity)
    count_filename = os.path.join(proc_dir, count_basename)
    count = int(open(count_filename).read())
    data_basename = '%s_%s_data.txt' % (split, polarity)
    data_filename = os.path.join(proc_dir, data_basename)
    texts = []
    each_line = open(data_filename)
    if verbose <= 2:
        each_line = tqdm(each_line, total=count, leave=False)
    for line in each_line:
        texts.append(line.strip())
    return texts


def _load_split(proc_dir, split, verbose, y_dtype):
    pos = _load_split_polarity(proc_dir, split, 'pos', verbose)
    neg = _load_split_polarity(proc_dir, split, 'neg', verbose)
    data = _combine(pos, neg)
    texts, labels = list(zip(*data))
    return texts, np.array(labels, dtype=y_dtype)


def _load(proc_dir, verbose, y_dtype):
    train = _load_split(proc_dir, 'train', verbose, y_dtype)
    val = _load_split(proc_dir, 'test', verbose, y_dtype)
    return train, val


@kwargs_only
def load_imdb_raw(dataset_name=_DATASET_NAME, proc_subdir=_PROC_SUBDIR,
                  tgz_subdir=_TGZ_SUBDIR, url=_URL, verbose=2,
                  y_dtype='float32'):
    dataset_dir = get_dataset_dir(dataset_name)
    proc_dir = os.path.join(dataset_dir, proc_subdir)
    if not os.path.exists(proc_dir):
        tgz_dir = os.path.join(dataset_dir, tgz_subdir)
        if not os.path.exists(tgz_dir):
            tgz_basename = os.path.basename(_URL)
            local = os.path.join(dataset_dir, tgz_basename)
            if not os.path.exists(local):
                download(_URL, local, verbose)
            _extract(dataset_dir, tgz_basename, verbose)
        _preprocess(tgz_dir, proc_dir, verbose)
    return _load(proc_dir, verbose, y_dtype)


@kwargs_only
def load_imdb(dataset_name=_DATASET_NAME, proc_subdir=_PROC_SUBDIR,
              tgz_subdir=_TGZ_SUBDIR, url=_URL, verbose=2, x_tf=None,
              y_dtype='float32'):
    (x_train, y_train), (x_val, y_val) = load_imdb_raw(
        dataset_name=dataset_name, proc_subdir=proc_subdir,
        tgz_subdir=tgz_subdir, url=url, verbose=verbose, y_dtype=y_dtype)
    if x_tf is None:
        x_tf = Pipe(Tokenize(), Length(512), Dict(), Numpy('int64'))
    x_train = x_tf.fit_transform(x_train)
    x_val = x_tf.transform(x_val)
    return ((x_train, y_train), (x_val, y_val)), x_tf
