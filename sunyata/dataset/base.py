import os
from time import time
from tqdm import tqdm
from urllib.request import urlretrieve


def _download_urlretrieve(url, filename, verbose):
    """
    Download using urlretrieve.

    in:
        url       str  Remote URL.
        filename  str  Local file path.
        verbose   int  Verbosity (one of 0, 1, 2).
    """
    class bridge(object):
        progress_bar = None

    def report_hook(count, chunk_size, total_size):
        if verbose < 2:
            return
        if bridge.progress_bar is None:
            if total_size == -1:
                total_size = None
            bridge.progress_bar = \
                tqdm(total=total_size, unit='B', unit_scale=True)
        else:
            bridge.progress_bar.update(chunk_size)

    try:
        urlretrieve(url, filename, report_hook)
    except:
        if os.path.exists(filename):
            os.remove(filename)
        raise


def download(url, filename, verbose=2):
    assert verbose in {0, 1, 2}
    if verbose:
        print('Downloading: %s' % url)
        print('         to: %s' % filename)
    if verbose == 1:
        t0 = time()
    dir_name = os.path.dirname(filename)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    _download_urlretrieve(url, filename, verbose)
    if verbose == 1:
        t = time() - t0
        print('...took %.3f sec.' % t)


def get_sunyata_home():
    dir_name = os.environ.get('SUNYATA_HOME')
    if dir_name:
        return dir_name
    return os.path.expanduser('~/.sunyata/')


def get_dataset_dir(dataset):
    return os.path.join(get_sunyata_home(), 'dataset', dataset)
