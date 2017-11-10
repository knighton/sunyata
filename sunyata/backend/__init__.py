import os


backend = os.environ.get('SUNYATA_BACKEND', 'pytorch')
if backend == 'chainer':
    from .chainer import ChainerBackend as Backend
elif backend == 'mxnet':
    from .mxnet import MXNetBackend as Backend
elif backend == 'pytorch':
    from .pytorch import PyTorchBackend as Backend
elif backend == 'tensorflow':
    from .tensorflow import TensorFlowBackend as Backend
else:
    assert False
