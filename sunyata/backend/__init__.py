import os
import sys


_BACKEND = os.environ.get('SUNYATA_BACKEND', 'pytorch')
if _BACKEND == 'chainer':
    from .chainer import ChainerBackend as Backend
elif _BACKEND == 'mxnet':
    from .mxnet import MXNetBackend as Backend
elif _BACKEND == 'pytorch':
    from .pytorch import PyTorchBackend as Backend
elif _BACKEND == 'tensorflow':
    from .tensorflow import TensorFlowBackend as Backend
else:
    assert False

_BACKEND = Backend()

module = sys.modules[__name__]
for method_name in filter(lambda s: not s.startswith('_'), dir(_BACKEND)):
    setattr(module, method_name, getattr(_BACKEND, method_name))
