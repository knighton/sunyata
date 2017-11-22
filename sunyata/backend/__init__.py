import os
import sys


_BACKEND = os.environ.get('SUNYATA_BACKEND', 'pytorch')
print('Backend: %s.' % _BACKEND)

if _BACKEND in {'ch', 'chainer'}:
    from .chainer import ChainerBackend as Backend
elif _BACKEND in {'mx', 'mxnet'}:
    from .mxnet import MXNetBackend as Backend
elif _BACKEND in {'pt', 'pytorch'}:
    from .pytorch import PyTorchBackend as Backend
elif _BACKEND in {'tf', 'tensorflow'}:
    from .tensorflow import TensorFlowBackend as Backend
else:
    assert False

_BACKEND = Backend()

module = sys.modules[__name__]
for method_name in filter(lambda s: not s.startswith('_'), dir(_BACKEND)):
    setattr(module, method_name, getattr(_BACKEND, method_name))
