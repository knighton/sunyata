from .....base.layer.shape.upsample import BaseUpsampleAPI
from .linear import PyTorchLinearUpsampleAPI
from .nearest import PyTorchNearestUpsampleAPI


class PyTorchUpsampleAPI(BaseUpsampleAPI, PyTorchLinearUpsampleAPI,
                         PyTorchNearestUpsampleAPI):
    def __init__(self):
        PyTorchLinearUpsampleAPI.__init__(self)
        PyTorchNearestUpsampleAPI.__init__(self)
