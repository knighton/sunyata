import numpy as np
py_max = max

from .avg import BaseAvgPoolAPI
from .max import BaseMaxPoolAPI


class BasePoolAPI(BaseAvgPoolAPI, BaseMaxPoolAPI):
    def __init__(self):
        BaseAvgPoolAPI.__init__(self)
        BaseMaxPoolAPI.__init__(self)

    def pool_out_shape(self, in_shape, face, stride, pad):
        spatial_ndim = len(in_shape) - 1
        face = self.to_shape(face, spatial_ndim)
        stride = self.to_shape(stride, spatial_ndim)
        pad = self.to_shape(pad, spatial_ndim)
        out_shape = [in_shape[0]]
        for i in range(len(in_shape) - 1):
            dim = (in_shape[i + 1] + 2 * pad[i] - face[i]) / stride[i]
            dim = int(np.floor(py_max(dim, 0))) + 1
            out_shape.append(dim)
        return tuple(out_shape)
