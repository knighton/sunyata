from ...base.core.shape import BaseShapeAPI


class PyTorchShapeAPI(BaseShapeAPI):
    def ndim(self, x):
        return x.dim()

    def shape(self, x):
        return tuple(x.size())

    def size(self, x):
        return x.nelement()

    def reshape(self, x, shape):
        return x.view(shape)

    def expand_dims(self, x, axis):
        return x.unsqueeze(axis)
