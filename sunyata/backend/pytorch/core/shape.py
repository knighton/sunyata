import torch

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

    def squeeze(self, x, axis=None):
        return x.squeeze(axis)

    def tile(self, x, reps):
        return x.repeat(*reps)

    def transpose(self, x, axes):
        return x.permute(*axes)

    def concat(self, xx, axis):
        return torch.cat(xx, axis)

    def stack(self, xx, axis=0):
        return torch.stack(xx, axis)
