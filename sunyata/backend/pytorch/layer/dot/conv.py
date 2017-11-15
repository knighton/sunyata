from torch.nn import functional as F

from ....base.layer.dot.conv import BaseConvAPI


class PyTorchConvAPI(BaseConvAPI):
    def __init__(self):
        BaseConvAPI.__init__(self)
        self.ndim2conv = {
            1: self.conv1d,
            2: self.conv2d,
            3: self.conv3d,
        }

    def conv(self, x, kernel, bias, stride, pad, dilation):
        ndim = x.dim() - 2
        return self.ndim2conv[ndim](x, kernel, bias, stride, pad, dilation)

    def conv1d(self, x, kernel, bias, stride, pad, dilation):
        stride = self.to_one(stride)
        pad = self.to_one(pad)
        dilation = self.to_one(dilation)
        return F.conv1d(x, kernel, bias, stride, pad, dilation)

    def conv2d(self, x, kernel, bias, stride, pad, dilation):
        return F.conv2d(x, kernel, bias, stride, pad, dilation)

    def conv3d(self, x, kernel, bias, stride, pad, dilation):
        return F.conv3d(x, kernel, bias, stride, pad, dilation)
