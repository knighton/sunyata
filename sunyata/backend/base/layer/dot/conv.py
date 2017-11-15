from ...base import APIMixin


class BaseConvAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def conv(self, x, kernel, bias, stride, pad, dilation):
        raise NotImplementedError

    def conv1d(self, x, kernel, bias, stride, pad, dilation):
        raise NotImplementedError

    def conv2d(self, x, kernel, bias, stride, pad, dilation):
        raise NotImplementedError

    def conv3d(self, x, kernel, bias, stride, pad, dilation):
        raise NotImplementedError
