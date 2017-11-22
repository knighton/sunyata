from .constant import BaseConstantPadAPI
from .edge import BaseEdgePadAPI
from .reflect import BaseReflectPadAPI


class BasePadAPI(BaseConstantPadAPI, BaseEdgePadAPI, BaseReflectPadAPI):
    def __init__(self):
        BaseConstantPadAPI.__init__(self)
        BaseEdgePadAPI.__init__(self)
        BaseReflectPadAPI.__init__(self)

    def pad_out_shape(self, x_shape, pad):
        pad = self.unpack_int_pad(pad, len(x_shape))
        y_shape = [x_shape[0]]
        for dim, (pad_left, pad_right) in zip(x_shape[1:], pad):
            y_shape.append(pad_left + dim + pad_right)
        return tuple(y_shape)
