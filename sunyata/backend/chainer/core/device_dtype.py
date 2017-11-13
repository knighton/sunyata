from chainer import functions as F
from chainer import Variable

from ...base.core.device_dtype import \
    BaseDataTypeAPI, BaseDeviceAPI, BaseDeviceDataTypeAPI


class ChainerDeviceAPI(BaseDeviceAPI):
    def __init__(self):
        BaseDeviceAPI.__init__(self)


class ChainerDataTypeAPI(BaseDataTypeAPI):
    def __init__(self):
        BaseDataTypeAPI.__init__(self)


class ChainerDeviceDataTypeAPI(BaseDeviceDataTypeAPI, ChainerDataTypeAPI,
                               ChainerDeviceAPI):
    def __init__(self):
        BaseDeviceDataTypeAPI.__init__(self)
        ChainerDataTypeAPI.__init__(self)
        ChainerDeviceAPI.__init__(self)
        num_gpus = self.discover_gpus()
        assert not num_gpus
        default_device_id = 0
        self.set_devices(num_gpus, default_device_id)
        supported_dtypes = sorted("""
            bool
            uint8 uint16 uint32 uint64
            int8 int16 int32 int64
            float16 float32 float64
        """.split())
        default_dtype = 'float32'
        self.set_supported_dtypes(supported_dtypes, default_dtype)

    def discover_gpus(self):
        return 0

    def device_of(self, x):
        return self._devices[0]

    def dtype_of(self, x):
        return x.dtype.name

    def _do_cast(self, x, from_dtype, to_dtype):
        if self.is_float_dtype(from_dtype):
            x = F.cast(x, to_dtype)
        else:
            x = Variable(x.data.astype(to_dtype))
        return x

    def cast_to(self, x, dtype=None, device=None, copy=True):
        from_device = self.device_of(x)
        assert from_device is self._devices[0]
        to_device = from_device if device is None else self.device(device)
        assert to_device is self._devices[0]
        from_dtype = self.dtype_of(x)
        to_dtype = from_dtype if dtype is None else self.dtype(dtype)
        if from_dtype != to_dtype or copy:
            x = self._do_cast(x, from_dtype, to_dtype)
        return x

    def cast_numpy_to(self, x, dtype=None, device=None):
        from_dtype = self.dtype_of(x)
        to_dtype = from_dtype if dtype is None else self.dtype(dtype)
        if from_dtype != to_dtype:
            x = x.astype(to_dtype)
        else:
            x = x.copy()
        return x
