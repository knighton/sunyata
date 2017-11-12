from ..base import APIMixin


class Device(object):
    def __init__(self, id):
        assert isinstance(id, int)
        assert 0 <= id
        self.id = id

    @property
    def type(self):
        return 'gpu' if self.id else 'cpu'

    def is_cpu(self):
        return not self.id

    def is_gpu(self):
        return bool(self.id)


class BaseDeviceAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def num_devices(self):
        return len(self._devices)

    def num_gpus(self):
        return len(self._devices) - 1

    def set_devices(self, num_gpus, default_device_id):
        self._devices = []
        for device_id in range(num_gpus + 1):
            device = Device(device_id)
            self._devices.append(device)
        self.set_default_device(default_device_id)

    def devices(self):
        return self._devices

    def set_default_device(self, device):
        if isinstance(device, Device):
            assert device in self._devices
        else:
            assert isinstance(device, int)
            assert 0 <= device < len(self._devices)
            device = self._devices[device]
        self._default_device = device

    def default_device(self):
        return self._default_device

    def device(self, x):
        if x is None:
            return self.default_device()
        elif isinstance(x, Device):
            device = x
        else:
            assert isinstance(x, int)
            assert 0 <= x < len(self._devices)
            device = self._devices[x]
        return device


class BaseDataTypeAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def set_supported_dtypes(self, supported_dtypes, default_dtype):
        assert supported_dtypes
        assert sorted(supported_dtypes) == supported_dtypes
        for dtype in supported_dtypes:
            assert dtype
            assert isinstance(dtype, str)
        self._supported_dtypes = supported_dtypes
        self.set_default_dtype(default_dtype)

    def supported_dtypes(self):
        return self._supported_dtypes

    def set_default_dtype(self, dtype):
        assert dtype in self._supported_dtypes
        self._default_dtype = dtype

    def default_dtype(self):
        return self._default_dtype

    def dtype(self, dtype):
        if dtype is None:
            dtype = self.default_dtype()
        else:
            assert dtype in self._supported_dtypes
        return dtype

    def is_float_dtype(self, dtype):
        return dtype.startswith('float')

    def is_sint_dtype(self, dtype):
        return dtype.startswith('int')

    def is_uint_dtype(self, dtype):
        return dtype.startswith('uint')

    def is_xint_dtype(self, dtype):
        return dtype.startswith('int') or dtype.startswith('uint')


class BaseDeviceDataTypeAPI(BaseDeviceAPI, BaseDataTypeAPI):
    def __init__(self):
        BaseDeviceAPI.__init__(self)
        BaseDataTypeAPI.__init__(self)

    def discover_gpus(self):
        raise NotImplementedError

    def device_of(self, x):
        raise NotImplementedError

    def dtype_of(self, x):
        raise NotImplementedError

    def cast_to(self, x, dtype=None, device=None, copy=False):
        raise NotImplementedError

    def cast_numpy_to(self, x, dtype=None, device=None):
        raise NotImplementedError

    def cast(self, x, dtype=None, copy=False):
        return self.cast_to(x, dtype, None, copy)

    def _cast_bool_output(self, input_arg, x, override_dtype):
        if override_dtype is None:
            to_dtype = self.dtype_of(input_arg)
        else:
            to_dtype = self.dtype(override_dtype)
        if self.dtype_of(x) != to_dtype:
            x = self.cast(x, to_dtype)
        return x

    def to(self, x, device=None, copy=False):
        return self.cast_to(x, None, device, copy)

    def to_cpu(self, x, copy=False):
        return self.to_device(x, 0, copy)

    def to_gpu(self, x, device, copy=False):
        device = self.to_device(device)
        assert device.is_gpu()
        return self.to_device(x, device, copy)
