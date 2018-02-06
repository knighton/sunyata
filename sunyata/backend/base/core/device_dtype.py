from contextlib import contextmanager

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
        return 0 < self.id

    def gpu_id(self):
        assert 0 < self.id
        return self.id - 1


class BaseDeviceAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def num_devices(self):
        return len(self._devices)

    def num_gpus(self):
        return len(self._devices) - 1

    def set_devices(self, num_gpus, unscoped_device_id):
        self._devices = []
        for device_id in range(num_gpus + 1):
            device = Device(device_id)
            self._devices.append(device)
        self._device_scopes = [None]
        self.set_unscoped_device(unscoped_device_id)

    def devices(self):
        return self._devices

    def set_unscoped_device(self, device):
        self._device_scopes[0] = self.device(device)

    def unscoped_device(self):
        return self._device_scopes[0]

    @contextmanager
    def device_scope(self, device):
        self._device_scopes.append(self.device(device))
        yield
        self._device_scopes.pop()

    def current_device(self):
        return self._device_scopes[-1]

    def device(self, x):
        if x is None:
            device = self.current_device()
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

    def cast_to_device(self, x, dtype, device=None, copy=False):
        raise NotImplementedError

    def cast_to_cpu(self, x, dtype, copy=False):
        device = self._devices[0]
        return self.cast_to_device(x, dtype, device, copy)

    def cast_to_gpu(self, x, dtype, device=None, copy=False):
        device = self.device(device)
        assert device.is_gpu()
        return self.cast_to_device(x, dtype, device, copy)

    def cast(self, x, dtype, copy=False):
        return self.cast_to_device(x, dtype, None, copy)

    def to_device(self, x, device=None, copy=False):
        return self.cast_to_device(x, None, device, copy)

    def to_cpu(self, x, copy=False):
        return self.to_device(x, 0, copy)

    def to_gpu(self, x, device=None, copy=False):
        device = self.device(device)
        assert device.is_gpu()
        return self.to_device(x, device, copy)

    def cast_numpy_to_device(self, x, dtype, device=None):
        raise NotImplementedError

    def cast_numpy_to_cpu(self, x, dtype):
        return self.cast_numpy_to_device(x, dtype, self._devices[0])

    def cast_numpy_to_gpu(self, x, dtype, device):
        device = self.device(device)
        assert device.is_gpu()
        return self.cast_numpy_to_device(x, dtype, device)

    def numpy_to_device(self, x, device=None):
        return self.cast_numpy_to_device(x, None, device)

    def numpy_to_cpu(self, x):
        return self.numpy_to_device(x, self._devices[0])

    def numpy_to_gpu(self, x, device):
        device = self.device(device)
        assert device.is_gpu()
        return self.numpy_to_device(x, device)
