import mxnet as mx
import subprocess

from ...base.core.device_dtype import \
    BaseDataTypeAPI, BaseDeviceAPI, BaseDeviceDataTypeAPI


class MXNetDeviceAPI(BaseDeviceAPI):
    def __init__(self):
        BaseDeviceAPI.__init__(self)


class MXNetDataTypeAPI(BaseDataTypeAPI):
    def __init__(self):
        BaseDataTypeAPI.__init__(self)


class MXNetDeviceDataTypeAPI(BaseDeviceDataTypeAPI, MXNetDataTypeAPI,
                             MXNetDeviceAPI):
    def __init__(self):
        BaseDeviceDataTypeAPI.__init__(self)
        MXNetDataTypeAPI.__init__(self)
        MXNetDeviceAPI.__init__(self)
        num_gpus = self.discover_gpus()
        default_device_id = 1 if num_gpus else 0
        self.set_devices(num_gpus, default_device_id)
        supported_dtypes = sorted("""
            bool
            uint8 uint16 uint32 uint64
            int8 int16 int32 int64
            float16 float32 float64
        """.split())
        default_dtype = 'float32'
        self.set_supported_dtypes(supported_dtypes, default_dtype)
        for i, device in enumerate(self._devices):
            if device.is_cpu():
                ctx = mx.cpu()
            elif device.is_gpu():
                ctx = mx.gpu(device.gpu_id())
            else:
                assert False
            device.mx_context = ctx

    def discover_gpus(self):
        cmd = 'nvidia-smi', '-L'
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE)
            lines = result.stdout.decode('unicode-escape')
            return len(lines)
        except:
            return 0

    def device_of(self, x):
        if x.context.device_type == 'cpu':
            device_id = 0
        elif x.context.device_type == 'gpu':
            device_id = x.context.device_id + 1
        else:
            assert False
        return self._devices[device_id]

    def dtype_of(self, x):
        return x.dtype.__name__

    def cast_to_device(self, x, dtype=None, device=None, copy=True):
        from_device = self.device_of(x)
        to_device = from_device if device is None else self.device(device)
        if from_device is not to_device:
            x = x.as_in_context(to_device.mx_context)
            copy = False
        from_dtype = self.dtype_of(x)
        to_dtype = from_dtype if dtype is None else self.dtype(dtype)
        if from_dtype != to_dtype:
            x = x.astype(to_dtype)
            copy = False
        if copy:
            x = x.copy()
        return x

    def cast_numpy_to_device(self, x, dtype=None, device=None):
        to_device = self.device(device)
        from_dtype = x.dtype.name
        to_dtype = from_dtype if dtype is None else self.dtype(dtype)
        return mx.nd.array(x, to_device.mx_context, to_dtype)
