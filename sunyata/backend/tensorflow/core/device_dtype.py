import tensorflow as tf
import tensorflow.contrib.eager as tfe

from ...base.core.device_dtype import \
    BaseDataTypeAPI, BaseDeviceAPI, BaseDeviceDataTypeAPI


class TensorFlowDeviceAPI(BaseDeviceAPI):
    def __init__(self):
        BaseDeviceAPI.__init__(self)


class TensorFlowDataTypeAPI(BaseDataTypeAPI):
    def __init__(self):
        BaseDataTypeAPI.__init__(self)


class TensorFlowDeviceDataTypeAPI(BaseDeviceDataTypeAPI, TensorFlowDataTypeAPI,
                                  TensorFlowDeviceAPI,
    def __init__(self):
        BaseDeviceDataTypeAPI.__init__(self)
        TensorFlowDataTypeAPI.__init__(self)
        TensorFlowDeviceAPI.__init__(self)
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

    def discover_gpus(self):
        return tfe.num_gpus()

    def device_of(self, x):
        if x.device == 'CPU:0':
            device_id = 0
        else:
            x = x.device.rindex('/')
            s = x.device[x + 1:]
            ss = s.split(':')
            assert ss[0] == 'device'
            assert ss[1] == 'GPU'
            device_id = int(ss[2]) + 1
        return self._devices[device_id]

    def dtype_of(self, x):
        return x.dtype.name

    def _device_name(self, device):
        if device.is_cpu():
            name = 'cpu:0'
        elif device.is_gpu():
            name = 'gpu:%d' % device.gpu_id()
        else:
            assert False
        return name

    def cast_to(self, x, dtype=None, device=None, copy=True):
        from_device = self.device_of(x)
        to_device = from_device if device is None else self.device(device)
        from_dtype = self.dtype_of(x)
        to_dtype = from_dtype if dtype is None else self.dtype(dtype)
        if from_device is to_device:
            if from_dtype != to_dtype or copy:
                x = tf.cast(x, to_dtype)
        else:
            with tf.device(self._device_name(to_device)):
                x = tf.convert_to_tensor(x, to_dtype)
        return x

    def cast_numpy_to(self, x, dtype=None, device=None):
        to_device = self.device(device)
        to_dtype = x.dtype.name if dtype is None else self.dtype(dtype)
        with tf.device(self._device_name(to_device)):
            return tf.convert_to_tensor(x, to_dtype)
