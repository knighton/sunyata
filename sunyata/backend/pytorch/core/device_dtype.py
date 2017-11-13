import importlib
import torch
from torch.autograd import Variable

from ...base.core.device_dtype import \
    BaseDataTypeAPI, BaseDeviceAPI, BaseDeviceDataTypeAPI


class PyTorchDeviceAPI(BaseDeviceAPI):
    def __init__(self):
        BaseDeviceAPI.__init__(self)


class PyTorchDataTypeAPI(BaseDataTypeAPI):
    def __init__(self):
        BaseDataTypeAPI.__init__(self)


class PyTorchDeviceDataTypeAPI(BaseDeviceDataTypeAPI, PyTorchDeviceAPI,
                               PyTorchDataTypeAPI):
    def __init__(self):
        num_gpus = self.discover_gpus()
        default_device_id = 1 if num_gpus else 0
        self.set_devices(num_gpus, default_device_id)
        data = """
            uint8    torch.ByteTensor    torch.cuda.ByteTensor
            int8     torch.CharTensor    torch.cuda.CharTensor
            int16    torch.ShortTensor   torch.cuda.ShortTensor
            int32    torch.IntTensor     torch.cuda.IntTensor
            int64    torch.LongTensor    torch.cuda.LongTensor
            float16  torch.HalfTensor    torch.cuda.HalfTensor
            float32  torch.FloatTensor   torch.cuda.FloatTensor
            float64  torch.DoubleTensor  torch.cuda.DoubleTensor
        """
        self._tensor2dtype = {}
        self._dtype2cpu = {}
        self._dtype2gpu = {}
        for line in data.strip().split('\n'):
            dtype, cpu, gpu = line.split()
            self._tensor2dtype[cpu] = dtype
            self._dtype2cpu[dtype] = cpu
            self._tensor2dtype[gpu] = dtype
            self._dtype2gpu[dtype] = gpu
        supported_dtypes = sorted(self._dtype2cpu)
        default_dtype = 'float32'
        self.set_supported_dtypes(supported_dtypes, default_dtype)

    def discover_gpus(self):
        return torch.cuda.device_count()

    def device_of(self, x):
        if x.is_cuda:
            device_id = x.get_device()
        else:
            device_id = 0
        return self._devices[device_id]

    def dtype_of(self, x):
        if isinstance(x, torch._TensorBase):
            tensor = x
        elif isinstance(x, Variable):
            tensor = x.data
        else:
            assert False
        return self._tensor2dtype[tensor.type()]

    def _get_tensor_class(self, dtype, device):
        if device.is_cpu():
            dtype2class = self._dtype2cpu
        elif device.is_gpu():
            dtype2class = self._dtype2gpu
        else:
            assert False
        path = dtype2class[dtype]
        x = path.rindex('.')
        module_name = path[:x]
        class_name = path[x + 1:]
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def cast_to_device(self, x, dtype=None, device=None, copy=True):
        from_dtype = self.dtype_of(x)
        to_dtype = from_dtype if dtype is None else self.dtype(dtype)
        from_device = self.device_of(x)
        to_device = from_device if device is None else self.device(device)
        to_tensor_class = self._get_tensor_class(to_dtype, to_device)
        if from_device is to_device:
            if from_dtype != to_dtype or copy:
                x = x.type(to_tensor_class)
        else:
            if to_device.is_cpu():
                x = to_tensor_class(x)
            elif to_device.is_gpu():
                with torch.cuda.device(to_device.gpu_id()):
                    x = x.type(to_tensor_class)
            else:
                assert False
        return x

    def cast_numpy_to_device(self, x, dtype=None, device=None):
        from_dtype = x.dtype.name
        to_dtype = from_dtype if dtype is None else self.dtype(dtype)
        to_device = self.device(device)
        to_tensor_class = self._get_tensor_class(to_dtype, to_device)
        if to_device.is_cpu():
            x = to_tensor_class(x)
        elif to_device.is_gpu():
            with torch.cuda.device(to_device.gpu_id()):
                x = to_tensor_class(x)
        else:
            assert False
        return x
