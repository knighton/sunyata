import importlib
import numpy as np
import torch
from torch.autograd import Variable as PTVariable
import torch.nn.functional as PTF

from ..base import \
    Device, BaseActivationAPI, BaseDataTypeAPI, BaseDeviceAPI, \
    BaseDeviceDataTypeAPI, BaseEpsilonAPI, BaseLogicAPI, BaseMapAPI, \
    BaseMetricAPI, BaseReduceAPI, BaseRelateAPI, BaseShapeAPI, \
    BaseVariableAPI, BaseBackend 


class PyTorchActivationAPI(BaseActivationAPI):
    def softmax(self, x):
        x_shape = x.size()
        tran = x.transpose(1, len(x_shape) - 1)
        tran_shape = tran.size()
        tran_2d = tran.contiguous().view(-1, tran_shape[-1])
        tran_2d = PTF.softmax(tran_2d)
        tran = tran_2d.view(*tran_shape)
        return tran.transpose(1, len(x_shape) - 1)


class PyTorchLogicAPI(BaseLogicAPI):
    def equal(self, a, b):
        return self.cast(a == b, self.dtype_of(a))


class PyTorchMapAPI(BaseMapAPI):
    def clip(self, x, min=-np.inf, max=np.inf):
        return x.clamp(min, max)

    def log(self, x):
        return x.log()

    def pow(self, x, a):
        return x ** a


class PyTorchMetricAPI(BaseMetricAPI):
    pass


class PyTorchReduceAPI(BaseReduceAPI):
    def argmax(self, x, axis=-1):
        return x.max(axis)[1]

    def _normalize_axis(self, axis, keepdims, ndim):
        if axis is None:
            if keepdims:
                axes = list(range(ndim))
            else:
                return None
        elif isinstance(axis, int):
            axes = [axis]
        elif isinstance(axis, tuple):
            axes = list(axis)
        elif isinstance(axis, list):
            pass
        else:
            assert False
        axes = list(map(lambda n: n % ndim, axes))
        return sorted(axes)

    def _reduce(self, name, x, axis=None, keepdims=False):
        axes = self._normalize_axis(axis, keepdims, x.dim())
        if axes is None:
            return getattr(x, name)(None, True)
        for axis in reversed(axes):
            if x.dim() == 1:
                keepdims = True
            x = getattr(x, name)(axis, keepdims)
            if isinstance(x, tuple):
                x = x[0]
        return x

    def mean(self, x, axis=None, keepdims=False):
        return self._reduce('mean', x, axis, keepdims)

    def sum(self, x, axis=None, keepdims=False):
        return self._reduce('sum', x, axis, keepdims)


class PyTorchRelateAPI(BaseRelateAPI):
    def dense(self, x, kernel, bias):
        return x.mm(kernel) + bias


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


class PyTorchDeviceAPI(BaseDeviceAPI):
    pass


class PyTorchDataTypeAPI(BaseDataTypeAPI):
    pass


class PyTorchDeviceDataTypeAPI(BaseDeviceDataTypeAPI):
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
        elif isinstance(x, PTVariable):
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

    def cast_to(self, x, dtype=None, device=None, copy=True):
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

    def cast_numpy_to(self, x, dtype=None, device=None):
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


class PyTorchVariableAPI(BaseVariableAPI):
    def constant(self, x):
        return PTVariable(x.clone(), requires_grad=False)

    def variable(self, x):
        return PTVariable(x.clone(), requires_grad=True)

    def gradients(self, params, forward, judges, aux_judges, xx, yy_true):
        yy_pred = forward(xx)
        score_vars = []
        score_grads = []
        for judge, y_true, y_pred in zip(judges, yy_true, yy_pred):
            score_vars.append(self.mean(judge(y_true, y_pred)))
            arr = np.ones((1,), self.dtype_of(y_true)) * judge.importance
            score_grads.append(self.cast_numpy_to(arr))
        torch.autograd.backward(score_vars, score_grads)
        scores = list(map(lambda x: x.data, score_vars))
        grads_and_params = list(map(lambda x: (x.grad.data, x), params))
        aux_scores = self._aux_scores(aux_judges, yy_true, yy_pred)
        return grads_and_params, scores, aux_scores

    def variable_to_tensor(self, x):
        return x.data

    def result_to_tensor(self, x):
        return x.data

    def assign(self, x, new_value):
        x.data = new_value
        x.grad.data.zero_()

    def numpy(self, x):
        if isinstance(x, torch._TensorBase):
            pass
        elif isinstance(x, PTVariable):
            x = x.data
        else:
            assert False
        return x.cpu().numpy()


class PyTorchBackend(BaseBackend, PyTorchActivationAPI, PyTorchDataTypeAPI,
                     PyTorchDeviceAPI, PyTorchDeviceDataTypeAPI,
                     PyTorchLogicAPI, PyTorchMapAPI, PyTorchMetricAPI,
                     PyTorchReduceAPI, PyTorchRelateAPI, PyTorchShapeAPI,
                     PyTorchVariableAPI):
    def __init__(self):
        BaseBackend.__init__(self)
        PyTorchActivationAPI.__init__(self)
        PyTorchDataTypeAPI.__init__(self)
        PyTorchDeviceAPI.__init__(self)
        PyTorchDeviceDataTypeAPI.__init__(self)
        PyTorchLogicAPI.__init__(self)
        PyTorchMapAPI.__init__(self)
        PyTorchMetricAPI.__init__(self)
        PyTorchReduceAPI.__init__(self)
        PyTorchRelateAPI.__init__(self)
        PyTorchShapeAPI.__init__(self)
        PyTorchVariableAPI.__init__(self)
        self.name = 'pytorch'
