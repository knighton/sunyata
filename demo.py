import chainer
import chainer.functions as CHF
from contextlib import contextmanager
import importlib
import keras
import mxnet as mx
import numpy as np
import os
import subprocess
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import torch
from torch.autograd import Variable as PTVariable
import torch.nn.functional as PTF


tfe.enable_eager_execution()


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


class APIBase(object):
    pass


class BaseActivationAPI(APIBase):
    def softmax(self, x):
        raise NotImplementedError


class PyTorchActivationAPI(BaseActivationAPI):
    def softmax(self, x):
        x_shape = x.size()
        tran = x.transpose(1, len(x_shape) - 1)
        tran_shape = tran.size()
        tran_2d = tran.contiguous().view(-1, tran_shape[-1])
        tran_2d = PTF.softmax(tran_2d)
        tran = tran_2d.view(*tran_shape)
        return tran.transpose(1, len(x_shape) - 1)


class TensorFlowActivationAPI(BaseActivationAPI):
    def softmax(self, x):
        return tf.nn.softmax(x)


class MXNetActivationAPI(BaseActivationAPI):
    def softmax(self, x):
        return mx.nd.softmax(x)


class ChainerActivationAPI(BaseActivationAPI):
    def softmax(self, x):
        return CHF.softmax(x)


class BaseEpsilonAPI(APIBase):
    def __init__(self):
        self.set_epsilon(1e-5)

    def set_epsilon(self, epsilon):
        assert isinstance(epsilon, float)
        assert 0 < epsilon < 1e-2
        self._epsilon = epsilon

    def epsilon(self):
        return self._epsilon


class BaseLogicAPI(APIBase):
    def equal(self, a, b):
        raise NotImplementedError


class PyTorchLogicAPI(BaseLogicAPI):
    def equal(self, a, b):
        return self.cast(a == b, self.dtype_of(a))


class MXNetLogicAPI(BaseLogicAPI):
    def equal(self, a, b):
        return a == b


class TensorFlowLogicAPI(BaseLogicAPI):
    def equal(self, a, b):
        return tf.cast(tf.equal(a, b), a.dtype)


class ChainerLogicAPI(BaseLogicAPI):
    def equal(self, a, b):
        return chainer.Variable((a.data == b.data).astype(a.dtype))


class BaseMapAPI(APIBase):
    def clip(self, x, min=-np.inf, max=np.inf):
        raise NotImplementedError

    def log(self, x):
        raise NotImplementedError

    def pow(self, x, a):
        raise NotImplementedError


class PyTorchMapAPI(BaseMapAPI):
    def clip(self, x, min=-np.inf, max=np.inf):
        return x.clamp(min, max)

    def log(self, x):
        return x.log()

    def pow(self, x, a):
        return x ** a


class TensorFlowMapAPI(BaseMapAPI):
    def clip(self, x, min=-np.inf, max=np.inf):
        return tf.clip_by_value(x, min, max)

    def log(self, x):
        return tf.log(x)

    def pow(self, x, a):
        return tf.pow(x, a)


class MXNetMapAPI(BaseMapAPI):
    def clip(self, x, min=-np.inf, max=np.inf):
        return mx.nd.clip(x, min, max)

    def log(self, x):
        return mx.nd.log(x)

    def pow(self, x, a):
        return mx.nd.power(x, a)


class ChainerMapAPI(BaseMapAPI):
    def clip(self, x, min=-np.inf, max=np.inf):
        return CHF.clip(x, float(min), float(max))

    def log(self, x):
        return CHF.math.exponential.log(x)

    def pow(self, x, a):
        return CHF.math.basic_math.pow(x, a)


class BaseMetricAPI(APIBase):
    def binary_cross_entropy(self, true, pred):
        pred = self.clip(pred, self.epsilon(), 1 - self.epsilon())
        return -true * self.log(pred) - (1 - true) * self.log(1 - pred)

    def categorical_cross_entropy(self, true, pred):
        pred = self.clip(pred, self.epsilon(), 1 - self.epsilon())
        return self.mean(-true * self.log(pred), -1)

    def mean_squared_error(self, true, pred):
        return self.mean(self.pow(true - pred, 2), -1)

    def categorical_accuracy(self, true, pred):
        true_indices = self.argmax(true, -1)
        pred_indices = self.argmax(pred, -1)
        hits = self.equal(true_indices, pred_indices)
        hits = self.cast(hits, self.dtype_of(true))
        return self.mean(hits, -1, False)


class PyTorchMetricAPI(BaseMetricAPI):
    pass


class MXNetMetricAPI(BaseMetricAPI):
    pass


class TensorFlowMetricAPI(BaseMetricAPI):
    pass


class ChainerMetricAPI(BaseMetricAPI):
    pass


class BaseReduceAPI(APIBase):
    def argmax(self, axis=-1):
        raise NotImplementedError

    def mean(self, x, axis=None, keepdims=False):
        raise NotImplementedError

    def sum(self, x, axis=None, keepdims=False):
        raise NotImplementedError


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
        i
        return self._reduce('mean', x, axis, keepdims)

    def sum(self, x, axis=None, keepdims=False):
        return self._reduce('sum', x, axis, keepdims)


class MXNetReduceAPI(BaseReduceAPI):
    def argmax(self, x, axis=-1):
        return mx.nd.argmax(x, axis)

    def _reduce(self, name, x, axis=None, keepdims=False):
        axis = mx.base._Null if axis is None else axis
        func = getattr(mx.nd, name)
        return func(x, axis, keepdims)

    def mean(self, x, axis=None, keepdims=False):
        return self._reduce('mean', x, axis, keepdims)

    def sum(self, x, axis=None, keepdims=False):
        return self._reduce('sum', x, axis, keepdims)


class TensorFlowReduceAPI(BaseReduceAPI):
    def argmax(self, x, axis=-1):
        return tf.argmax(x, axis)

    def mean(self, x, axis=None, keepdims=False):
        return tf.reduce_mean(x, axis, keepdims)

    def sum(self, x, axis=None, keepdims=False):
        return tf.reduce_sum(x, axis, keepdims)


class ChainerReduceAPI(BaseReduceAPI):
    def argmax(self, x, axis=-1):
        return CHF.argmax(x, axis)

    def mean(self, x, axis=None, keepdims=False):
        if axis is None:
            denom = x.size
            x = CHF.math.sum.sum(x, axis, keepdims)
            if not keepdims:
                x = self.expand_dims(x, 0)
            x /= denom
        else:
            if isinstance(axis, int):
                axes = [axis]
            else:
                axes = axes
            axes = tuple(map(lambda axis: axis % x.ndim, axes))
            denom = 1
            for axis in axes:
                denom *= x.shape[axis]
            x = CHF.math.sum.sum(x, axes, keepdims) / denom
        return x

    def sum(self, x, axis=None, keepdims=False):
        axis = tuple(axis) if isinstance(axis, list) else axis
        x = CHF.math.sum.sum(x, axis, keepdims)
        if not x.ndim:
            x = self.expand_dims(x, 0)
        return x


class BaseRelateAPI(APIBase):
    def dense(self, x, kernel, bias):
        raise NotImplementedError


class PyTorchRelateAPI(BaseRelateAPI):
    def dense(self, x, kernel, bias):
        return x.mm(kernel) + bias


class MXNetRelateAPI(BaseRelateAPI):
    def dense(self, x, kernel, bias):
        return mx.nd.dot(x, kernel) + bias


class TensorFlowRelateAPI(BaseRelateAPI):
    def dense(self, x, kernel, bias):
        return tf.matmul(x, kernel) + bias


class ChainerRelateAPI(BaseRelateAPI):
    def dense(self, x, kernel, bias):
        return CHF.connection.linear.linear(x, kernel, bias)


class BaseShapeAPI(APIBase):
    def ndim(self, x):
        raise NotImplementedError

    def shape(self, x):
        raise NotImplementedError

    def size(self, x):
        raise NotImplementedError

    def reshape(self, x, shape):
        raise NotImplementedError

    def expand_dims(self, x, axis):
        raise NotImplementedError


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


class MXNetShapeAPI(BaseShapeAPI):
    def ndim(self, x):
        return len(x.ndim)

    def shape(self, x):
        return x.shape

    def size(self, x):
        return x.size

    def reshape(self, x, shape):
        return mx.nd.reshape(x, shape)

    def expand_dims(self, x, axis):
        return mx.nd.expand_dims(x, axis)


class TensorFlowShapeAPI(BaseShapeAPI):
    def ndim(self, x):
        return int(tf.rank(x).numpy())

    def shape(self, x):
        return tuple(map(int, x.shape))

    def size(self, x):
        return int(tf.size(x).numpy())

    def reshape(self, x, shape):
        return tf.reshape(x, shape)

    def expand_dims(self, x, axis):
        return tf.expand_dims(x, axis)


class ChainerShapeAPI(BaseShapeAPI):
    def ndim(self, x):
        return x.ndim

    def shape(self, x):
        return x.shape

    def size(self, x):
        return x.size

    def reshape(self, x, shape):
        return x.reshape(shape)

    def expand_dims(self, x, axis):
        return CHF.array.expand_dims.expand_dims(x, axis)


class BaseDeviceAPI(APIBase):
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


class BaseDataTypeAPI(APIBase):
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
        if x is None:
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


class BaseDeviceDataTypeAPI(APIBase):
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

    def to(self, x, device=None, copy=False):
        return self.cast_to(x, None, device, copy)

    def to_cpu(self, x, copy=False):
        return self.to_device(x, 0, copy)

    def to_gpu(self, x, device, copy=False):
        device = self.to_device(device)
        assert device.is_gpu()
        return self.to_device(x, device, copy)


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


class MXNetDeviceAPI(BaseDeviceAPI):
    pass


class MXNetDataTypeAPI(BaseDataTypeAPI):
    pass


class MXNetDeviceDataTypeAPI(BaseDeviceDataTypeAPI):
    def __init__(self):
        num_gpus = self.discover_gpus()
        default_device_id = 1 if num_gpus else 0
        self.set_devices(num_gpus, default_device_id)
        supported_dtypes = sorted("""
            uint8
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

    def cast_to(self, x, dtype=None, device=None, copy=True):
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

    def cast_numpy_to(self, x, dtype=None, device=None):
        to_device = self.device(device)
        from_dtype = x.dtype.name
        to_dtype = from_dtype if dtype is None else self.dtype(dtype)
        return mx.nd.array(x, to_device.mx_context, to_dtype)


class TensorFlowDeviceAPI(BaseDeviceAPI):
    pass


class TensorFlowDataTypeAPI(BaseDataTypeAPI):
    pass


class TensorFlowDeviceDataTypeAPI(BaseDeviceDataTypeAPI):
    def __init__(self):
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
        from_dtype = x.dtype.name if device is None else self.device(device)
        to_dtype = x.dtype if dtype is None else self.dtype(dtype)
        with tf.device(self._device_name(to_device)):
            return tf.convert_to_tensor(x, to_dtype)


class ChainerDeviceAPI(BaseDeviceAPI):
    pass


class ChainerDataTypeAPI(BaseDataTypeAPI):
    pass


class ChainerDeviceDataTypeAPI(BaseDeviceDataTypeAPI):
    def __init__(self):
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
            x = CHF.cast(x, to_dtype)
        else:
            x = chainer.Variable(x.data.astype(to_dtype))
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


class BaseVariableAPI(APIBase):
    def constant(self, x):
        raise NotImplementedError

    def _name(self, name=None):
        if name is None:
            name = str(1 << 30)
        else:
            assert isinstance(name, str)
            assert name
        return name

    def variable(self, x):
        raise NotImplementedError

    def _aux_scores(self, aux_judges, yy_true, yy_pred):
        if aux_judges is None:
            return None
        aux_scores = []
        for y_aux_judges, y_true, y_pred in zip(aux_judges, yy_true, yy_pred):
            y_aux_scores = []
            for judge in y_aux_judges:
                result = self.mean(judge(y_true, y_pred))
                y_aux_scores.append(Z.result_to_tensor(result))
            aux_scores.append(y_aux_scores)
        return aux_scores

    def gradients(self, params, forward, judges, aux_judges, xx, yy_true):
        raise NotImplementedError

    def variable_to_tensor(self, x):
        raise NotImplementedError

    def result_to_tensor(self, x):
        raise NotImplementedError

    def assign(self, x, new_value):
        raise NotImplementedError

    def numpy(self, x):
        raise NotImplementedError

    def list(self, x):
        return self.numpy(x).tolist()

    def scalar(self, x):
        assert self.size(x) == 1
        return self.numpy(x).flatten()[0]


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
            arr = np.ones((1,), Z.dtype_of(y_true)) * judge.importance
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


class MXNetVariableAPI(BaseVariableAPI):
    def constant(self, x):
        return x.copy()

    def variable(self, x):
        x = x.copy()
        x.attach_grad()
        return x

    def gradients(self, params, forward, judges, aux_judges, xx, yy_true):
        scores = []
        score_grads = []
        with mx.autograd.record():
            yy_pred = forward(xx)
            for judge, y_true, y_pred in zip(judges, yy_true, yy_pred):
                scores.append(self.mean(judge(y_true, y_pred)))
                arr = np.ones((1,), Z.dtype_of(y_true)) * judge.importance
                score_grads.append(self.cast_numpy_to(arr))
        mx.autograd.backward(scores, score_grads)
        grads_and_params = list(map(lambda x: (x.grad, x), params))
        aux_scores = self._aux_scores(aux_judges, yy_true, yy_pred)
        return grads_and_params, scores, aux_scores

    def variable_to_tensor(self, x):
        return x

    def result_to_tensor(self, x):
        return x

    def assign(self, x, new_value):
        x[:] = new_value
        x.grad[:] = 0

    def numpy(self, x):
        return x.asnumpy()


class TensorFlowVariableAPI(BaseVariableAPI):
    def constant(self, x):
        return tf.constant(x)

    def variable(self, x):
        return tfe.Variable(x, name=self._name())

    def _ivag_inner(self, forward, judges, aux_judges, xx, yy_true, bridge):
        yy_pred = forward(xx)
        scores = []
        for judge, y_true, y_pred in zip(judges, yy_true, yy_pred):
            scores.append(self.mean(judge(y_true, y_pred)) * judge.importance)
        bridge.append(self._aux_scores(aux_judges, yy_true, yy_pred))
        return scores

    def gradients(self, params, forward, judges, aux_judges, xx, yy_true):
        ivag = tfe.implicit_value_and_gradients(self._ivag_inner)
        bridge = []
        scores, grads_and_params = \
            ivag(forward, judges, aux_judges, xx, yy_true, bridge)
        aux_scores = bridge.pop()
        return grads_and_params, scores, aux_scores

    def variable_to_tensor(self, x):
        return x[:]

    def result_to_tensor(self, x):
        return x

    def assign(self, x, new_value):
        x.assign(new_value)

    def numpy(self, x):
        return x.numpy()


class ChainerVariableAPI(BaseVariableAPI):
    def constant(self, x):
        return x

    def variable(self, x):
        return chainer.Variable(x)

    def gradients(self, params, forward, judges, aux_judges, xx, yy_true):
        yy_pred = forward(xx)
        score_vars = []
        score_grads = []
        for judge, y_true, y_pred in zip(judges, yy_true, yy_pred):
            score_vars.append(self.mean(judge(y_true, y_pred)))
            arr = np.ones(Z.shape(y_true), Z.dtype_of(y_true)) * \
                judge.importance
            score_grads.append(self.cast_numpy_to(arr))
        broadcasted_score_vars = []
        for score_var, score_grad in zip(score_vars, score_grads):
            x = CHF.array.broadcast.broadcast_to(score_var, score_grad.shape)
            broadcasted_score_vars.append(x)
        grad_vars = chainer.grad(broadcasted_score_vars, params, score_grads)
        scores = list(map(lambda x: x.data, score_vars))
        grads_and_params = []
        for param, grad_var in zip(params, grad_vars):
            grads_and_params.append((grad_var.data, param))
        aux_scores = self._aux_scores(aux_judges, yy_true, yy_pred)
        return grads_and_params, scores, aux_scores

    def variable_to_tensor(self, x):
        return x.data

    def result_to_tensor(self, x):
        return x.data

    def assign(self, x, new_value):
        x.data = new_value

    def numpy(self, x):
        return x.data.copy() if isinstance(x, chainer.Variable) else x.copy()


class BaseAPI(BaseActivationAPI, BaseDeviceDataTypeAPI, BaseEpsilonAPI,
              BaseLogicAPI, BaseMapAPI, BaseMetricAPI, BaseReduceAPI,
              BaseRelateAPI, BaseShapeAPI, BaseVariableAPI):
    def __init__(self):
        BaseActivationAPI.__init__(self)
        BaseDataTypeAPI.__init__(self)
        BaseDeviceAPI.__init__(self)
        BaseDeviceDataTypeAPI.__init__(self)
        BaseEpsilonAPI.__init__(self)
        BaseLogicAPI.__init__(self)
        BaseMapAPI.__init__(self)
        BaseMetricAPI.__init__(self)
        BaseReduceAPI.__init__(self)
        BaseRelateAPI.__init__(self)
        BaseShapeAPI.__init__(self)
        BaseVariableAPI.__init__(self)


class PyTorchAPI(BaseAPI, PyTorchActivationAPI, PyTorchDataTypeAPI,
                 PyTorchDeviceAPI, PyTorchDeviceDataTypeAPI, PyTorchLogicAPI,
                 PyTorchMapAPI, PyTorchMetricAPI, PyTorchReduceAPI,
                 PyTorchRelateAPI, PyTorchShapeAPI, PyTorchVariableAPI):
    def __init__(self):
        BaseAPI.__init__(self)
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


class MXNetAPI(BaseAPI, MXNetActivationAPI, MXNetDataTypeAPI, MXNetDeviceAPI,
               MXNetDeviceDataTypeAPI, MXNetLogicAPI, MXNetMapAPI,
               MXNetMetricAPI, MXNetReduceAPI, MXNetRelateAPI, MXNetShapeAPI,
               MXNetVariableAPI):
    def __init__(self):
        BaseAPI.__init__(self)
        MXNetActivationAPI.__init__(self)
        MXNetDataTypeAPI.__init__(self)
        MXNetDeviceAPI.__init__(self)
        MXNetDeviceDataTypeAPI.__init__(self)
        MXNetLogicAPI.__init__(self)
        MXNetMapAPI.__init__(self)
        MXNetMetricAPI.__init__(self)
        MXNetReduceAPI.__init__(self)
        MXNetRelateAPI.__init__(self)
        MXNetShapeAPI.__init__(self)
        MXNetVariableAPI.__init__(self)


class TensorFlowAPI(BaseAPI, TensorFlowActivationAPI,
                    TensorFlowDataTypeAPI, TensorFlowDeviceAPI,
                    TensorFlowDeviceDataTypeAPI, TensorFlowLogicAPI,
                    TensorFlowMapAPI, TensorFlowMetricAPI,
                    TensorFlowReduceAPI, TensorFlowRelateAPI,
                    TensorFlowShapeAPI, TensorFlowVariableAPI):
    def __init__(self):
        BaseAPI.__init__(self)
        TensorFlowActivationAPI.__init__(self)
        TensorFlowDataTypeAPI.__init__(self)
        TensorFlowDeviceAPI.__init__(self)
        TensorFlowDeviceDataTypeAPI.__init__(self)
        TensorFlowLogicAPI.__init__(self)
        TensorFlowMapAPI.__init__(self)
        TensorFlowMetricAPI.__init__(self)
        TensorFlowReduceAPI.__init__(self)
        TensorFlowRelateAPI.__init__(self)
        TensorFlowShapeAPI.__init__(self)
        TensorFlowVariableAPI.__init__(self)


class ChainerAPI(BaseAPI, ChainerActivationAPI, ChainerDeviceAPI,
                 ChainerDataTypeAPI, ChainerDeviceDataTypeAPI,
                 ChainerLogicAPI, ChainerMapAPI, ChainerMetricAPI,
                 ChainerReduceAPI, ChainerRelateAPI, ChainerShapeAPI,
                 ChainerVariableAPI):
    def __init__(self):
        BaseAPI.__init__(self)
        ChainerActivationAPI.__init__(self)
        ChainerDataTypeAPI.__init__(self)
        ChainerDeviceAPI.__init__(self)
        ChainerDeviceDataTypeAPI.__init__(self)
        ChainerLogicAPI.__init__(self)
        ChainerMapAPI.__init__(self)
        ChainerReduceAPI.__init__(self)
        ChainerRelateAPI.__init__(self)
        ChainerShapeAPI.__init__(self)
        ChainerVariableAPI.__init__(self)



BACKEND = os.environ['b']
if BACKEND == 'pytorch':
    Z = PyTorchAPI()
elif BACKEND == 'mxnet':
    Z = MXNetAPI()
elif BACKEND == 'tensorflow':
    Z = TensorFlowAPI()
elif BACKEND == 'chainer':
    Z = ChainerAPI()
else:
    assert False


class Form(object):
    def __init__(self, shape, dtype):
        assert isinstance(shape, tuple)
        for dim in shape:
            assert isinstance(dim, int)
            assert 1 <= dim
        assert dtype in Z.supported_dtypes()
        self.shape = shape
        self.dtype = dtype

    def check(self, x):
        assert Z.shape(x)[1:] == self.shape
        assert Z.dtype_of(x) == self.dtype


class Layer(object):
    def params(self):
        return []

    def forward_multi(self, xx):
        raise NotImplementedError


class MergeLayer(Layer):
    pass


class TransformLayer(Layer):
    def forward_one(self, x):
        raise NotImplementedError

    def forward_multi(self, xx):
        assert len(xx) == 1
        x, = xx
        x = self.forward_one(x)
        return [x]


class InputLayer(TransformLayer):
    def __init__(self, form):
        self.form = form

    def forward_one(self, x):
        self.form.check(x)
        return x


class DenseLayer(TransformLayer):
    def __init__(self, kernel, bias):
        if BACKEND == 'chainer':
            kernel = kernel.T
        self.kernel = Z.variable(Z.cast_numpy_to(kernel))
        self.bias = Z.variable(Z.cast_numpy_to(bias))

    def params(self):
        return [self.kernel, self.bias]

    def forward_one(self, x):
        return Z.dense(x, self.kernel, self.bias)


class FlattenLayer(TransformLayer):
    def forward_one(self, x):
        return Z.reshape(x, (Z.shape(x)[0], -1))


class ReLULayer(TransformLayer):
    def forward_one(self, x):
        return Z.clip(x, min=0)


class SoftmaxLayer(TransformLayer):
    def forward_one(self, x):
        return Z.softmax(x)


class SequenceLayer(TransformLayer):
    def __init__(self, layers):
        for layer in layers:
            assert isinstance(layer, TransformLayer)
        self.layers = layers

    def params(self):
        params = []
        for layer in self.layers:
            params += layer.params()
        return params

    def forward_one(self, x):
        for layer in self.layers:
            x = layer.forward_one(x)
        return x


class Spec(object):
    def build_multi(self, forms):
        raise NotImplementedError


class MergeSpec(Spec):
    pass


class TransformSpec(Spec):
    def build_one(self, form):
        raise NotImplementedError

    def build_multi(self, forms):
        assert len(forms) == 1
        form, = forms
        layer, form = self.build_one(form)
        return layer, [form]


class InputSpec(TransformSpec):
    def __init__(self, shape, dtype):
        self.form = Form(shape, dtype)

    def build_one(self, form):
        assert form is None
        return InputLayer(self.form), self.form


class DenseSpec(TransformSpec):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def build_one(self, form):
        assert len(form.shape) == 1
        in_dim, = form.shape
        kernel = np.random.normal(
            0, 0.1, (in_dim, self.out_dim)).astype('float32')
        bias = np.random.normal(0, 0.1, (self.out_dim,)).astype('float32')
        out_shape = self.out_dim,
        return DenseLayer(kernel, bias), Form(out_shape, form.dtype)


class FlattenSpec(TransformSpec):
    def build_one(self, form):
        out_shape = (int(np.prod(form.shape)),)
        form = Form(out_shape, form.dtype)
        return FlattenLayer(), form


class ReLUSpec(TransformSpec):
    def build_one(self, form):
        return ReLULayer(), form


class SoftmaxSpec(TransformSpec):
    def build_one(self, form):
        return SoftmaxLayer(), form


class SequenceSpec(TransformSpec):
    def __init__(self, specs):
        self.specs = specs

    def build_one(self, form):
        layers = []
        for spec in self.specs:
            layer, form = spec.build_one(form)
            layers.append(layer)
        return SequenceLayer(layers), form


class Metric(object):
    def __init__(self, importance=1):
        self.importance = importance

    def __call__(self, true, pred):
        raise NotImplementedError


class Loss(Metric):
    pass


class BinaryCrossEntropy(Loss):
    def __call__(self, true, pred):
        return Z.binary_cross_entropy(true, pred)


class CategoricalCrossEntropy(Loss):
    def __call__(self, true, pred):
        return Z.categorical_cross_entropy(true, pred)


class MeanSquaredError(Loss):
    def __call__(self, true, pred):
        return Z.mean_squared_error(true, pred)


class CategoricalAccuracy(Metric):
    def __call__(self, true, pred):
        return Z.categorical_accuracy(true, pred)


class Optimizer(object):
    def set_params(self, params):
        self.params = params

    def update_param(self, gradient, param):
        raise NotImplementedError

    def update(self, grads_and_params):
        for grad, param in grads_and_params:
            self.update_param(grad, param)


class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def update_param(self, gradient, variable):
        Z.assign(variable, Z.variable_to_tensor(variable) - self.lr * gradient)


def one_hot(indices, num_classes, dtype):
    assert indices.ndim == 1
    assert isinstance(num_classes, int)
    assert 0 < num_classes
    assert dtype in Z.supported_dtypes()
    x = np.zeros((len(indices), num_classes), dtype)
    x[np.arange(len(indices)), indices] = 1
    return x


def scale_pixels(x):
    return (x / 255 - 0.5) * 2


def get_data(dtype):
    (x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, 1).astype(dtype)
    x_train = scale_pixels(x_train)
    y_train = one_hot(y_train, 10, dtype)
    x_val = np.expand_dims(x_val, 1).astype(dtype)
    x_val = scale_pixels(x_val)
    y_val = one_hot(y_val, 10, dtype)
    return (x_train, y_train), (x_val, y_val)


dtype = Z.default_dtype()
image_shape = 1, 28, 28
hidden_dim = 100
lr = 0.05
num_epochs = 10
batch_size = 64

(x_train, y_train), (x_val, y_val) = get_data(dtype)

num_classes = y_train.shape[1]
batches_per_epoch = len(x_train) // batch_size

model = SequenceSpec([
    InputSpec(image_shape, dtype),
    FlattenSpec(),
    DenseSpec(hidden_dim),
    ReLUSpec(),
    DenseSpec(num_classes),
    SoftmaxSpec(),
])
model, out_shape = model.build_one(None)

opt = SGD(lr)
opt.set_params(model.params())

cxe = CategoricalCrossEntropy()
acc = CategoricalAccuracy()
judges = [cxe]
aux_judges = [[acc]]

for epoch_id in range(num_epochs):
    for batch_id in range(batches_per_epoch):
        i = batch_id * batch_size
        x = x_train[i:i + batch_size]
        x = Z.constant(Z.cast_numpy_to(x))
        y = y_train[i:i + batch_size]
        y = Z.constant(Z.cast_numpy_to(y))
        grads_and_params, losses, aux_metrics = Z.gradients(
            opt.params, model.forward_multi, judges, aux_judges, [x], [y])
        loss = Z.scalar(losses[0])
        acc = Z.scalar(aux_metrics[0][0])
        assert not np.isnan(loss)
        assert not np.isnan(acc)
        print('epoch %4d batch %4d loss %.4f acc %.4f' %
            (epoch_id, batch_id, loss, acc))
        opt.update(grads_and_params)
