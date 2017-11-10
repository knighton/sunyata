import chainer
import chainer.functions as F
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


class BaseMapAPI(APIBase):
    def clip(self, x, min=-np.inf, max=np.inf):
        raise NotImplementedError

    def pow(self, x, a):
        raise NotImplementedError


class PyTorchMapAPI(BaseMapAPI):
    def clip(self, x, min=-np.inf, max=np.inf):
        return x.clamp(min, max)

    def pow(self, x, a):
        return x ** a


class TensorFlowMapAPI(BaseMapAPI):
    def clip(self, x, min=-np.inf, max=np.inf):
        return tf.clip_by_value(x, min, max)

    def pow(self, x, a):
        return tf.pow(x, a)


class MXNetMapAPI(BaseMapAPI):
    def clip(self, x, min=-np.inf, max=np.inf):
        return mx.nd.clip(x, min, max)

    def pow(self, x, a):
        return mx.nd.power(x, a)


class ChainerMapAPI(BaseMapAPI):
    def clip(self, x, min=-np.inf, max=np.inf):
        return F.clip(x, float(min), float(max))

    def pow(self, x, a):
        return F.math.basic_math.pow(x, a)


class BaseReduceAPI(APIBase):
    def sum(self, x, axis=None, keepdims=False):
        raise NotImplementedError


class PyTorchReduceAPI(BaseReduceAPI):
    @classmethod
    def _normalize_axis(cls, axis, keepdims, ndim):
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

    @classmethod
    def _reduce(cls, name, x, axis=None, keepdims=False):
        axes = cls._normalize_axis(axis, keepdims, x.dim())
        if axes is None:
            return getattr(x, name)()
        for axis in axes:
            x = getattr(x, name)(axis, keepdims)
            if isinstance(x, tuple):
                x = x[0]
        if not keepdims:
            for axis in reversed(axes):
                x = x.squeeze(axis)
        return x

    def sum(self, x, axis=None, keepdims=False):
        return self._reduce('sum', x, axis, keepdims)


class MXNetReduceAPI(BaseReduceAPI):
    @classmethod
    def _reduce(cls, name, x, axis=None, keepdims=False):
        axis = mx.base._Null if axis is None else axis
        func = getattr(mx.nd, name)
        return func(x, axis, keepdims)

    def sum(self, x, axis=None, keepdims=False):
        return self._reduce('sum', x, axis, keepdims)


class TensorFlowReduceAPI(BaseReduceAPI):
    def sum(self, x, axis=None, keepdims=False):
        return tf.reduce_sum(x, axis, keepdims)


class ChainerReduceAPI(BaseReduceAPI):
    @classmethod
    def _reduce(cls, func, x, axis=None, keepdims=False):
        axis = tuple(axis) if isinstance(axis, list) else axis
        return func(x, axis, keepdims)

    def sum(self, x, axis=None, keepdims=False):
        return self._reduce(F.math.sum.sum, x, axis, keepdims)


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
        return F.connection.linear.linear(x, kernel, bias)


class BaseShapeAPI(APIBase):
    def ndim(self, x):
        raise NotImplementedError

    def shape(self, x):
        raise NotImplementedError

    def size(self, x):
        raise NotImplementedError

    def reshape(self, x, shape):
        raise NotImplementedError


class PyTorchShapeAPI(BaseShapeAPI):
    def ndim(self, x):
        return len(x.size())

    def shape(self, x):
        return tuple(x.size())

    def size(self, x):
        return x.nelement()

    def reshape(self, x, shape):
        return x.view(shape)


class MXNetShapeAPI(BaseShapeAPI):
    def ndim(self, x):
        return len(x.ndim)

    def shape(self, x):
        return x.shape

    def size(self, x):
        return x.size

    def reshape(self, x, shape):
        return mx.nd.reshape(x, shape)


class TensorFlowShapeAPI(BaseShapeAPI):
    def ndim(self, x):
        return int(tf.rank(x).numpy())

    def shape(self, x):
        return x.shape

    def size(self, x):
        return int(tf.size(x).numpy())

    def reshape(self, x, shape):
        return tf.reshape(x, shape)


class ChainerShapeAPI(BaseShapeAPI):
    def ndim(self, x):
        return x.ndim

    def shape(self, x):
        return x.shape

    def size(self, x):
        return x.size

    def reshape(self, x, shape):
        return x.reshape(shape)


class BaseDeviceDTypeAPI(APIBase):
    def __init__(self, num_gpus, default_device_id, supported_dtypes,
                 default_dtype):
        self._devices = []
        for device_id in range(num_gpus + 1):
            device = Device(device_id)
            self._devices.append(device)
        assert isinstance(default_device_id, int)
        assert 0 <= default_device_id < len(self._devices)
        self._default_device = self._devices[default_device_id]
        self._supported_dtypes = supported_dtypes
        assert default_dtype in supported_dtypes
        self._default_dtype = default_dtype

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

    def num_devices(self):
        return len(self._devices)

    def num_gpus(self):
        return len(self._devices) - 1

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

    def cast(self, x, dtype=None, copy=False):
        return self.cast_to(x, dtype, None, copy)

    def dtype_of(self, x):
        raise NotImplementedError

    def device_of(self, x):
        raise NotImplementedError

    def cast_to(self, x, dtype=None, device=None, copy=False):
        raise NotImplementedError

    def cast_numpy_to(self, x, dtype=None, device=None):
        raise NotImplementedError

    def to(self, x, device=None, copy=False):
        return self.cast_to(x, None, device, copy)

    def to_cpu(self, x, copy=False):
        return self.to_device(x, 0, copy)

    def to_gpu(self, x, device, copy=False):
        device = self.to_device(device)
        assert device.is_gpu()
        return self.to_device(x, device, copy)


class PyTorchDeviceDTypeAPI(BaseDeviceDTypeAPI):
    def __init__(self):
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
        num_gpus = self.discover_gpus()
        default_device_id = 1 if num_gpus else 0
        supported_dtypes = sorted(self._dtype2cpu)
        default_dtype = 'float32'
        BaseDeviceDTypeAPI.__init__(
            self, num_gpus, default_device_id, supported_dtypes, default_dtype)

    def discover_gpus(self):
        return torch.cuda.device_count()

    def dtype_of(self, x):
        if isinstance(x, torch._TensorBase):
            tensor = x
        elif isinstance(x, PTVariable):
            tensor = x.data
        else:
            assert False
        return self._tensor2dtype[tensor.type()]

    def device_of(self, x):
        if x.is_cuda:
            device_id = x.get_device()
        else:
            device_id = 0
        return self._devices[device_id]

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


class MXNetDeviceDTypeAPI(BaseDeviceDTypeAPI):
    def __init__(self):
        num_gpus = self.discover_gpus()
        default_device_id = 1 if num_gpus else 0
        supported_dtypes = sorted("""
            uint8
            int8 int16 int32 int64
            float16 float32 float64
        """.split())
        default_dtype = 'float32'
        BaseDeviceDTypeAPI.__init__(
            self, num_gpus, default_device_id, supported_dtypes, default_dtype)
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

    def dtype_of(self, x):
        return x.dtype.__name__

    def device_of(self, x):
        if x.context.device_type == 'cpu':
            device_id = 0
        elif x.context.device_type == 'gpu':
            device_id = x.context.device_id + 1
        else:
            assert False
        return self._devices[device_id]

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


class TensorFlowDeviceDTypeAPI(BaseDeviceDTypeAPI):
    def __init__(self):
        num_gpus = tfe.num_gpus()
        default_device_id = 1 if num_gpus else 0
        supported_dtypes = sorted("""
            bool
            uint8 uint16 uint32 uint64
            int8 int16 int32 int64
            float16 float32 float64
        """.split())
        default_dtype = 'float32'
        BaseDeviceDTypeAPI.__init__(
            self, num_gpus, default_device_id, supported_dtypes, default_dtype)

    def discover_gpus(self):
        return tfe.num_gpus()

    def dtype_of(self, x):
        return x.dtype.name

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
        return device_id

    def _device_name(self, device):
        if device.is_cpu():
            name = 'cpu:0'
        elif device.is_gpu():
            name = 'gpu:%d' % device.gpu_id()
        else:
            assert False
        return name

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
        if from_device is not to_device:
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


class ChainerDeviceDTypeAPI(BaseDeviceDTypeAPI):
    def __init__(self):
        num_gpus = 0
        default_device_id = 0
        supported_dtypes = sorted("""
            bool
            uint8 uint16 uint32 uint64
            int8 int16 int32 int64
            float16 float32 float64
        """.split())
        default_dtype = 'float32'
        BaseDeviceDTypeAPI.__init__(
            self, num_gpus, default_device_id, supported_dtypes, default_dtype)

    def discover_gpus(self):
        return 0

    def dtype_of(self, x):
        return x.dtype.name

    def device_of(self, x):
        return self._devices[0]

    def cast_to(self, x, dtype=None, device=None, copy=True):
        from_dtype = self.dtype_of(x)
        to_dtype = from_dtype if dtype is None else self.dtype(dtype)
        from_device = self.device_of(x)
        assert from_device is self._devices[0]
        to_device = from_device if device is None else self.device(device)
        assert to_device is self._devices[0]
        if from_dtype != to_dtype or copy:
            x = F.cast(x, to_dtype)
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

    def gradients(self, params, forward, judges, xx, yy_true):
        raise NotImplementedError

    def data(self, x):
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

    def gradients(self, variables, forward, judges, xx, yy_true):
        yy_pred = forward(xx)
        loss_variables = []
        loss_gradients = []
        for judge, y_true, y_pred in zip(judges, yy_true, yy_pred):
            loss_variables.append(judge(y_true, y_pred))
            arr = np.ones((1,), Z.dtype_of(y_true)) * judge.importance
            loss_gradients.append(self.cast_numpy_to(arr))
        torch.autograd.backward(loss_variables, loss_gradients)
        loss_tensors = list(map(lambda x: x.data, loss_variables))
        grads_and_vars = list(map(lambda x: (x.grad.data, x), variables))
        return loss_tensors, grads_and_vars

    def data(self, x):
        if isinstance(x, torch._TensorBase):
            pass
        elif isinstance(x, PTVariable):
            x = x.data
        else:
            assert False
        return x

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

    def gradients(self, variables, forward, judges, xx, yy_true):
        loss_variables = []
        loss_gradients = []
        with mx.autograd.record():
            yy_pred = forward(xx)
            for judge, y_true, y_pred in zip(judges, yy_true, yy_pred):
                loss_variables.append(judge(y_true, y_pred))
                arr = np.ones((1,), Z.dtype_of(y_true)) * judge.importance
                loss_gradients.append(self.cast_numpy_to(arr))
        mx.autograd.backward(loss_variables, loss_gradients)
        loss_tensors = loss_variables
        grads_and_vars = list(map(lambda x: (x.grad, x), variables))
        return loss_tensors, grads_and_vars

    def data(self, x):
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

    def gradients(self, variables, forward, judges, xx, yy_true):
        params_set = set(variables)
        assert len(params_set) == len(variables)
        def get_losses():
            yy_pred = forward(xx)
            loss_tensors = []
            for judge, y_true, y_pred in zip(judges, yy_true, yy_pred):
                loss_tensors.append(judge(y_true, y_pred) * judge.importance)
            return loss_tensors
        return tfe.implicit_value_and_gradients(get_losses)()

    def data(self, x):
        return x[:]

    def assign(self, x, new_value):
        x.assign(new_value)

    def numpy(self, x):
        return x.numpy()


class ChainerVariableAPI(BaseVariableAPI):
    def constant(self, x):
        return x

    def variable(self, x):
        return chainer.Variable(x)

    def gradients(self, variables, forward, judges, xx, yy_true):
        yy_pred = forward(xx)
        loss_variables = []
        loss_gradients = []
        for judge, y_true, y_pred in zip(judges, yy_true, yy_pred):
            loss_variables.append(judge(y_true, y_pred))
            arr = np.ones(Z.shape(y_true), Z.dtype_of(y_true)) * \
                judge.importance
            loss_gradients.append(self.cast_numpy_to(arr))
            loss_gradients.append(None)
        gradients = chainer.grad(loss_variables, variables, loss_gradients)
        loss_tensors = list(map(lambda x: x.data, loss_variables))
        grads_and_vars = []
        for var, grad in zip(variables, gradients):
            grads_and_vars.append((grad.data, var))
        return loss_tensors, grads_and_vars

    def data(self, x):
        return x.data

    def assign(self, x, new_value):
        x.data = new_value

    def numpy(self, x):
        return x.data.copy() if isinstance(x, chainer.Variable) else x.copy()


class BaseAPI(BaseDeviceDTypeAPI, BaseMapAPI, BaseReduceAPI, BaseRelateAPI,
              BaseShapeAPI, BaseVariableAPI):
    def __init__(self):
        BaseDeviceDTypeAPI.__init__(self)
        BaseMapAPI.__init__(self)
        BaseReduceAPI.__init__(self)
        BaseRelateAPI.__init__(self)
        BaseShapeAPI.__init__(self)
        BaseVariableAPI.__init__(self)


class PyTorchAPI(PyTorchDeviceDTypeAPI, PyTorchMapAPI, PyTorchReduceAPI,
                 PyTorchRelateAPI, PyTorchShapeAPI, PyTorchVariableAPI):
    def __init__(self):
        PyTorchDeviceDTypeAPI.__init__(self)
        PyTorchMapAPI.__init__(self)
        PyTorchReduceAPI.__init__(self)
        PyTorchRelateAPI.__init__(self)
        PyTorchShapeAPI.__init__(self)
        PyTorchVariableAPI.__init__(self)


class MXNetAPI(MXNetDeviceDTypeAPI, MXNetMapAPI, MXNetReduceAPI, MXNetRelateAPI,
               MXNetShapeAPI, MXNetVariableAPI):
    def __init__(self):
        MXNetDeviceDTypeAPI.__init__(self)
        MXNetMapAPI.__init__(self)
        MXNetReduceAPI.__init__(self)
        MXNetRelateAPI.__init__(self)
        MXNetShapeAPI.__init__(self)
        MXNetVariableAPI.__init__(self)


class TensorFlowAPI(TensorFlowDeviceDTypeAPI, TensorFlowMapAPI,
                    TensorFlowReduceAPI, TensorFlowRelateAPI,
                    TensorFlowShapeAPI, TensorFlowVariableAPI):
    def __init__(self):
        TensorFlowDeviceDTypeAPI.__init__(self)
        TensorFlowMapAPI.__init__(self)
        TensorFlowReduceAPI.__init__(self)
        TensorFlowRelateAPI.__init__(self)
        TensorFlowShapeAPI.__init__(self)
        TensorFlowVariableAPI.__init__(self)


class ChainerAPI(ChainerDeviceDTypeAPI, ChainerMapAPI, ChainerReduceAPI,
                 ChainerRelateAPI, ChainerShapeAPI, ChainerVariableAPI):
    def __init__(self):
        ChainerDeviceDTypeAPI.__init__(self)
        ChainerMapAPI.__init__(self)
        ChainerReduceAPI.__init__(self)
        ChainerRelateAPI.__init__(self)
        ChainerShapeAPI.__init__(self)
        ChainerVariableAPI.__init__(self)



BACKEND = os.environ['b']
if BACKEND == 'pytorch':
    Z = PyTorchAPI()
elif BACKEND == 'mxnet':
    Z = PyTorchAPI()
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


class SequenceSpec(TransformSpec):
    def __init__(self, specs):
        self.specs = specs

    def build_one(self, form):
        layers = []
        for spec in self.specs:
            layer, form = spec.build_one(form)
            layers.append(layer)
        return SequenceLayer(layers), form


def mean_squared_error(true, pred):
    return Z.sum(Z.pow(true - pred, 2))


class Loss(object):
    def __init__(self, importance=1):
        self.importance = importance

    def __call__(self, true, pred):
        raise NotImplementedError


class MeanSquaredError(Loss):
    def __call__(self, true, pred):
        return mean_squared_error(true, pred)


class Optimizer(object):
    def set_params(self, params):
        self.params = params

    def update_param(self, gradient, variable):
        raise NotImplementedError

    def update(self, grads_and_vars):
        for gradient, variable in grads_and_vars:
            self.update_param(gradient, variable)


class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def update_param(self, gradient, variable):
        Z.assign(variable, Z.data(variable) - self.lr * gradient)


def one_hot(indices, num_classes, dtype):
    assert indices.ndim == 1
    assert isinstance(num_classes, int)
    assert 0 < num_classes
    assert dtype in Z.supported_dtypes()
    x = np.zeros((len(indices), num_classes), dtype)
    x[np.arange(len(indices)), indices] = 1
    return x


def get_data(dtype):
    (x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, 1).astype(dtype)
    y_train = one_hot(y_train, 10, dtype)
    x_val = np.expand_dims(x_val, 1).astype(dtype)
    y_val = one_hot(y_val, 10, dtype)
    return (x_train, y_train), (x_val, y_val)


dtype = Z.default_dtype()
image_shape = 1, 28, 28
hidden_dim = 100
lr = 1e-6
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
])
model, out_shape = model.build_one(None)

opt = SGD(lr)
opt.set_params(model.params())

mse = MeanSquaredError()

for epoch_id in range(num_epochs):
    for batch_id in range(batches_per_epoch):
        i = batch_id * batch_size
        x = x_train[i:i + batch_size]
        x = Z.constant(Z.cast_numpy_to(x))
        y = y_train[i:i + batch_size]
        y = Z.constant(Z.cast_numpy_to(y))
        loss_tensors, grads_and_vars = Z.gradients(
            opt.params, model.forward_multi, [mse], [x], [y])
        loss_tensor, = loss_tensors
        loss_scalar = Z.scalar(loss_tensor)
        assert not np.isnan(loss_scalar)
        print('epoch %4d batch %4d: %.6f' % (epoch_id, batch_id, loss_scalar))
        opt.update(grads_and_vars)
