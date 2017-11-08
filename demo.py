from contextlib import contextmanager
import importlib
import mxnet as mx
import numpy as np
import subprocess
import torch
from torch.autograd import Variable


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

    def pow(self, x, exp):
        raise NotImplementedError


class PyTorchMapAPI(BaseMapAPI):
    def clip(self, x, min=-np.inf, max=np.inf):
        return x.clamp(min, max)

    def pow(self, x, exp):
        return x ** exp


class MXNetMapAPI(BaseMapAPI):
    def clip(self, x, min=-np.inf, max=np.inf):
        return mx.nd.clip(x, min, max)

    def pow(self, x, a):
        return mx.nd.power(x, a)


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


class BaseRelateAPI(APIBase):
    def matmul(self, a, b):
        raise NotImplementedError


class PyTorchRelateAPI(BaseRelateAPI):
    def matmul(self, a, b):
        return a.mm(b)


class MXNetRelateAPI(BaseRelateAPI):
    def matmul(self, a, b):
        return mx.nd.dot(a, b)


class BaseShapeAPI(APIBase):
    def ndim(self, x):
        raise NotImplementedError

    def shape(self, x):
        raise NotImplementedError

    def size(self, x):
        raise NotImplementedError


class PyTorchShapeAPI(BaseShapeAPI):
    def ndim(self, x):
        return len(x.size())

    def shape(self, x):
        return tuple(x.size())

    def size(self, x):
        return x.nelement()


class MXNetShapeAPI(BaseShapeAPI):
    def ndim(self, x):
        return len(x.ndim)

    def shape(self, x):
        return x.shape

    def size(self, x):
        return x.size


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

    def dtype(self, x):
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
        elif isinstance(x, Variable):
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
            if to_device.is_cpu():
                ctx = mx.cpu()
            elif to_device.is_gpu():
                ctx = mx.gpu(to_device.gpu_id())
            else:
                assert False
            x = x.as_in_context(ctx)
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
        if to_device.is_cpu():
            ctx = mx.cpu()
        elif to_device.is_gpu():
            ctx = mx.gpu(to_device.gpu_id())
        else:
            assert False
        return mx.nd.array(x, ctx, to_dtype)


class BaseVariableAPI(APIBase):
    def constant(self, x):
        raise NotImplementedError

    def variable(self, x):
        raise NotImplementedError

    @contextmanager
    def autograd_record(self):
        yield

    def data(self, x):
        raise NotImplementedError

    def grad(self, x):
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
        return Variable(x.clone(), requires_grad=False)

    def variable(self, x):
        return Variable(x.clone(), requires_grad=True)

    def data(self, x):
        if isinstance(x, torch._TensorBase):
            pass
        elif isinstance(x, Variable):
            x = x.data
        else:
            assert False
        return x

    def grad(self, x):
        if isinstance(x, Variable):
            x = x.grad.data
        else:
            assert False
        return x

    def assign(self, x, new_value):
        x.data = new_value
        x.grad.data.zero_()

    def numpy(self, x):
        if isinstance(x, torch._TensorBase):
            pass
        elif isinstance(x, Variable):
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

    @contextmanager
    def autograd_record(self):
        with mx.autograd.record():
            yield

    def data(self, x):
        return x

    def grad(self, x):
        return x.grad

    def assign(self, x, new_value):
        x[:] = new_value
        x.grad[:] = 0

    def numpy(self, x):
        return x.asnumpy()


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


# Z = PyTorchAPI()
Z = MXNetAPI()


class Form(object):
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def check(self, x):
        assert tuple(Z.shape(x)[1:]) == self.shape
        assert Z.dtype_of(x) == self.dtype


class Layer(object):
    def params(self):
        return []

    def forward(self, x):
        raise NotImplementedError


class InputLayer(Layer):
    def __init__(self, form):
        self.form = form

    def forward(self, x):
        self.form.check(x)
        return x


class DenseLayer(Layer):
    def __init__(self, kernel, bias):
        self.kernel = Z.variable(Z.cast_numpy_to(kernel))
        self.bias = Z.variable(Z.cast_numpy_to(bias))

    def params(self):
        return [self.kernel, self.bias]

    def forward(self, x):
        return Z.matmul(x, self.kernel) + self.bias


class ReLULayer(Layer):
    def forward(self, x):
        return Z.clip(x, min=0)


class SequenceLayer(Layer):
    def __init__(self, layers):
        self.layers = layers

    def params(self):
        params = []
        for layer in self.layers:
            params += layer.params()
        return params

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


class Spec(object):
    def build(self, form=None):
        raise NotImplementedError


class InputSpec(Spec):
    def __init__(self, shape, dtype):
        self.form = Form(shape, dtype)

    def build(self, form=None):
        assert form is None
        return InputLayer(self.form), self.form


class DenseSpec(Spec):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def build(self, form=None):
        in_dim, = form.shape
        kernel = np.random.normal(
            0, 1, (in_dim, self.out_dim)).astype('float32')
        bias = np.random.normal(0, 1, (self.out_dim,)).astype('float32')
        out_shape = self.out_dim,
        return DenseLayer(kernel, bias), Form(out_shape, form.dtype)


class ReLUSpec(Spec):
    def build(self, form=None):
        return ReLULayer(), form


class SequenceSpec(Spec):
    def __init__(self, specs):
        self.specs = specs

    def build(self, form=None):
        layers = []
        for spec in self.specs:
            layer, form = spec.build(form)
            layers.append(layer)
        return SequenceLayer(layers), form


def mean_squared_error(true, pred):
    return Z.sum(Z.pow(true - pred, 2))


class Optimizer(object):
    def set_params(self, params):
        self.params = params

    def update_param(self, param):
        raise NotImplementedError

    def update(self):
        for param in self.params:
            self.update_param(param)


class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def update_param(self, param):
        Z.assign(param, Z.data(param) - self.lr * Z.grad(param))


batch_size = 64
in_dim = 1000
hidden_dim = 100
num_classes = 10
lr = 1e-6
x = np.random.normal(0, 1, (batch_size, in_dim)).astype('float32')
x = Z.constant(Z.cast_numpy_to(x))
y = np.random.normal(0, 1, (batch_size, num_classes)).astype('float32')
y = Z.constant(Z.cast_numpy_to(y))
model = SequenceSpec([
    InputSpec((in_dim,), 'float32'),
    DenseSpec(hidden_dim),
    ReLUSpec(),
    DenseSpec(num_classes),
])
model, out_shape = model.build()
opt = SGD(lr)
opt.set_params(model.params())
for t in range(500):
    with Z.autograd_record():
        y_pred = model.forward(x)
        loss = mean_squared_error(y, y_pred)
    print(t, Z.scalar(loss))
    loss.backward()
    opt.update()
