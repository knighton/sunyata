import numpy as np
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


class MapAPI(BaseMapAPI):
    def clip(self, x, min=-np.inf, max=np.inf):
        return x.clamp(min, max)


class BaseRelateAPI(APIBase):
    def matmul(self, a, b):
        raise NotImplementedError


class RelateAPI(BaseRelateAPI):
    def matmul(self, a, b):
        return a.mm(b)


class BaseShapeAPI(APIBase):
    def ndim(self, x):
        raise NotImplementedError

    def shape(self, x):
        raise NotImplementedError

    def size(self, x):
        raise NotImplementedError


class ShapeAPI(BaseShapeAPI):
    def ndim(self, x):
        return len(x.size())

    def shape(self, x):
        return tuple(x.size())

    def size(self, x):
        return x.nelement()


class BaseDeviceDataTypeAPI(APIBase):
    def __init__(self, supported_dtypes, default_dtype):
        self._supported_dtypes = supported_dtypes
        assert default_dtype in supported_dtypes
        self._default_dtype = default_dtype

    def num_gpus(self):
        raise NotImplementedError

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
        return self.cast_onto(x, dtype, None, copy)

    def dtype_of(self, x):
        raise NotImplementedError

    def device_of(self, x):
        raise NotImplementedError

    def cast_onto(self, x, dtype=None, device=None, copy=False):
        raise NotImplementedError

    def cast_numpy_onto(self, x, dtype=None, device=None):
        raise NotImplementedError

    def to_device(self, x, device=None, copy=False):
        return self.cast_onto(x, None, device, copy)

    def to_cpu(self, x, copy=False):
        return self.to_device(x, 0, copy)

    def to_gpu(self, x, device, copy=False):
        device = self.to_device(device)
        assert device.is_gpu()
        return self.to_device(x, device, copy)


class DeviceDataTypeAPI(BaseDeviceDataTypeAPI):
    def __init__(self, num_gpus):
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

        self._supported_dtypes = sorted(self._dtype2cpu)
        self._default_dtype = 'float32'

        self._devices = []
        for device_id in range(num_gpus + 1):
            device = Device(device_id)
            self._devices.append(device)
        self._default_device = self._devices[-1]

    def num_gpus(self):
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
            dtype2class = self._dtype2cpu[dtype]
        elif device.is_gpu():
            dtype2class = self._dtype2gpu[dtype]
        else:
            assert False
        return dtype2class[dtype]

    def cast_onto(self, x, dtype=None, device=None, copy=True):
        from_dtype = self.dtype_of(x)
        to_dtype = from_dtype if dtype is None else self.dtype(to_dtype)
        from_device = self.device_of(x)
        to_device = from_device if device is None else self.device(device)
        to_tensor_class = self._get_tensor_class(to_dtype, device)
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

    def cast_numpy_onto(self, x, dtype=None, device=None):
        from_dtype = x.dtype
        to_dtype = from_dtype if dtype is None else self.dtype(dtype)
        from_device = self._devices[0]
        to_device = from_device if device is None else self.device(to_device)
        to_tensor_class = self._get_tensor_class(to_dtype, device)
        if to_device.is_cpu():
            x = tensor_class(x)
        elif to_device.is_gpu():
            with torch.cuda.device(to_device.gpu_id()):
                x = tensor_class(x)
        else:
            assert False
        return x


class BaseVariableAPI(APIBase):
    def constant(self, tensor):
        raise NotImplementedError

    def variable(self, tensor):
        raise NotImplementedError

    def assign(self, x, new_value):
        raise NotImplementedError


class VariableAPI(BaseVariableAPI):
    def constant(self, tensor):
        return Variable(tensor, requires_grad=False)

    def variable(self, tensor):
        return Variable(tensor, requires_grad=True)

    def assign(self, x, new_value):
        x.data = new_value
        x.grad.data.zero_()


class API(MapAPI, RelateAPI, ShapeAPI, DeviceDataTypeAPI, VariableAPI):
    def __init__(self):
        MapAPI.__init__(self)
        RelateAPI.__init__(self)
        ShapeAPI.__init__(self)
        DeviceDataTypeAPI.__init__(self, self.num_gpus())
        VariableAPI.__init__(self)


Z = API()


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
        self.kernel = Z.variable(Z.cast_numpy_onto(kernel))
        self.bias = Z.variable(Z.cast_numpy_onto(bias))

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
    return (true - pred).pow(2).sum()


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
        Z.assign(param, param.data - self.lr * param.grad.data)


batch_size = 64
in_dim = 1000
hidden_dim = 100
num_classes = 10
lr = 1e-6

x = np.random.normal(0, 1, (batch_size, in_dim)).astype('float32')
x = Z.constant(Z.cast_numpy_onto(x))

y = np.random.normal(0, 1, (batch_size, num_classes)).astype('float32')
y = Z.constant(Z.cast_numpy_onto(y))

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
    y_pred = model.forward(x)

    loss = mean_squared_error(y, y_pred)
    print(t, loss.data[0])

    loss.backward()

    opt.update()
