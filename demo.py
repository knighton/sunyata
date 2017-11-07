import numpy as np
import torch
from torch.autograd import Variable


class API(object):
    def dtype_of(self, x):
        assert isinstance(x, torch.FloatTensor) or \
            (isinstance(x, Variable) and isinstance(x.data, torch.FloatTensor))
        return np.float32

    def clip(self, x, min=-np.inf, max=np.inf):
        return x.clamp(min=0)

    def rank(self, x):
        return len(x.size())

    def shape(self, x):
        return tuple(x.size())

    def size(self, x):
        return x.nelement()

    def matmul(self, a, b):
        return a.mm(b)

    def default_dtype(self):
        return np.float32

    def dtype(self, x):
        return x or self.default_dtype()

    def device(self, x):
        assert x is None
        return x

    def cast_tensor_onto(self, x, dtype=None, device=None, copy=False):
        dtype = self.dtype(dtype)
        assert dtype == np.float32
        if isinstance(x, np.ndarray):
            print('!', x.dtype)
            assert x.dtype == 'float32'
        else:
            print("!", x)
            assert isinstance(x, torch.FloatTensor)
        device = self.device(device)
        assert copy is False
        return torch.from_numpy(x)


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
        self.kernel = Variable(Z.cast_tensor_onto(kernel), requires_grad=True)
        self.bias = Variable(Z.cast_tensor_onto(bias), requires_grad=True)

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
        param.data -= self.lr * param.grad.data
        param.grad.data.zero_()


batch_size, in_dim, hidden_dim, num_classes = 64, 1000, 100, 10
lr = 1e-6

x = np.random.normal(0, 1, (batch_size, in_dim)).astype('float32')
x = Variable(Z.cast_tensor_onto(x), requires_grad=False)

y = np.random.normal(0, 1, (batch_size, num_classes)).astype('float32')
y = Variable(Z.cast_tensor_onto(y), requires_grad=False)

model = SequenceSpec([
    InputSpec((in_dim,), np.float32),
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
