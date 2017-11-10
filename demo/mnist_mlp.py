import chainer
import chainer.functions as CHF
import keras
import numpy as np
import os

from sunyata.backend.base import \
    Device, BaseActivationAPI, BaseDataTypeAPI, BaseDeviceAPI, \
    BaseDeviceDataTypeAPI, BaseEpsilonAPI, BaseLogicAPI, BaseMapAPI, \
    BaseMetricAPI, BaseReduceAPI, BaseRelateAPI, BaseShapeAPI, \
    BaseVariableAPI, BaseAPI
from sunyata.backend.mxnet import MXNetAPI
from sunyata.backend.pytorch import PyTorchAPI
from sunyata.backend.tensorflow import TensorFlowAPI



class ChainerActivationAPI(BaseActivationAPI):
    def softmax(self, x):
        return CHF.softmax(x)


class ChainerLogicAPI(BaseLogicAPI):
    def equal(self, a, b):
        return chainer.Variable((a.data == b.data).astype(a.dtype))


class ChainerMapAPI(BaseMapAPI):
    def clip(self, x, min=-np.inf, max=np.inf):
        return CHF.clip(x, float(min), float(max))

    def log(self, x):
        return CHF.math.exponential.log(x)

    def pow(self, x, a):
        return CHF.math.basic_math.pow(x, a)


class ChainerMetricAPI(BaseMetricAPI):
    pass


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


class ChainerRelateAPI(BaseRelateAPI):
    def dense(self, x, kernel, bias):
        return CHF.connection.linear.linear(x, kernel, bias)


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
            arr = np.ones((1,), self.dtype_of(y_true)) * judge.importance
            score_grads.append(self.cast_numpy_to(arr))
        grad_vars = chainer.grad(score_vars, params, score_grads)
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
        x.data = new_value.copy()

    def numpy(self, x):
        return x.data.copy() if isinstance(x, chainer.Variable) else x.copy()


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
