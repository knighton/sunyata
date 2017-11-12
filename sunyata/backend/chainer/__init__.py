import chainer
import chainer.functions as F
import numpy as np

from ..base import \
    BaseActivationAPI, BaseDataTypeAPI, BaseDeviceAPI, BaseDeviceDataTypeAPI, \
    BaseLogicAPI, BaseMapAPI, BaseMetricAPI, BaseReduceAPI, BaseRelateAPI, \
    BaseShapeAPI, BaseVariableAPI, BaseBackend


class ChainerActivationAPI(BaseActivationAPI):
    def softmax(self, x):
        return F.softmax(x)


class ChainerLogicAPI(BaseLogicAPI):
    def minimum(self, a, b):
        assert a.dtype == b.dtype
        if a.dtype.name.startswith('float'):
            x = F.minimum(a, b)
        else:
            x = chainer.Variable(np.minimum(a.data, b.data))
        return x

    def maximum(self, a, b):
        assert a.dtype == b.dtype
        if a.dtype.name.startswith('float'):
            x = F.maximum(a, b)
        else:
            x = chainer.Variable(np.maximum(a.data, b.data))
        return x

    def equal(self, a, b, dtype=None):
        data = a.data == b.data
        x = chainer.Variable(data)
        return self._cast_logic_output(a, x, dtype)

    def not_equal(self, a, b, dtype=None):
        data = a.data != b.data
        x = chainer.Variable(data)
        return self._cast_logic_output(a, x, dtype)

    def less(self, a, b, dtype=None):
        data = a.data < b.data
        x = chainer.Variable(data)
        return self._cast_logic_output(a, x, dtype)

    def less_equal(self, a, b, dtype=None):
        data = a.data <=  b.data
        x = chainer.Variable(data)
        return self._cast_logic_output(a, x, dtype)

    def greater_equal(self, a, b, dtype=None):
        data = a.data >= b.data
        x = chainer.Variable(data)
        return self._cast_logic_output(a, x, dtype)

    def greater(self, a, b, dtype=None):
        data = a.data > b.data
        x = chainer.Variable(data)
        return self._cast_logic_output(a, x, dtype)


class ChainerMapAPI(BaseMapAPI):
    def clip(self, x, min=-np.inf, max=np.inf):
        return F.clip(x, float(min), float(max))

    def log(self, x):
        return F.math.exponential.log(x)

    def pow(self, x, a):
        return F.math.basic_math.pow(x, a)


class ChainerMetricAPI(BaseMetricAPI):
    pass


class ChainerReduceAPI(BaseReduceAPI):
    def argmax(self, x, axis=-1):
        return F.argmax(x, axis)

    def mean(self, x, axis=None, keepdims=False):
        if axis is None:
            denom = x.size
            x = F.math.sum.sum(x, axis, keepdims)
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
            x = F.math.sum.sum(x, axes, keepdims) / denom
        return x

    def sum(self, x, axis=None, keepdims=False):
        axis = tuple(axis) if isinstance(axis, list) else axis
        x = F.math.sum.sum(x, axis, keepdims)
        if not x.ndim:
            x = self.expand_dims(x, 0)
        return x


class ChainerRelateAPI(BaseRelateAPI):
    def dense(self, x, kernel, bias):
        return F.connection.linear.linear(x, kernel, bias)


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
        return F.array.expand_dims.expand_dims(x, axis)


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
            x = F.cast(x, to_dtype)
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


class ChainerBackend(BaseBackend, ChainerActivationAPI, ChainerDeviceAPI,
                     ChainerDataTypeAPI, ChainerDeviceDataTypeAPI,
                     ChainerLogicAPI, ChainerMapAPI, ChainerMetricAPI,
                     ChainerReduceAPI, ChainerRelateAPI, ChainerShapeAPI,
                     ChainerVariableAPI):
    def __init__(self):
        BaseBackend.__init__(self)
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
        self.name = 'chainer'
