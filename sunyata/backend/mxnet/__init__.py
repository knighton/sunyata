import mxnet as mx
import numpy as np
import subprocess

from ..base import \
    BaseActivationAPI, BaseDataTypeAPI, BaseDeviceAPI, BaseDeviceDataTypeAPI, \
    BaseLogicAPI, BaseMapAPI, BaseMetricAPI, BaseReduceAPI, BaseRelateAPI, \
    BaseShapeAPI, BaseVariableAPI, BaseBackend


class MXNetActivationAPI(BaseActivationAPI):
    def softmax(self, x):
        return mx.nd.softmax(x)


class MXNetLogicAPI(BaseLogicAPI):
    def minimum(self, a, b):
        return mx.nd.broadcast_minimum(a, b)

    def maximum(self, a, b):
        return mx.nd.broadcast_maximum(a, b)

    def equal(self, a, b, dtype=None):
        x = mx.nd.broadcast_equal(a, b)
        return self._cast_bool_output(a, x, dtype)

    def not_equal(self, a, b, dtype=None):
        x = mx.nd.broadcast_not_equal(a, b)
        return self._cast_bool_output(a, x, dtype)

    def less(self, a, b, dtype=None):
        x = mx.nd.broadcast_less(a, b)
        return self._cast_bool_output(a, x, dtype)

    def less_equal(self, a, b, dtype=None):
        x = mx.nd.broadcast_less_equal(a, b)
        return self._cast_bool_output(a, x, dtype)

    def greater_equal(self, a, b, dtype=None):
        x = mx.nd.broadcast_greater_equal(a, b)
        return self._cast_bool_output(a, x, dtype)

    def greater(self, a, b, dtype=None):
        x = mx.nd.broadcast_greater(a, b)
        return self._cast_bool_output(a, x, dtype)


class MXNetMapAPI(BaseMapAPI):
    def abs(self, x):
        return mx.nd.abs(x)

    def neg(self, x):
        return mx.nd.neg(x)

    def sign(self, x):
        return mx.nd.sign(x)

    def clip(self, x, min=-np.inf, max=np.inf):
        return mx.nd.clip(x, min, max)

    def log(self, x):
        return mx.nd.log(x)

    def pow(self, x, a):
        return mx.nd.power(x, a)


class MXNetMetricAPI(BaseMetricAPI):
    pass


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


class MXNetRelateAPI(BaseRelateAPI):
    def dense(self, x, kernel, bias):
        return mx.nd.dot(x, kernel) + bias


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
                arr = np.ones((1,), self.dtype_of(y_true)) * judge.importance
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


class MXNetBackend(BaseBackend, MXNetActivationAPI, MXNetDataTypeAPI,
                   MXNetDeviceAPI, MXNetDeviceDataTypeAPI, MXNetLogicAPI,
                   MXNetMapAPI, MXNetMetricAPI, MXNetReduceAPI, MXNetRelateAPI,
                   MXNetShapeAPI, MXNetVariableAPI):
    def __init__(self):
        BaseBackend.__init__(self)
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
        self.name = 'mxnet'
