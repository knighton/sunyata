import numpy as np


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
    def minimum(self, a, b):
        raise NotImplementedError

    def maximum(self, a, b):
        raise NotImplementedError

    def equal(self, a, b, dtype=None):
        raise NotImplementedError

    def not_equal(self, a, b, dtype=None):
        raise NotImplementedError

    def less(self, a, b, dtype=None):
        raise NotImplementedError

    def less_equal(self, a, b, dtype=None):
        raise NotImplementedError

    def greater_equal(self, a, b, dtype=None):
        raise NotImplementedError

    def greater(self, a, b, dtype=None):
        raise NotImplementedError


class BaseMapAPI(APIBase):
    def clip(self, x, min=-np.inf, max=np.inf):
        raise NotImplementedError

    def log(self, x):
        raise NotImplementedError

    def pow(self, x, a):
        raise NotImplementedError

    def square(self, x):
        return self.pow(x, 2)

    def sqrt(self, x):
        return self.pow(x, 0.5)


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


class BaseReduceAPI(APIBase):
    def argmax(self, axis=-1):
        raise NotImplementedError

    def mean(self, x, axis=None, keepdims=False):
        raise NotImplementedError

    def sum(self, x, axis=None, keepdims=False):
        raise NotImplementedError


class BaseRelateAPI(APIBase):
    def dense(self, x, kernel, bias):
        raise NotImplementedError


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
        if dtype is None:
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

    def _cast_bool_output(self, input_arg, x, override_dtype):
        if override_dtype is None:
            to_dtype = self.dtype_of(input_arg)
        else:
            to_dtype = self.dtype(override_dtype)
        if self.dtype_of(x) != to_dtype:
            x = self.cast(x, to_dtype)
        return x

    def to(self, x, device=None, copy=False):
        return self.cast_to(x, None, device, copy)

    def to_cpu(self, x, copy=False):
        return self.to_device(x, 0, copy)

    def to_gpu(self, x, device, copy=False):
        device = self.to_device(device)
        assert device.is_gpu()
        return self.to_device(x, device, copy)


class BaseVariableAPI(APIBase):
    def constant(self, x):
        raise NotImplementedError

    def _variable_name(self, name=None):
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
                y_aux_scores.append(self.result_to_tensor(result))
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

    def incr(self, x, incr):
        self.assign(x, self.variable_to_tensor(x) + incr)

    def decr(self, x, decr):
        self.assign(x, self.variable_to_tensor(x) - decr)

    def numpy(self, x):
        raise NotImplementedError

    def list(self, x):
        return self.numpy(x).tolist()

    def scalar(self, x):
        assert self.size(x) == 1
        return self.numpy(x).flatten()[0]


class BaseBackend(BaseActivationAPI, BaseDeviceDataTypeAPI, BaseEpsilonAPI,
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

    def zeros_like(self, x):
        return self.cast_numpy_to(np.zeros(self.shape(x), self.dtype_of(x)))
