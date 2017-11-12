import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from sunyata.backend.base import \
    BaseActivationAPI, BaseDataTypeAPI, BaseDeviceAPI, BaseDeviceDataTypeAPI, \
    BaseLogicAPI, BaseMapAPI, BaseMetricAPI, BaseReduceAPI, BaseRelateAPI, \
    BaseShapeAPI, BaseVariableAPI, BaseBackend


tfe.enable_eager_execution()


class TensorFlowActivationAPI(BaseActivationAPI):
    def softmax(self, x):
        return tf.nn.softmax(x)


class TensorFlowLogicAPI(BaseLogicAPI):
    def minimum(self, a, b):
        return tf.minimum(a, b)

    def maximum(self, a, b):
        return tf.maximum(a, b)

    def equal(self, a, b, dtype=None):
        x = tf.equal(a, b)
        return self._cast_bool_output(a, x, dtype)

    def not_equal(self, a, b, dtype=None):
        x = tf.not_equal(a, b)
        return self._cast_bool_output(a, x, dtype)

    def less(self, a, b, dtype=None):
        x = tf.less(a, b)
        return self._cast_bool_output(a, x, dtype)

    def less_equal(self, a, b, dtype=None):
        x = tf.less_equal(a, b)
        return self._cast_bool_output(a, x, dtype)

    def greater_equal(self, a, b, dtype=None):
        x = tf.greater_equal(a, b)
        return self._cast_bool_output(a, x, dtype)

    def greater(self, a, b, dtype=None):
        x = tf.greater(a, b)
        return self._cast_bool_output(a, x, dtype)


class TensorFlowMapAPI(BaseMapAPI):
    def abs(self, x):
        return tf.abs(x)

    def neg(self, x):
        return tf.negative(x)

    def sign(self, x):
        return tf.sign(x)

    def clip(self, x, min=-np.inf, max=np.inf):
        return tf.clip_by_value(x, min, max)

    def log(self, x):
        return tf.log(x)

    def pow(self, x, a):
        return tf.pow(x, a)


class TensorFlowMetricAPI(BaseMetricAPI):
    pass


class TensorFlowReduceAPI(BaseReduceAPI):
    def argmax(self, x, axis=-1):
        return tf.argmax(x, axis)

    def mean(self, x, axis=None, keepdims=False):
        return tf.reduce_mean(x, axis, keepdims)

    def sum(self, x, axis=None, keepdims=False):
        return tf.reduce_sum(x, axis, keepdims)


class TensorFlowRelateAPI(BaseRelateAPI):
    def dense(self, x, kernel, bias):
        return tf.matmul(x, kernel) + bias


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
        to_dtype = x.dtype.name if dtype is None else self.dtype(dtype)
        with tf.device(self._device_name(to_device)):
            return tf.convert_to_tensor(x, to_dtype)


class TensorFlowVariableAPI(BaseVariableAPI):
    def constant(self, x):
        return tf.constant(x)

    def variable(self, x):
        return tfe.Variable(x, name=self._variable_name())

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


class TensorFlowBackend(BaseBackend, TensorFlowActivationAPI,
                        TensorFlowDataTypeAPI, TensorFlowDeviceAPI,
                        TensorFlowDeviceDataTypeAPI, TensorFlowLogicAPI,
                        TensorFlowMapAPI, TensorFlowMetricAPI,
                        TensorFlowReduceAPI, TensorFlowRelateAPI,
                        TensorFlowShapeAPI, TensorFlowVariableAPI):
    def __init__(self):
        BaseBackend.__init__(self)
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
        self.name = 'tensorflow'
