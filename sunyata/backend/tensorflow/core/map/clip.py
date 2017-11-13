import numpy as np
import tensorflow as tf

from ....base.core.map.clip import BaseClipAPI


class TensorFlowClipAPI(BaseClipAPI):
    def __init__(self):
        BaseClipAPI.__init__(self)

    def clip(self, x, min=-np.inf, max=np.inf):
        return tf.clip_by_value(x, min, max)
