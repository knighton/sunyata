import sys

from .base import Initializer
from .constant import *  # noqa
from .eye import *  # noqa
from .normal import *  # noqa
from .ones import *  # noqa
from .orthogonal import *  # noqa
from .smart_scaler import *  # noqa
from .trunc_normal import *  # noqa
from .uniform import *  # noqa
from .zeros import *  # noqa


def get(x):
    if isinstance(x, Initializer):
        pass
    elif isinstance(x, str):
        mod = sys.modules[__name__]
        x = getattr(mod, x)()
    else:
        assert False
    return x
