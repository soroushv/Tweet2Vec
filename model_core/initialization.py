
import numpy as np

from model_utilities import float_arr, rng


class Init(object):
    def __init__(self, init_type=None, **kwargs):

        self.init_type = init_type
        self.kwargs = kwargs

    def initialize(self, shape=None):

        if shape is None:
            raise Exception("Shape is None")
        else:
            self.shape = shape
        if self.init_type is None:
            raise Exception("Initializer is None")
        return getattr(self, self.init_type)(**self.kwargs)

    def normal(self, **kwargs):
        mean = kwargs.get("mean", 0.0)
        std = kwargs.get("std", 0.05)
        return float_arr(rng.normal(mean, std, size=self.shape))

    def uniform(self, **kwargs):
        range = kwargs.get("range", None)
        std = kwargs.get("std", None)
        if std is not None:
            mean = kwargs.get("mean", 0.)
            x = mean - np.sqrt(3) * std
            y = mean + np.sqrt(3) * std
        elif isinstance(range, tuple):
            x, y = range
        else:
            x, y = -range, range
        return float_arr(rng.uniform(low=x, high=y, size=self.shape))

    def _get_fan_in_out(self, **kwargs):
        cud = kwargs.get('cud', False)
        if len(self.shape) == 2:
            _in, _out = self.shape
        elif cud and len(self.shape) == 4:
            _in = np.prod(self.shape[:3])
            _out = np.prod(self.shape[1:])
        else:
            _in = (np.prod(self.shape[1:]))
            _out = self.shape[0] * (np.prod(self.shape[2:]))
        return _in, _out

    def glorot_uniform(self, **kwargs):
        cb=kwargs.get('conv',False)
        gain=kwargs.get('gain',1.0)
        if cb:
            if len(self.shape) != 4:
                raise Exception("Shape should be 4")
            n1, n2 = self.shape[0], self.shape[3]
            receptive_field_size = self.shape[1] * self.shape[2]
        else:
            if len(self.shape) < 2:
                raise Exception("only works with shapes of >2")

            n1, n2 = self.shape[:2]
            receptive_field_size = np.prod(self.shape[2:])

        std = gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))

        kwargs.setdefault('std', std)
        return self.uniform(**kwargs)

    def glorot_normal(self, **kwargs):
        cb=kwargs.get('conv',False)
        gain=kwargs.get('gain',1.0)
        if cb:
            if len(self.shape) != 4:
                raise Exception("Shape should be 4")
            n1, n2 = self.shape[0], self.shape[3]
            receptive_field_size = self.shape[1] * self.shape[2]
        else:
            if len(self.shape) < 2:
                raise Exception("only works with shapes of >2")

            n1, n2 = self.shape[:2]
            receptive_field_size = np.prod(self.shape[2:])

        std = gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))

        kwargs.setdefault('std', std)
        return self.normal(**kwargs)

    def lecun_uniform(self, **kwargs):
        fan_in, fan_out = self._get_fan_in_out(**kwargs)
        s = np.sqrt(3. / fan_in)
        kwargs.setdefault('range', (-s, s))
        return self.uniform(**kwargs)

    def he_uniform(self, **kwargs):
        fan_in, fan_out = self._get_fan_in_out(**kwargs)
        s = np.sqrt(6. / (fan_in))
        kwargs.setdefault('range', (-s, s))
        return self.uniform(**kwargs)

    def he_normal(self, **kwargs):

        fan_in, fan_out = self._get_fan_in_out(**kwargs)
        s = np.sqrt(2. / (fan_in))
        kwargs.setdefault('std', s)
        return self.normal(**kwargs)

    def constant(self, **kwargs):
        val = kwargs.get('val', 0.)
        return float_arr(np.ones(self.shape) * val)
