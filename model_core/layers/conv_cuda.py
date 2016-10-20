import theano.tensor as T

from model_core.initialization import *
from model_core.activations import *
from model_core.layers.base_layer import BLayer
from model_core.layers.conv_op import ConvOps
from model_core.model_utilities import cast_tuple, output_shape_conv, output_shape_pool, conv_input_length
from theano.tensor.signal.pool import pool_2d


class CustomConv(BLayer):
    def __init__(self, prev_layers, num_filters, filter_size, stride=1, padding=0, untie_biases=False,
                 W=Init(shape=None, init_type='glorot_uniform'), b=Init(init_type='constant', val=0.), activation=relu,
                 flip_filters=True, n=None, **kwargs):
        super(CustomConv, self).__init__(prev_layers, **kwargs)
        self.activation = relu if activation is None else activation

        if n is None:
            n = len(self.input_shape) - 2
        elif n != len(self.input_shape) - 2:
            raise Exception("Shape Issues")
        self.n = n
        self.num_filters = num_filters
        self.filter_size = cast_tuple(filter_size, n, int)
        self.flip_filters = flip_filters
        self.stride = cast_tuple(stride, n, int)
        self.untie_biases = untie_biases

        if padding == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise Exception('Immplementation padding same error')
        if padding == 'valid':
            self.pad = cast_tuple(0, n)
        elif padding in ('full', 'same'):
            self.pad = padding
        else:
            self.pad = cast_tuple(padding, n, int)

        self.W = self.set_conf_params(W, self.get_W_shape(), name="W")
        output_shape = self.get_output_shape()
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters,) + output_shape[2:]
            else:
                biases_shape = (num_filters,)
            self.b = self.set_conf_params(b, biases_shape, name="b",
                                          reg=False)

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]

        return (self.num_filters, num_input_channels) + self.filter_size

    def calc_output_shape(self, input_shape):
        padding = self.pad if isinstance(self.pad, tuple) else (self.pad,) * self.n
        batchsize = input_shape[0]
        res_shape = ()
        op_size = (batchsize, self.num_filters)
        d_size = ()
        for input, filter, stride, pad in zip(input_shape[2:], self.filter_size, self.stride, padding):
            d_size += (output_shape_conv(input, filter, stride, pad),)

        res_shape = (op_size + tuple(d_size))

        return res_shape

    def calc_output(self, input, **kwargs):
        conved = self.conv_op(input, **kwargs)
        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + T.shape_padleft(self.b, 1)
        else:
            activation = conved + self.b.dimshuffle(('x', 0) + ('x',) * self.n)
        return self.activation(activation)

    def conv_op(self, input, **kwargs):
        raise Exception("No implementation")


class Convolution1D(CustomConv):
    def __init__(self, prev_layers, num_filters, filter_size, stride=1,
                 padding=0, untie_biases=False,
                 W=Init(shape=None, init_type='glorot_uniform'), b=Init(init_type='constant', val=0.),
                 act=relu, flip_filters=True,
                 convolution=None, channel='multi', is_width=True, **kwargs):
        super(Convolution1D, self).__init__(prev_layers, num_filters, filter_size,
                                            stride, padding, untie_biases, W, b,
                                            act, flip_filters, n=1,
                                            **kwargs)
        if convolution is not None:
            if isinstance(convolution, ConvOps):
                self.convolution = convolution
            else:
                raise Exception("Convolution Object is not an instance of ConvOps, but: %s" % type(convolution))

        else:
            self.convolution = ConvOps(dim=1, channel=channel, is_width=is_width)

    def conv_op(self, input, **kwargs):
        border_mode = 'half' if self.pad == 'same' else self.pad
        conved = self.convolution.conv(input, self.W,
                                       self.input_shape, self.get_W_shape(),
                                       subsample=self.stride,
                                       border_mode=border_mode,
                                       filter_flip=self.flip_filters)
        return conved


class Conv2D(CustomConv):
    def __init__(self, prev_layers, num_filters, filter_size, stride=(1, 1),
                 padding=0, untie_biases=False,
                 W=Init(shape=None, init_type='glorot_uniform'), b=Init(init_type='constant', val=0.),
                 act=relu, flip_filters=True,
                 convolution=T.nnet.conv2d, **kwargs):
        super(Conv2D, self).__init__(prev_layers, num_filters, filter_size,
                                          stride, padding, untie_biases, W, b,
                                          act, flip_filters, n=2,
                                          **kwargs)
        if convolution is not None:
            if isinstance(convolution, ConvOps):
                self.convolution = convolution
            else:
                raise Exception("Convolution Object is not an instance of ConvOps, but: %s" % type(convolution))

        else:
            self.convolution = ConvOps(dim=2)

        self.convolution = convolution

    def conv_op(self, input, **kwargs):
        border_mode = 'half' if self.pad == 'same' else self.pad
        conved = self.convolution.conv(input, self.W,
                                       self.input_shape, self.get_W_shape(),
                                       subsample=self.stride,
                                       border_mode=border_mode,
                                       filter_flip=self.flip_filters)
        return conved


class Pool1D(BLayer):
    def __init__(self, incoming, pool_size, stride=None, padding=0,
                 ignore_border=True, pool_type='max', **kwargs):
        super(Pool1D, self).__init__(incoming, **kwargs)

        if len(self.input_shape) != 3:
            raise Exception("Less dimensions")

        self.pool_size = cast_tuple(pool_size, 1)
        self.stride = self.pool_size if stride is None else cast_tuple(stride, 1)
        self.padding = cast_tuple(padding, 1)
        self.ignore_border = ignore_border
        self.pool_type = pool_type

    def calc_output_shape(self, input_shape):
        output_shape = list(input_shape)

        output_shape[-1] = output_shape_pool(input_shape[-1], pool_size=self.pool_size[0], stride=self.stride[0],
                                             pad=self.padding[0], ignore_border=self.ignore_border)

        return tuple(output_shape)

    def calc_output(self, input, **kwargs):
        input_updated = T.shape_padright(input, 1)
        pooled = pool_2d(input_updated, ds=(self.pool_size[0], 1), st=(self.stride[0], 1),
                         ignore_border=self.ignore_border, padding=(self.padding[0], 0), mode=self.pool_type)
        return pooled[:, :, :, 0]


class Pool2D(BLayer):
    def __init__(self, incoming, pool_size, stride=None, padding=(0, 0), ignore_border=True, pool_type='max', **kwargs):
        super(Pool2D, self).__init__(incoming, **kwargs)

        self.pool_size = cast_tuple(pool_size, 2)

        if len(self.input_shape) != 4:
            raise Exception("Shape issues")

        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = cast_tuple(stride, 2)

        self.padding = cast_tuple(padding, 2)

        self.ignore_border = ignore_border
        self.pool_type = pool_type

    def calc_output_shape(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[2] = output_shape_pool(input_shape[2], pool_size=self.pool_size[0], stride=self.stride[0],
                                            pad=self.padding[0], ignore_border=self.ignore_border)

        output_shape[3] = output_shape_pool(input_shape[3], pool_size=self.pool_size[1], stride=self.stride[1],
                                            pad=self.padding[1], ignore_border=self.ignore_border)

        return tuple(output_shape)

    def calc_output(self, input, **kwargs):
        pooled = pool_2d(input, ds=self.pool_size, st=self.stride, ignore_border=self.ignore_border,
                         padding=self.padding, mode=self.pool_type)
        return pooled


class MaxPool1D(Pool1D):
    def __init__(self, prev_layer, pool_size, stride=None, padding=0,
                 ignore_border=True, **kwargs):
        super(MaxPool1D, self).__init__(prev_layer, pool_size, stride, padding, ignore_border, pool_type='max',
                                        **kwargs)


class MaxPool2D(Pool2D):
    def __init__(self, prev_layers, pool_size, stride=None, padding=(0, 0),
                 ignore_border=True, **kwargs):
        super(MaxPool2D, self).__init__(prev_layers, pool_size, stride, padding, ignore_border, pool_type='max',
                                        **kwargs)
