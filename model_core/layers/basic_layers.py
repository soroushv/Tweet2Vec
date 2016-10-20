from model_core import activations
from model_core.activations import linear, relu
from model_core.initialization import Init
from model_core.layers.base_layer import  BLayer

__author__ = 'pralav'
import numpy as np
import theano.tensor as T

import theano



class FC(BLayer):
    def __init__(self, prev_layer, out_dim, W = Init(shape=None,init_type='glorot_uniform'), b=Init(init_type='constant',val=0.), act=relu, **kwargs):
        super(FC, self).__init__(prev_layer, **kwargs)
        self.activation = (linear if act is None else act)
        self.out_dim = out_dim
        input_dim = int(np.prod(self.input_shape[1:]))
        self.W = self.set_conf_params(W, (input_dim, out_dim), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.set_conf_params(b, (out_dim,), name="b",reg=False)


    def calc_output_shape(self, input_shape):
        return (input_shape[0], self.out_dim)

    def calc_output(self, input, **kwargs):
        if input.ndim > 2:
            input = input.flatten(2)
        out = T.dot(input, self.W)
        if self.b is not None:
            out = out + self.b.dimshuffle('x', 0)
        return self.activation(out)

class VectorEmbedding(BLayer):

    def __init__(self, prev_layer, input_size, output_size,
                 W=Init(init_type='normal'), **kwargs):
        super(VectorEmbedding, self).__init__(prev_layer, **kwargs)

        self.input_size = input_size
        self.output_size = output_size
        self.W = self.set_conf_params(W, (input_size, output_size), name="W")

    def calc_output_shape(self, input_shape):
        return input_shape + (self.output_size, )

    def calc_output(self, input, **kwargs):
        return self.W[input]


class Input(BLayer):

    def __init__(self, shape, theano_sym=None, name=None, **kwargs):
        super(Input, self).__init__(shape,name)
        ndim = len(shape)
        if theano_sym is None:
            theano_sym_type = T.TensorType(theano.config.floatX, [False] * ndim)
            var_name = ("%s.ip" % name) if name is not None else "ip"
            theano_sym = theano_sym_type(var_name)
        else:
            if theano_sym.ndim != ndim:
                raise Exception("Dim Mismatch %s %s" % (ndim, theano_sym.ndim))
        self.input_theano_sym = theano_sym


class Slice(BLayer):

    def __init__(self, prev_layer, idxs, axis=-1, **kwargs):
        super(Slice, self).__init__(prev_layer, **kwargs)
        self.slice = idxs
        self.axis = axis

    def calc_output_shape(self, input_shape):
        output_shape = list(input_shape)
        if isinstance(self.slice, int):
            del output_shape[self.axis]
        elif input_shape[self.axis] is not None:
            output_shape[self.axis] = len(
                range(*self.slice.indices(input_shape[self.axis])))
        else:
            output_shape[self.axis] = None
        return tuple(output_shape)

    def calc_output(self, input, **kwargs):
        axis = self.axis
        if axis < 0:
            axis += input.ndim
        return input[(slice(None),) * axis + (self.slice,)]


class Reshape(BLayer):
    def __init__(self, incoming, shape, **kwargs):
        super(Reshape, self).__init__(incoming, **kwargs)
        shape = tuple(shape)
        for s in shape:
            if isinstance(s, int):
                if s == 0 or s < - 1:
                    raise Exception("Rehshape errors")
            elif isinstance(s, list):
                if len(s) != 1 or not isinstance(s[0], int) or s[0] < 0:
                    raise Exception("Rehshape errors")
            elif isinstance(s, T.TensorVariable):
                if s.ndim != 0:
                    raise Exception(
                        "A symbolic variable in a shape specification must be "
                        "a scalar, but had %i dimensions" % s.ndim)
            else:
                raise Exception("shape proper req")
        if sum(s == -1 for s in shape) > 1:
            raise Exception("shape cannot -1")
        self.shape = shape
        self.calc_output_shape(self.input_shape)

    def calc_output_shape(self, input_shape):
        output_shape = list(self.shape)

        masked_input_shape = list(input_shape)
        masked_output_shape = list(output_shape)
        for dim, o in enumerate(output_shape):
            if isinstance(o, list):
                if o[0] >= len(input_shape):
                    raise Exception("specification contains [%d], but input "
                                     "shape has %d dimensions only" %
                                     (o[0], len(input_shape)))
                output_shape[dim] = input_shape[o[0]]
                masked_output_shape[dim] = input_shape[o[0]]
                if (input_shape[o[0]] is None) \
                        and (masked_input_shape[o[0]] is None):

                    masked_input_shape[o[0]] = 1
                    masked_output_shape[dim] = 1

        for dim, o in enumerate(output_shape):
            if isinstance(o, T.TensorVariable):
                output_shape[dim] = None
                masked_output_shape[dim] = None
        input_size = (None if any(x is None for x in masked_input_shape)
                      else np.prod(masked_input_shape))
        output_size = (None if any(x is None for x in masked_output_shape)
                       else np.prod(masked_output_shape))
        del masked_input_shape, masked_output_shape
        if -1 in output_shape:
            dim = output_shape.index(-1)
            if (input_size is None) or (output_size is None):
                output_shape[dim] = None
                output_size = None
            else:
                output_size *= -1
                output_shape[dim] = input_size // output_size
                output_size *= output_shape[dim]
        if (input_size is not None) and (output_size is not None) \
                and (input_size != output_size):
            raise Exception("Size mismatch")
        return tuple(output_shape)

    def calc_output(self, input, **kwargs):
        output_shape = list(self.shape)
        for dim, o in enumerate(output_shape):
            if isinstance(o, list):
                output_shape[dim] = input.shape[o[0]]
        return input.reshape(tuple(output_shape))


class Repeat(BLayer):
    def __init__(self, incoming, n, **kwargs):

        super(Repeat, self).__init__(incoming, **kwargs)
        self.n = n

    def calc_output_shape(self, input_shape):
        return tuple([input_shape[0], self.n] + list(input_shape[1:]))

    def calc_output(self, input, **kwargs):
        #repeat the input n times
        tensors = [input]*self.n
        stacked = T.stack(*tensors)
        dim = [1, 0] + range(2, input.ndim + 1)
        return stacked.dimshuffle(dim)
