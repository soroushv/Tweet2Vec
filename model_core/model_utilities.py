
__author__ = 'pralav'
import numpy as np

import theano
import theano.tensor as T

rng=np.random

def set_random(latest_rng):
    global rng
    rng=latest_rng

def float_arr(arr):
    return np.asarray(arr, dtype=theano.config.floatX)



def empty_shared_var(dim=2, dtype=None):

    if dtype is None:
        dtype = theano.config.floatX

    shp = tuple([1] * dim)
    return theano.shared(np.zeros(shp, dtype=dtype))


def cast_theano_expr(inp):

    if isinstance(inp, theano.gof.Variable):
        return inp
    else:
        try:
            return theano.tensor.constant(inp)
        except Exception as e:
            raise Exception("Not able to cast as expr: %s)" % (type(inp)))


def get_shared_vars(theano_expr):

    if isinstance(theano_expr, theano.Variable):
        theano_expr = [theano_expr]
    return [inp for inp in theano.gof.graph.inputs(reversed(theano_expr))
            if isinstance(inp, theano.compile.SharedVariable)]


def one_hot(a, l=None):
    if l is None:
        l = T.cast(T.max(a) + 1, 'int32')

    return T.eye(l)[T.cast(a, 'int32')]

def ordered_set(arr):
    orderedset = []
    exists = {}
    for a in arr:
        if a not in exists:
            orderedset.append(a)
            exists[(a)]=1

    return orderedset


def cast_tuple(x, N, t=None):
    try:
        X = tuple(x)
    except TypeError:
        X = (x,) * N

    if (t is not None):
        X=tuple(map(t,X))
        # for i,v in enumerate(X):
        #     print i,t(v)
        #     X[i]=t(v)

    if len(X) != N:
        raise Exception("Wrong length %d, Expected: %d"%(len(X),N))

    return X


def compute_norms(array, axis=1):# norm_axes=None):

    if not isinstance(array, theano.Variable) and \
            not isinstance(array, np.ndarray):
        raise RuntimeError(
            "Unsupported type {}. "
            "Only theano variables and numpy arrays "
            "are supported".format(type(array))
        )

    # Compute default axes to sum over
    ndim = array.ndim
    if axis is not None:
        sum_over = tuple(axis)
    elif ndim == 1:          # For Biases that are in 1d (e.g. b of DenseLayer)
        sum_over = ()
    elif ndim == 2:          # DenseLayer
        sum_over = (0,)
    elif ndim in [3, 4, 5]:  # Conv{1,2,3}DLayer
        sum_over = tuple(range(1, ndim))
    else:
        raise Exception(
            "Unsupported tensor dimensionality {}. "
            "Must specify `norm_axes`".format(array.ndim)
        )

    if isinstance(array, theano.Variable):
        if len(sum_over) == 0:
            norms = T.abs_(array)
        else:
            norms = T.sqrt(T.sum(array**2, axis=sum_over))
    elif isinstance(array, np.ndarray):
        if len(sum_over) == 0:
            norms = abs(array)
        else:
            norms = np.sqrt(np.sum(array**2, axis=sum_over))

    return norms

def conv_input_length(output_length, filter_size, stride, pad=0):

    if output_length is None:
        return None
    if pad == 'valid':
        pad = 0
    elif pad == 'full':
        pad = filter_size - 1
    elif pad == 'same':
        pad = filter_size // 2
    if not isinstance(pad, int):
        raise Exception('Invalid pad: {0}'.format(pad))
    return (output_length - 1) * stride - 2 * pad + filter_size


def set_init_layer_param(spec, shape, name=None):

    shape = tuple(shape)
    if isinstance(spec, theano.Variable):
        if spec.ndim != len(shape):
            raise Exception("Should be %d !! Param  has %d dimensions." % (len(shape), spec.ndim))
        return spec

    elif isinstance(spec, np.ndarray):
        if spec.shape != shape:
            raise RuntimeError("Should be %s !! Parameter has shape %s" % (shape, spec.shape))
        return theano.shared(spec, name=name)

    elif hasattr(spec,'initialize'):#isinstance(spec,Init):#hasattr(spec, '__call__'):

        arr = spec.initialize(shape)
        try:
            arr = float_arr(arr)
        except Exception,e:
            raise e
        if arr.shape != shape:
            raise Exception("Wrong shape : %s vs %s"%(arr.shape,shape))
        return theano.shared(arr, name=name)

    else:
        raise Exception("Spec is of type : %s . Undefined Action for this type"%type(spec))

def output_shape_pool(input_length, pool_size, stride, pad, ignore_border):

    if input_length is None or pool_size is None:
        return None

    if ignore_border:
        output_length = input_length + 2 * pad - pool_size + 1
        output_length = (output_length + stride - 1) // stride

    else:
        assert pad == 0
        if stride >= pool_size:
            output_length = (input_length + stride - 1) // stride
        else:
            output_length = max(
                0, (input_length - pool_size + stride - 1) // stride) + 1

    return output_length

def output_shape_conv(input_length, filter_size, stride, pad=0):
    if input_length is None:
        return None
    if pad == 'valid':
        output_length = input_length - filter_size + 1
    elif pad == 'full':
        output_length = input_length + filter_size - 1
    elif pad == 'same':
        output_length = input_length
    elif isinstance(pad, int):
        output_length = input_length + 2 * pad - filter_size + 1
    else:
        raise Exception('Invalid pad: {0}'.format(pad))
    output_length = (output_length + stride - 1) // stride
    return output_length