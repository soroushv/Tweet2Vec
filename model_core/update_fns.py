import model_utilities as utils
__author__ = 'pralav'

from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T


def calculate_gradients(loss_gradients, params):
    for p in params:
        if not isinstance(p, theano.compile.SharedVariable):
            raise Exception("Shared Vars only!!")

    if isinstance(loss_gradients, list):
        if  len(loss_gradients) != len(params):
            raise Exception("Lenghts don't match %s - %s" %
                             (len(loss_gradients), len(params)))
        return loss_gradients
    else:
        return theano.grad(loss_gradients, params)


def adagrad(loass_gradient, params, learning_rate=1.0, epsilon=1e-6):

    grads = calculate_gradients(loass_gradient, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accumulator = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        new_accumulator = accumulator + grad ** 2
        updates[accumulator] = new_accumulator
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(new_accumulator + epsilon))

    return updates


def adadelta(loss_gradient, params, learning_rate=1.0, rho=0.95, epsilon=1e-6):

    grads = calculate_gradients(loss_gradient, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accumulator = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        delta_accumulator = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)

        new_accumulator = rho * accumulator + (1 - rho) * grad ** 2
        updates[accumulator] = new_accumulator

        change = (grad * T.sqrt(delta_accumulator + epsilon) /
                  T.sqrt(new_accumulator + epsilon))
        updates[param] = param - learning_rate * change

        delta_accu_new = rho * delta_accumulator + (1 - rho) * change ** 2
        updates[delta_accumulator] = delta_accu_new

    return updates


def adam(loss_gradient, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):

    all_grads = calculate_gradients(loss_gradient, params)
    t_prev = theano.shared(utils.float_arr(0.))
    updates = OrderedDict()

    t = t_prev + 1
    learning_rate_t = learning_rate * T.sqrt(1 - beta2 ** t)/(1 - beta1 ** t)

    for param, grad in zip(params, all_grads):
        value = param.get_value(borrow=True)
        m_t_1 = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        v_t_1 = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t = beta1 * m_t_1 + (1 - beta1) * grad
        v_t = beta2 * v_t_1 + (1 - beta2) * grad ** 2
        time_step = learning_rate_t * m_t / (T.sqrt(v_t) + epsilon)
        param_t = param - time_step

        updates[m_t_1] = m_t
        updates[v_t_1] = v_t
        updates[param] = param_t

    updates[t_prev] = t
    return updates



def norm_constraint(tensor_var, max_norm, norm_axes=None, epsilon=1e-7):

    ndim = tensor_var.ndim

    if norm_axes is not None:
        sum_over = tuple(norm_axes)
    elif ndim == 2:
        sum_over = (0,)
    elif ndim in [3, 4, 5]:
        sum_over = tuple(range(1, ndim))
    else:
        raise Exception("Not supported")

    dtype = np.dtype(theano.config.floatX).type
    norms = T.sqrt(T.sum(T.sqr(tensor_var), axis=sum_over, keepdims=True))
    target_norms = T.clip(norms, 0, dtype(max_norm))
    constrained_output = \
        (tensor_var * (target_norms / (dtype(epsilon) + norms)))

    return constrained_output


def ful_norm_const(tensor_vars, max_norm, epsilon=1e-7,
                          return_norm=False):

    norm = T.sqrt(sum(T.sum(tensor**2) for tensor in tensor_vars))
    dtype = np.dtype(theano.config.floatX).type
    target_norm = T.clip(norm, 0, dtype(max_norm))
    multiplier = target_norm / (dtype(epsilon) + norm)
    tensor_vars_scaled = [step*multiplier for step in tensor_vars]

    if return_norm:
        return tensor_vars_scaled, norm
    else:
        return tensor_vars_scaled