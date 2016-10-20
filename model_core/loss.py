from keras.layers import Dense

__author__ = 'pralav'


import theano.tensor as T


def binary_crossentropy(predictions, targets):
    return T.nnet.binary_crossentropy(predictions, targets)


def categorical_crossentropy(predictions, targets):
    return T.nnet.categorical_crossentropy(predictions, targets)


def mse(targets, predictions):
    return T.mean(T.square(predictions - targets), axis=-1)


def mae(targets, predictions):
    return T.mean(T.abs_(predictions - targets), axis=-1)

def aggregate(loss, weights=None, mode='mean'):

    if weights is not None:
        loss = loss * weights
    if mode == 'mean':
        return loss.mean()
    elif mode == 'sum':
        return loss.sum()
    elif mode == 'normalized_sum':
        if weights is None:
            raise ValueError("require weights for mode='normalized_sum'")
        return loss.sum() / weights.sum()
    else:
        raise ValueError("mode must be 'mean', 'sum' or 'normalized_sum', "
                         "got %r" % mode)


def binary_hinge_loss(predictions, targets, binary=True, delta=1):

    if binary:
        targets = 2 * targets - 1
    return T.nnet.relu(delta - predictions * targets)




def binary_acc(predictions, targets, threshold=0.5):
    predictions = T.ge(predictions, threshold)
    return T.eq(predictions, targets)


def categorical_acc(predictions, targets, top_k=1):

    if targets.ndim == predictions.ndim:
        targets = T.argmax(targets, axis=-1)
    elif targets.ndim != predictions.ndim - 1:
        raise TypeError('rank mismatch between targets and predictions')

    if top_k == 1:
        # standard categorical accuracy
        top = T.argmax(predictions, axis=-1)
        return T.eq(top, targets)
    else:
        # top-k accuracy
        top = T.argsort(predictions, axis=-1)
        # (Theano cannot index with [..., -top_k:], we need to simulate that)
        top = top[[slice(None) for _ in range(top.ndim - 1)] +
                  [slice(-top_k, None)]]
        targets = T.shape_padaxis(targets, axis=-1)
        return T.any(T.eq(top, targets), axis=-1)