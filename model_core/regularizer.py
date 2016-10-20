__author__ = 'pralav'

import theano.tensor as T

def l1(x):
    return T.sum(abs(x))

def l2(x):
    return T.sum((x ** 2))