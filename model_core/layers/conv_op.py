__author__ = 'pralav'

import numpy as np
import theano.tensor as T

SINGLE=0
MULTI=1

class ConvOps(object):
    def __init__(self,dim=1,channel=SINGLE,is_width=True):
        self.channel=channel
        self.is_width=is_width
        self.dim=dim


    def conv(self,input, filters, image_shape=None, filter_shape=None,
             border_mode='valid', subsample=(1,), filter_flip=True):
        if self.dim==1:
            if self.channel==SINGLE:
                return self.single_conv1d(input, filters, image_shape=image_shape, filter_shape=filter_shape,
                                   border_mode=border_mode, subsample=subsample, filter_flip=filter_flip)
            elif self.channel==MULTI:
                return self.multi_conv1d(input, filters, image_shape=image_shape, filter_shape=filter_shape,
                                  border_mode=border_mode, subsample=subsample, filter_flip=filter_flip,is_width=self.is_width)
        elif self.dim==2:
            return T.nnet.conv2d(input, filters=filters,image_shape=image_shape,filter_shape= filter_shape,subsample=subsample,border_mode=border_mode,filter_flip=filter_flip)

    def single_conv1d(self,input, filters, image_shape=None, filter_shape=None,
                      border_mode='valid', subsample=(1,), filter_flip=True):
        if border_mode not in ('valid', 0, (0,)):
            raise RuntimeError("Unsupported border_mode for conv1d_sc: "
                               "%s" % border_mode)

        if image_shape is None:
            image_shape_sc = None
        else:
            image_shape_sc = (image_shape[0], 1, image_shape[1], image_shape[2])

        if filter_shape is None:
            filter_shape_sc = None
        else:
            filter_shape_sc = (filter_shape[0], 1, filter_shape[1],
                               filter_shape[2])

        input_sc = input.dimshuffle(0, 'x', 1, 2)
        filters_sc = filters.dimshuffle(0, 'x', 1, 2)[:, :, ::-1, :]

        conved = T.nnet.conv2d(input_sc, filters_sc, image_shape_sc,
                               filter_shape_sc, subsample=(1, subsample[0]),
                               filter_flip=filter_flip)
        return conved[:, :, 0, :]

    def multi_conv1d(self,input, filters, image_shape=None, filter_shape=None,
                   border_mode='valid', subsample=(1,), filter_flip=True,is_width=False):


        if image_shape is None:
            image_shape_mc0 = None
        else:
            image_shape_mc0 = (image_shape[0], image_shape[1], 1, image_shape[2]) if is_width else (image_shape[0], image_shape[1], image_shape[2], 1)


        if filter_shape is None:
            filter_shape_mc0 = None
        else:
            filter_shape_mc0 = (filter_shape[0], filter_shape[1], 1, filter_shape[2]) if is_width else (filter_shape[0], filter_shape[1], filter_shape[2], 1)


        if isinstance(border_mode, tuple):
            (border_mode,) = border_mode
        if isinstance(border_mode, int):
            border_mode = (0, border_mode) if is_width else (border_mode, 0)

        input_mc0 = input.dimshuffle(0, 1, 'x', 2) if is_width else input.dimshuffle(0, 1, 2, 'x')
        filters_mc0 = filters.dimshuffle(0, 1, 'x', 2) if is_width else filters.dimshuffle(0, 1, 2, 'x')

        conved = T.nnet.conv2d(input_mc0, filters_mc0, image_shape_mc0, filter_shape_mc0, subsample=(1, subsample[0]),
                               border_mode=border_mode, filter_flip=filter_flip) if is_width else \
                 T.nnet.conv2d(input_mc0, filters_mc0, image_shape_mc0, filter_shape_mc0, subsample=(subsample[0], 1),
                               border_mode=border_mode, filter_flip=filter_flip)
        return conved[:, :, 0, :] if is_width else conved[:, :, :, 0] # drop the unused dimension





