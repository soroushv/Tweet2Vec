from model_core.initialization import Init

__author__ = 'pralav'
from collections import OrderedDict

import model_core.model_utilities as utils

import theano
import theano.tensor as T

class BaseLayer(object):
    def __init__(self, name=None):

        self.layer_name = name
        self.layer_params = OrderedDict()
        self.get_output_kwargs = []

    def calc_output_shape(self, input_shape):
        return input_shape

    def calc_output(self, input, **kwargs):
        raise Exception("Base Class")

    def set_conf_params(self, spec, shape, name=None, **tags):
        if self.layer_name is not None and name is not None:
            name = "%s.%s" % (self.layer_name, name)
        param = utils.set_init_layer_param(spec, shape, name)
        tags['train'] = tags.get('train', True)
        tags['reg'] = tags.get('reg', True)
        self.layer_params[param] = set(tag for tag, value in tags.items() if value)
        return param

    def get_conf_params(self, return_shared_expr=True,**kwargs):

        result = list(self.layer_params.keys())

        only = set(tag for tag, value in kwargs.items() if value)
        if only:
            result = [param for param in result
                      if not (only.difference(self.layer_params[param]))]

        exclude = set(tag for tag, value in kwargs.items() if not value)
        if exclude:
            result = [param for param in result
                      if not (exclude.intersection(self.layer_params[param]))]

        if return_shared_expr:
            return utils.get_shared_vars(result)
        else:
            return result



class BLayer(BaseLayer):
    def __init__(self, prev_layer,  name=None):
        super(BLayer,self).__init__(name)
        if isinstance(prev_layer, tuple):
            self.input_shape = prev_layer
            self.prev_layer = None

        else:
            self.input_shape = prev_layer.get_output_shape()# calc_output_shape(prev_layer.input_shape)
            self.prev_layer = prev_layer



        if any(d is not None and d <= 0 for d in self.input_shape):
            raise Exception("Wrong input shape ")

    def get_output_shape(self):
        return self.calc_output_shape(self.input_shape)


class Fuse(BaseLayer):
    def __init__(self, prev_layers, name=None):
        self.input_shapes = [prev_layer if isinstance(prev_layer, tuple)
                             else prev_layer.get_output_shape()#output_shape
                             for prev_layer in prev_layers]
        self.prev_layers = [None if isinstance(prev_layer, tuple)
                             else prev_layer
                             for prev_layer in prev_layers]

        super(Fuse,self).__init__(name)

    def calc_output_shape(self, input_shape):
        raise Exception("Base Fuse Class")

    def calc_output(self, inputs, **kwargs):
        raise Exception("Base Fuse Class")

    def get_output_shape(self):
        return self.calc_output_shape(self.input_shapes)


