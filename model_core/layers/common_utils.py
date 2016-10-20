__author__ = 'pralav'
import collections
from model_core.layers.base_layer import BLayer, Fuse
from model_core.layers.basic_layers import FC,Input
from model_core.model_utilities import cast_theano_expr, ordered_set



from inspect import getargspec
from warnings import warn

import numpy as np



def collect_all_layers(layer, layer_begin=None):
    all_layers = []
    layer_list=[]
    visited_node=set()
    completed = set()
    try:
        if isinstance(layer,collections.Iterable):
            layer_list=list(layer)
        else:
            layer_list=[layer]
    except TypeError:
        layer_list=[layer]

    if layer_begin is not None:
        visited_node=visited_node.union(layer_begin)

    while len(layer_list)>0:
        layer = layer_list[0]
        if layer is None:
            layer_list.pop(0)
        elif layer not in visited_node:
            if isinstance(layer, Fuse):
                layer_list=(layer.prev_layers)+layer_list
            elif isinstance(layer, BLayer):
                layer_list=[layer.prev_layer]+layer_list
            visited_node.add(layer)
        else:
            layer_list.pop(0)
            if layer not in completed:
                all_layers.append(layer)
                completed.add(layer)
    return all_layers


def calculate_output(layer_or_layers, inputs=None, **kwargs):

    layer_begin=[]

    if isinstance(inputs,dict):
        layer_begin=inputs.keys()

    layers=collect_all_layers(layer_or_layers,layer_begin=layer_begin)

    layer_expr_outputs=[]
    for layer in layers:
        if isinstance(layer,Input) and layer not in layer_begin:
            layer_expr_outputs.append((layer,layer.input_theano_sym))
    layer_expr_outputs=dict(layer_expr_outputs)

    if isinstance(inputs,dict):
        all_exprs=[]
        for layer,expr in inputs.iteritems():
            all_exprs.append((layer,cast_theano_expr(expr)))
        layer_expr_outputs.update(all_exprs)

    elif inputs is not None:
        if len(layer_expr_outputs)>1:
            raise Exception("Called with one input expression with many input layers")
        for layer in layer_expr_outputs:
            layer_expr_outputs[layer]=cast_theano_expr(inputs)
    kwargs_all = {'deterministic'}
    for layer in layers:
        if layer not in layer_expr_outputs:
            try:
                if isinstance(layer, Fuse):
                    inputs_l = [layer_expr_outputs[input_layer]
                                    for input_layer in layer.prev_layers]
                elif isinstance(layer, BLayer):
                    inputs_l = layer_expr_outputs[layer.prev_layer]
                else:
                    inputs_l=[]
            except KeyError:
                raise Exception("No input expr given for layer")

            layer_expr_outputs[layer]=layer.calc_output(inputs_l,**kwargs)
            try:
                names, _, _, defaults = getargspec(layer.calc_output)
            except TypeError:
                pass
            else:
                if defaults is not None:
                    kwargs_all=kwargs_all.union(set(names[-len(defaults):]))
            kwargs_all = kwargs_all.union(set(layer.get_output_kwargs))
    unused_kwargs = set(kwargs.keys()) - kwargs_all
    if unused_kwargs:
        warn("calc_output was called with unused kwargs:\n\t%s"% "\n\t".join(unused_kwargs))
    try:
        return [layer_expr_outputs[layer] for layer in layer_or_layers]
    except TypeError:
        return layer_expr_outputs[layer_or_layers]



def calculate_output_shape(layer_or_layers, inputs_shape=None):

    layer_begin=[]

    if isinstance(inputs_shape,dict):
        layer_begin=inputs_shape.keys()
    elif inputs_shape is None or len(inputs_shape)==0:
        if  isinstance(inputs_shape, collections.Iterable):
            return [layer.get_output_shape() for layer in layer_or_layers]
        else:
            return layer_or_layers.get_output_shape()


    layers=collect_all_layers(layer_or_layers,layer_begin=layer_begin)

    layer_expr_shapes=[]
    for layer in layers:
        if isinstance(layer,Input) and layer not in layer_begin:
            layer_expr_shapes.append((layer,layer.input_theano_sym))
    layer_expr_shapes=dict(layer_expr_shapes)

    if isinstance(inputs_shape,dict):

        layer_expr_shapes.update(inputs_shape)

    elif inputs_shape is not None:
        if len(layer_expr_shapes)>1:
            raise Exception("Called with one input expression with many input layers")
        for layer in layer_expr_shapes:
            layer_expr_shapes[layer]=inputs_shape

    for layer in layers:
        if layer not in layer_expr_shapes:

            if isinstance(layer, Fuse):
                input_shape = [layer_expr_shapes[input_layer]
                                for input_layer in layer.prev_layers]
            else:
                input_shape = layer_expr_shapes[layer.prev_layer]


            layer_expr_shapes[layer]=layer.calc_output_shape(input_shape)

    try:
        return [layer_expr_shapes[layer] for layer in layer_or_layers]
    except TypeError:
        return layer_expr_shapes[layer_or_layers]


def extend_iters(iterables):
    for it in iterables:
        for element in it:
            yield element

def collect_all_conf_params(layer, return_shared_expr=True,**kwargs):
    layers = collect_all_layers(layer)
    all_conf_params=[layer.get_conf_params(return_shared_expr=return_shared_expr,**kwargs) for layer in layers]
    all_params=extend_iters(all_conf_params)
    return ordered_set(all_params)


def count_params(layer, **kwargs):
    all_conf_params = collect_all_conf_params(layer, **kwargs)
    shapes = [param.get_value().shape for param in all_conf_params]
    total_sum=0
    for shape in shapes:
        total_sum+=np.prod(shape)
    return total_sum


def get_conf_param_values(layer, return_shared_expr=True, **kwargs):
    all_params = collect_all_conf_params(layer,return_shared_expr=return_shared_expr, **kwargs)
    return [param.get_value() for param in all_params]


def set_conf_param_values(layer, values, **kwargs):
    all_params = collect_all_conf_params(layer, **kwargs)
    if len(all_params) != len(values):
        raise Exception("No. of Params error")

    for param, value in zip(all_params, values):
        if param.get_value().shape != value.shape:
            raise Exception("Param shape error")
        else:
            param.set_value(value)

if __name__ == '__main__':

    l_in = InputLayer((100, 20))
    l1 = FC(l_in, out_dim=50)
    param_count = count_params(l1)
    print param_count
    # 1050
    print param_count == 20 * 50 + 50  # 20 input * 50 units + 50 biases
    l1 = FC(l_in, out_dim=50)
    all_param_values = get_conf_param_values(l1)
    print (all_param_values[0] == l1.W.get_value()).all()
    print (all_param_values[1] == l1.b.get_value()).all()

    l_in = InputLayer((100, 20))
    l1 = FC(l_in, out_dim=50)
    all_param_values = get_conf_param_values(l1)
    set_conf_param_values(l1, all_param_values)
