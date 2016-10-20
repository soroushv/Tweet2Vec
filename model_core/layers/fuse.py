__author__ = 'pralav'
from theano.tensor.shared_randomstreams import RandomStreams
from model_core.layers.base_layer import Fuse, BLayer
import theano.tensor as T
import numpy as np
_rng = np.random

class FuseLayer(Fuse):
    def __init__(self, prev_layers, axis=1, cropping=None, **kwargs):
        super(FuseLayer, self).__init__(prev_layers, **kwargs)
        self.axis = axis
        if cropping is not None:
            cropping = list(cropping)
            cropping[axis] = None
        self.cropping = cropping

    def calc_output_shape(self, input_shapes):
        input_shapes = fix_dims_array_shapes(input_shapes, self.cropping)
        output_shape = [next((s for s in sizes if s is not None), None)
                        for sizes in zip(*input_shapes)]

        def match(shape1, shape2):
            axis = self.axis if self.axis >= 0 else len(shape1) + self.axis
            return (len(shape1) == len(shape2) and
                    all(i == axis or s1 is None or s2 is None or s1 == s2
                        for i, (s1, s2) in enumerate(zip(shape1, shape2))))
        if not all(match(shape, output_shape) for shape in input_shapes):
            raise ValueError("Mismatch: input shapes must be the same except "
                             "in the concatenation axis")
        sizes = [input_shape[self.axis] for input_shape in input_shapes]
        concat_size = None if any(s is None for s in sizes) else sum(sizes)
        output_shape[self.axis] = concat_size
        return tuple(output_shape)

    def calc_output(self, inputs, **kwargs):
        inputs = fix_dims(inputs, self.cropping)
        return T.concatenate(inputs, axis=self.axis)

class DotFuse(Fuse):


    def __init__(self, incomings, merge_function, cropping=None, **kwargs):
        super(DotFuse, self).__init__(incomings, **kwargs)
        self.merge_function = merge_function
        self.cropping = cropping

    def calc_output_shape(self, input_shape):
        input_shapes = fix_dims_array_shapes(input_shape, self.cropping)
        # Infer the output shape by grabbing, for each axis, the first
        # input size that is not `None` (if there is any)
        output_shape = tuple(next((s for s in sizes if s is not None), None)
                             for sizes in zip(*input_shapes))

        def match(shape1, shape2):
            return (len(shape1) == len(shape2) and
                    all(s1 is None or s2 is None or s1 == s2
                        for s1, s2 in zip(shape1, shape2)))

        # Check for compatibility with inferred output shape
        if not all(match(shape, output_shape) for shape in input_shapes):
            raise ValueError("Mismatch: not all input shapes are the same")
        return output_shape

    def calc_output(self, inputs, **kwargs):
        inputs = fix_dims(inputs, self.cropping)
        output = None
        for input in inputs:
            if output is not None:
                output = self.merge_function(output, input)
            else:
                output = input
        return output


class DotSum(DotFuse):

    def __init__(self, incomings, coeffs=1, cropping=None, **kwargs):
        super(DotSum, self).__init__(incomings, T.add,
                                               cropping=cropping, **kwargs)
        if isinstance(coeffs, list):
            if len(coeffs) != len(incomings):
                raise ValueError("Mismatch: got %d coeffs for %d incomings" %
                                 (len(coeffs), len(incomings)))
        else:
            coeffs = [coeffs] * len(incomings)

        self.coeffs = coeffs

    def calc_output(self, inputs, **kwargs):
        inputs = [input * coeff if coeff != 1 else input
                  for coeff, input in zip(self.coeffs, inputs)]

        return super(DotSum, self).calc_output(inputs, **kwargs)


def fix_dims(inputs, dim_fix):

    if dim_fix is None:
        return inputs
    else:
        dims = inputs[0].ndim
        if not all(input.ndim == dims for input in inputs):
            raise Exception("Not all inputs are of the same dimensionality")

        shapes = [input.shape for input in inputs]
        shapes_tensor = T.as_tensor_variable(shapes)
        min_shape = T.min(shapes_tensor, axis=0)

        slices_by_input = [[] for i in range(len(inputs))]
        dim_fix = list(dim_fix)
        if dims > len(dim_fix):
            diff=dims-len(dim_fix)
            dim_fix = list(dim_fix) + [None] * diff

        for dim, fx in enumerate(dim_fix):
            if fx is None:
                slice_all = slice(None)
                for slices in slices_by_input:
                    slices.append(slice_all)
            else:
                size_sh = min_shape[dim]
                if fx == 'lower':
                    slc_lower = slice(None, size_sh)
                    for slices in slices_by_input:
                        slices.append(slc_lower)
                elif fx == 'upper':
                    slc_upper = slice(-size_sh, None)
                    for slices in slices_by_input:
                        slices.append(slc_upper)
                elif fx == 'center':
                    for sh, slices in zip(shapes, slices_by_input):
                        offset = (sh[dim] - size_sh) // 2
                        slices.append(slice(offset, offset+size_sh))
                else:
                    raise Exception("Wrong value")

        return [input[slices] for input, slices in
                zip(inputs, slices_by_input)]


def fix_dims_array_shapes(input_shapes, dim_fix):

    if dim_fix is None:
        return input_shapes
    else:
        dims = len(input_shapes[0])
        if not all(len(sh) == dims for sh in input_shapes):
            raise Exception("Not all inputs are of the same dimensionality")

        result = []

        dim_fix = list(dim_fix)
        if dims > len(dim_fix):
            diff=dims-len(dim_fix)
            dim_fix = list(dim_fix) + [None] * diff

        for sh, fx in zip(zip(*input_shapes), dim_fix):
            if fx is None:
                result.append(sh)
            elif fx in {'lower', 'center', 'upper'}:
                result.append([min(sh)] * len(sh))
            else:
                raise Exception("Wrong value")
        return [tuple(sh) for sh in zip(*result)]

class Dropout(BLayer):

    def __init__(self, prev_layer, p=0.5, rescale=True, **kwargs):
        super(Dropout, self).__init__(prev_layer, **kwargs)
        self.scale = rescale
        self.p = p
        self._srng = RandomStreams(_rng.randint(1, 2147462579))

    def calc_output(self, input, deterministic=False, **kwargs):
        if deterministic or self.p == 0:
            return input
        else:
            q = 1 - self.p
            if self.scale:
                input /= q
            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            return input * self._srng.binomial(input_shape, p=q,
                                               dtype=input.dtype)

if __name__ == '__main__':
    import numpy
    import theano
    a = numpy.random.random((1, 2, 3, 4))
    b = numpy.random.random((5, 4, 4, 2))
    c = numpy.random.random((7, 1, 8, 9))
    cropping = [None, 'lower', 'center', 'upper']

    xa, xb, xc = fix_dims([theano.shared(a),
                               theano.shared(b),
                               theano.shared(c)], cropping)
    xa, xb, xc = xa.eval(), xb.eval(), xc.eval()

    print (xa == a[:, :1, :3, -2:]).all()

    print (xb == b[:, :1, :3, -2:]).all()

    print (xc == c[:, :1, 2:5:, -2:]).all()