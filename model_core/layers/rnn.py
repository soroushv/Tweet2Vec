import numpy as np
import theano
import theano.tensor as T
from model_core.activations import relu, linear, sigmoid, tanh
from model_core.initialization import Init
from model_core.layers.base_layer import BaseLayer, Fuse, BLayer
from model_core.layers.common_utils import calculate_output
from model_core.layers.basic_layers import FC, Input
from model_core.layers.common_utils import collect_all_conf_params


class Gate(object):
    def __init__(self, W_i=Init(init_type='normal', mean=0.1), U_h=Init(init_type='normal', mean=0.1),
                 W_cell=Init(init_type='normal', mean=0.1), b=Init(init_type='constant', val=0.),
                 act=sigmoid):
        self.W_i = W_i
        self.U_h = U_h
        # Don't store a cell weight vector when cell is None
        if W_cell is not None:
            self.W_c = W_cell
        self.b = b
        self.activation = linear if act is None else act


class BaseRNNLayer(Fuse):
    def __init__(self, prev_layer, i_h, h_h, act=relu, no_return_seq=False, h_beg=Init(init_type='constant', val=0.),
                 back_rnn=False,
                 trainable_init_param=False, gradient_steps=-1, grad_clip=0, unroll_scan=False, precompute=True,
                 mask_ip=None, **kwargs):
        self.i_h = i_h
        self.h_h = h_h
        self.no_return_seq = no_return_seq
        self.activation = linear if act is None else act
        if not isinstance(prev_layer, list):
            prev_layers = [prev_layer]
        else:
            prev_layers = prev_layer

        if mask_ip is not None:
            self.mask_idx = len(prev_layers)
            prev_layers.append(mask_ip)
        else:
            self.mask_idx = None

        self.back_rnn = back_rnn
        self.grad_clip = grad_clip
        self.unroll_scan = unroll_scan
        self.gradient_steps = gradient_steps

        self.trainable_init_param = trainable_init_param
        self.precompute = precompute

        if isinstance(h_beg, BLayer):
            self.h_beg_idx = len(prev_layers)
            prev_layers.append(h_beg)
        else:
            self.h_beg_idx = None

        super(BaseRNNLayer, self).__init__(prev_layers, **kwargs)
        i_h_output_shape = self.i_h.calc_output_shape(self.i_h.input_shape)
        h_h_output_shape = self.i_h.calc_output_shape(self.i_h.input_shape)
        if isinstance(h_beg, BLayer):
            self.h_beg = h_beg
        else:
            self.h_beg = self.set_conf_params(h_beg, (1,) + h_h_output_shape[1:], name="h_beg", reg=False,
                                              train=trainable_init_param)

        first_shape = self.input_shapes[0]
        precomputed_shape_check = precompute and i_h_output_shape[0] is not None and first_shape[0] is not None and \
                                  first_shape[1] is not None and i_h_output_shape[0] != (
            first_shape[0] * first_shape[1])
        non_precompute_shape_check = not precompute and i_h_output_shape[0] is not None and h_h_output_shape[
                                                                                                0] is not None and \
                                     i_h_output_shape[0] != h_h_output_shape[0]

        def check_two_shape_tuples(x, y):
            shape_valid = True
            for a, b in zip(x, y):
                shape_valid &= (a == b or a is None or b is None)
            return shape_valid

        i_h_h_h_op_shape_check = not check_two_shape_tuples(i_h_output_shape[1:], h_h_output_shape[1:])
        i_h_op_h_h_ip_shape_check = not check_two_shape_tuples(i_h_output_shape[1:], h_h.input_shape[1:])
        if precomputed_shape_check or non_precompute_shape_check or i_h_h_h_op_shape_check or i_h_op_h_h_ip_shape_check:
            raise Exception("Shape check failed !! Verify shapes!! ")

    def calc_output_shape(self, all_input_shape):
        input_shape = all_input_shape[0]
        h_h_output_shape = self.h_h.calc_output_shape(self.h_h.input_shape)
        if self.no_return_seq:
            return (input_shape[0],) + h_h_output_shape[1:]
        else:
            return ((input_shape[0], input_shape[1]) + h_h_output_shape[1:])

    def calc_output(self, fused_inputs, **kwargs):

        main_input = fused_inputs[0]
        mask = fused_inputs[self.mask_idx] if self.mask_idx is not None else None
        h_beg = fused_inputs[self.h_beg_idx] if self.h_beg_idx is not None else None
        main_input = main_input.dimshuffle(1, 0, *range(2, main_input.ndim))
        seq_len, n_batch = main_input.shape[:2]

        if self.precompute:
            main_input = T.reshape(main_input, (seq_len * n_batch,) + main_input.shape[2:main_input.ndim])
            main_input = calculate_output(self.i_h, main_input, **kwargs)
            main_input = T.reshape(main_input, (seq_len, n_batch) + main_input.shape[1:main_input.ndim])

        non_seqs = collect_all_conf_params(self.h_h)

        if not self.precompute:
            non_seqs += collect_all_conf_params(self.i_h)

        def step_fn(input_precomputed_orig, h_prev):
            if self.precompute:
                h_val = input_precomputed_orig + calculate_output(self.h_h, h_prev, **kwargs)
            else:
                h_val = calculate_output(self.i_h, input_precomputed_orig, **kwargs) + calculate_output(
                    self.h_h, h_prev, **kwargs)
            if self.grad_clip:
                h_val = theano.gradient.grad_clip(
                    h_val, -self.grad_clip, self.grad_clip)

            return self.activation(h_val)

        def step_mask_fn(input_precomputed_orig, imask, h_prev):
            new_h = step_fn(input_precomputed_orig, h_prev)
            out_h = T.switch(imask, ift=new_h, iff=h_prev)
            return [out_h]

        if self.mask_idx is None:
            seqs = main_input
            step_func = step_fn
        else:
            seqs = [main_input, mask.dimshuffle(1, 0, 'x')]
            step_func = step_mask_fn

        if not isinstance(self.h_beg, BLayer):
            dot_dims = (list(range(1, self.h_beg.ndim - 1)) +
                        [0, self.h_beg.ndim - 1])
            h_beg = T.dot(T.ones((n_batch, 1)),
                          self.h_beg.dimshuffle(dot_dims))

        out_h = theano.scan(fn=step_func, sequences=seqs, outputs_info=[h_beg], non_sequences=non_seqs,
                            truncate_gradient=self.gradient_steps, go_backwards=self.back_rnn, strict=True)[0]
        if self.no_return_seq:
            out_h = out_h[-1]
        else:
            out_h = out_h.dimshuffle(1, 0, *range(2, out_h.ndim))

            if self.back_rnn:
                out_h = out_h[:, ::-1]

        return out_h

    def get_conf_params(self, **kwargs):
        conf_params = super(BaseRNNLayer, self).get_conf_params(**kwargs)
        conf_params.extend(collect_all_conf_params(self.i_h, **kwargs))
        conf_params.extend(collect_all_conf_params(self.h_h, **kwargs))
        return conf_params


class VanillaRNN(BaseRNNLayer):
    def __init__(self, prev_layer, num_units,
                 W_i_h=Init(init_type='uniform'),
                 W_h_h=Init(init_type='uniform'),
                 b=Init(init_type='constant', val=0.),
                 act=relu,
                 h_beg=Init(init_type='constant', val=0.),
                 back_rnn=False,
                 trainable_init_param=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):

        if isinstance(prev_layer, tuple):
            input_shape = prev_layer
        else:
            input_shape = prev_layer.output_shape

        if 'name' in kwargs:
            varname = kwargs['name'] + '.'

            layer_kwargs = {key: arg for key, arg in kwargs.items()
                            if key != 'name'}
        else:
            varname = ''
            layer_kwargs = kwargs
        input_ = Input((None,) + input_shape[2:])
        hid_ = Input((None, num_units))
        i_h = FC(input_, num_units, W=W_i_h, b=b, act=None, name="%s%s" % (varname, 'i2h'), **layer_kwargs)
        h_h = FC(hid_, num_units, W=W_h_h, b=None, act=None, name="%s%s" % (varname, 'h2h'), **layer_kwargs)
        self.W_i_h = i_h.W
        self.W_h_h = h_h.W
        self.b = i_h.b

        super(VanillaRNN, self).__init__(
            prev_layer, i_h, h_h, act=act,
            h_beg=h_beg, back_rnn=back_rnn, trainable_init_param=trainable_init_param,
            gradient_steps=gradient_steps,
            grad_clipping=grad_clipping, unroll_scan=unroll_scan,
            precompute_input=precompute_input, mask_input=mask_input,
            only_return_final=only_return_final, **kwargs)


class LSTM(Fuse):
    def __init__(self, prev_layer, output_units, i_gate=Gate(), f_gate=Gate(), c_gate=Gate(W_cell=None, act=tanh),
                 o_gate=Gate(), act=tanh, cell_beg=Init(init_type='constant', val=0.),
                 h_beg=Init(init_type='constant', val=0.),
                 back_rnn=False, trainable_init_param=False, peepholes=True, gradient_steps=-1, grad_clip=0,
                 precompute=True, nof=False, cifg=False, noi=False,
                 mask_input=None, no_return_seq=False, **kwargs):

        prev_layers = [prev_layer]
        self.i_gate = i_gate
        self.f_gate = f_gate
        self.o_gate = o_gate
        self.noi = noi
        self.nof = nof
        self.cifg = cifg
        self.c_gate = c_gate
        self.gradient_steps = gradient_steps
        self.grad_clip = grad_clip
        self.precompute = precompute
        self.no_return_seq = no_return_seq
        self.trainable_init_param = trainable_init_param
        self.output_units = output_units
        self.back_rnn = back_rnn
        self.peepholes = peepholes

        if mask_input is not None:
            self.mask_idx = len(prev_layers)
            prev_layers.append(mask_input)

        else:
            self.mask_idx = None
        if isinstance(h_beg, BaseLayer):
            self.h_beg_idx = len(prev_layers)
            prev_layers.append(h_beg)
        else:
            self.h_beg_idx = None

        if isinstance(cell_beg, BaseLayer):
            self.cell_beg_idx = len(prev_layers)
            prev_layers.append(cell_beg)
        else:
            self.cell_beg_idx = None



        super(LSTM, self).__init__(prev_layers, **kwargs)

        self.activation = linear if act is None else act
        input_shape = self.input_shapes[0]
        input_dims = np.prod(input_shape[2:])

        if isinstance(h_beg, BaseLayer):
            self.h_beg = h_beg
        else:
            self.h_beg = self.set_conf_params(
                h_beg, (1, self.output_units), name="h_beg",
                train=trainable_init_param, reg=False)

        if isinstance(cell_beg, BaseLayer):
            self.cell_beg = cell_beg
        else:
            self.cell_beg = self.set_conf_params(
                cell_beg, (1, output_units), name="cell_beg",
                train=trainable_init_param, reg=False)

        if self.peepholes:
            self.W_pci = self.set_conf_params(
                i_gate.W_c, (output_units,), name="W_pci")

            self.W_pcf = self.set_conf_params(
                f_gate.W_c, (output_units,), name="W_pcf")

            self.W_pco = self.set_conf_params(
                o_gate.W_c, (output_units,), name="W_pco")

        self.W_i = self.set_conf_params(i_gate.W_i, (input_dims, output_units), name="W_i2%s_gate" % ('input'))
        self.U_i = self.set_conf_params(i_gate.U_h, (output_units, output_units), name="U_h2%s_gate" % ('input'))
        self.b_i = self.set_conf_params(i_gate.b, (output_units,), name="b%s_gate" % ('input'), reg=False)
        self.i_gate_act = i_gate.activation

        self.W_f = self.set_conf_params(f_gate.W_i, (input_dims, output_units), name="W_i2%s_gate" % ('forget'))
        self.U_f = self.set_conf_params(f_gate.U_h, (output_units, output_units), name="U_h2%s_gate" % ('forget'))
        self.b_f = self.set_conf_params(f_gate.b, (output_units,), name="b%s_gate" % ('forget'), reg=False)
        self.f_gate_act = f_gate.activation

        self.W_c = self.set_conf_params(c_gate.W_i, (input_dims, output_units), name="W_i2%s_gate" % ('cell'))
        self.U_c = self.set_conf_params(c_gate.U_h, (output_units, output_units), name="U_h2%s_gate" % ('cell'))
        self.b_c = self.set_conf_params(c_gate.b, (output_units,), name="b%s_gate" % ('cell'), reg=False)
        self.c_gate_act = c_gate.activation

        self.W_o = self.set_conf_params(o_gate.W_i, (input_dims, output_units), name="W_i2%s_gate" % ('output'))
        self.U_o = self.set_conf_params(o_gate.U_h, (output_units, output_units), name="U_h2%s_gate" % ('ouput'))
        self.b_o = self.set_conf_params(o_gate.b, (output_units,), name="b%s_gate" % ('output'), reg=False)
        self.o_gate_act = o_gate.activation

    def calc_output_shape(self, input_shapes):
        main_input_shape = input_shapes[0]
        if self.no_return_seq:
            return main_input_shape[0], self.output_units
        else:
            return main_input_shape[0], main_input_shape[1], self.output_units

    def calc_output(self, inputs, **kwargs):

        input = inputs[0]
        mask = inputs[self.mask_idx] if self.mask_idx is not None  else None
        h_beg = inputs[self.h_beg_idx] if self.h_beg_idx is not None  else None
        cell_beg = inputs[self.cell_beg_idx] if self.cell_beg_idx is not None  else None
        if input.ndim > 3:
            input = T.flatten(input, 3)

        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape
        W_st = T.concatenate([self.W_i, self.W_f, self.W_c, self.W_o], axis=1)
        U_st = T.concatenate([self.U_i, self.U_f, self.U_c, self.U_o], axis=1)
        b_st = T.concatenate([self.b_i, self.b_f, self.b_c, self.b_o], axis=0)

        if self.precompute:
            input = T.dot(input, W_st) + b_st

        non_seqs = [U_st]

        if self.peepholes:
            non_seqs += [self.W_pci,
                         self.W_pcf,
                         self.W_pco]
        if not self.precompute:
            non_seqs += [W_st, b_st]

        def gate_data(wt_mat, gate_id):
            return wt_mat[:, gate_id * self.output_units:(gate_id + 1) * self.output_units]

        def step(i_t, c_tm1, h_tm1, *args):
            if not self.precompute:
                i_t = T.dot(i_t, W_st) + b_st
            gates = i_t + T.dot(h_tm1, U_st)
            if self.grad_clip:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clip, self.grad_clip)
            ingate = gate_data(gates, 0)
            forgetgate = gate_data(gates, 1)
            cell_input = gate_data(gates, 2)
            outgate = gate_data(gates, 3)

            if self.peepholes:
                ingate += c_tm1*self.W_pci
                forgetgate += c_tm1*self.W_pcf

            ingate = self.i_gate_act(ingate)
            forgetgate = self.f_gate_act(forgetgate)
            cell_input = self.c_gate_act(cell_input)

            cell = forgetgate*c_tm1 + ingate*cell_input

            if self.peepholes:
                outgate += cell*self.W_pco
            outgate = self.o_gate_act(outgate)

            hid = outgate*self.o_gate_act(cell)

            return [cell, hid]

        def step_masked(input_n, mask_n, cell_previous, hid_previous, *args):
            cell, hid = step(input_n, cell_previous, hid_previous, *args)
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)

            return [cell, hid]

        if mask is not None:

            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_beg, BaseLayer):
            cell_beg = T.dot(ones, self.cell_beg)

        if not isinstance(self.h_beg, BaseLayer):
            h_beg = T.dot(ones, self.h_beg)

        cell_out, hid_out = theano.scan(
            fn=step_fun,
            sequences=sequences,
            outputs_info=[cell_beg, h_beg],
            go_backwards=self.back_rnn,
            truncate_gradient=self.gradient_steps,
            non_sequences=non_seqs,
            strict=True)[0]
        if self.no_return_seq:
            hid_out = hid_out[-1]
        else:
            hid_out = hid_out.dimshuffle(1, 0, 2)
            if self.back_rnn:
                hid_out = hid_out[:, ::-1]

        return hid_out


class LSTMLayer(Fuse):

    def __init__(self, incoming, num_units,
                 ingate=Gate(),
                 forgetgate=Gate(),
                 cell=Gate(W_cell=None, act=tanh),
                 outgate=Gate(),
                 act=tanh,
                 cell_init=Init(init_type='constant', val=0.),
                 hid_init=Init(init_type='constant', val=0.),
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 no_return_seq=False,
                 **kwargs):


        prev_layers = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        self.cell_init_incoming_index = -1
        if mask_input is not None:
            prev_layers.append(mask_input)
            self.mask_incoming_index = len(prev_layers)-1
        if isinstance(hid_init, BaseLayer):
            prev_layers.append(hid_init)
            self.hid_init_incoming_index = len(prev_layers)-1
        if isinstance(cell_init, BaseLayer):
            prev_layers.append(cell_init)
            self.cell_init_incoming_index = len(prev_layers)-1

        super(LSTMLayer, self).__init__(prev_layers, **kwargs)

        if act is None:
            self.activation = linear
        else:
            self.activation = act

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = no_return_seq

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):

            return (self.set_conf_params(gate.W_i, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.set_conf_params(gate.U_h, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.set_conf_params(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   reg=False),
                    gate.activation)

        (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate,
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')

        (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate,
                                                         'forgetgate')

        (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')

        (self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate,
         self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')


        if self.peepholes:
            self.W_cell_to_ingate = self.set_conf_params(
                ingate.W_c, (num_units, ), name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.set_conf_params(
                forgetgate.W_c, (num_units, ), name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.set_conf_params(
                outgate.W_c, (num_units, ), name="W_cell_to_outgate")

        if isinstance(cell_init, BaseLayer):
            self.cell_init = cell_init
        else:
            self.cell_init = self.set_conf_params(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)

        if isinstance(hid_init, BaseLayer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.set_conf_params(
                hid_init, (1, self.num_units), name="hid_init",
                train=learn_init, reg=False)

    def calc_output_shape(self, input_shapes):

        input_shape = input_shapes[0]
        if self.only_return_final:
            return input_shape[0], self.num_units
        else:
            return input_shape[0], input_shape[1], self.num_units

    def calc_output(self, inputs, **kwargs):

        input = inputs[0]
        mask = None
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        if input.ndim > 3:
            input = T.flatten(input, 3)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape


        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=1)

        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)

        if self.precompute_input:

            input = T.dot(input, W_in_stacked) + b_stacked


        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]


        def step(input_n, cell_previous, hid_previous, *args):
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate

            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            cell = forgetgate*cell_previous + ingate*cell_input

            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)

            hid = outgate*self.activation(cell)
            return [cell, hid]

        def step_masked(input_n, mask_n, cell_previous, hid_previous, *args):
            cell, hid = step(input_n, cell_previous, hid_previous, *args)

            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)

            return [cell, hid]

        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, BaseLayer):
            cell_init = T.dot(ones, self.cell_init)

        if not isinstance(self.hid_init, BaseLayer):
            hid_init = T.dot(ones, self.hid_init)

        non_seqs = [W_hid_stacked]
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]


        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        cell_out, hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)[0]


        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            hid_out = hid_out.dimshuffle(1, 0, 2)

            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out

