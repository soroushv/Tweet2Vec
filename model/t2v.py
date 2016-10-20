__author__ = 'pralav'
from collections import OrderedDict
from model_core.activations import softmax, relu

from model_core.layers.basic_layers import FC, Input, Repeat, Reshape
from model_core.layers.common_utils import calculate_output, calculate_output_shape
from model_core.layers.conv_cuda import Convolution1D, MaxPool1D
from model_core.layers.conv_op import ConvOps
from model_core.layers.fuse import FuseLayer, Dropout
from model_core.layers.common_utils import collect_all_conf_params
from model_core.layers.rnn import LSTM,LSTMLayer
import theano
import theano.tensor as T
from model_core.loss import categorical_crossentropy
from model_core.update_fns import adam, ful_norm_const
import numpy as np


ADAM_LR=theano.shared(np.float32(0.001), 'Adam LR')
def build_model(num_filters=256,encoder_size=256,decoder_size=128,max_chars=150,char_vocab_size=75,grad_clip=15.,norm_const=25,learning_rate=ADAM_LR):
    input_var = T.tensor3('input_var')
    y = T.imatrix('output_var')
    y_mask = T.bmatrix('mask_var')
    convops=ConvOps()
    X = np.random.randint(0,10,size=(10,75,150)).astype('float32')
    # Y = np.random.randint(0,10,size=(10,75,150).astype('int32'))

    input_layer = Input((None, 75, 150), theano_sym=input_var, name='l_in')
    conv_1 = Convolution1D(input_layer, num_filters, 7, convolution=convops)
    pool_1 = MaxPool1D(conv_1, 3)
    conv_2 = Convolution1D(pool_1, num_filters, 7, convolution=convops)
    pool_2 = MaxPool1D(conv_2, 3)
    conv_3 = Convolution1D(pool_2, num_filters, 3, convolution=convops)
    conv_4 = Convolution1D(conv_3, num_filters, 3, convolution=convops)
    encoding_layer = LSTM(conv_4, encoder_size, no_return_seq=True,name='encoder')
    l_in_rep = Repeat(encoding_layer, n=max_chars)
    l_dec_1 = LSTM(l_in_rep, output_units=decoder_size, name='decoder1')
    l_dec_2 = LSTM(l_dec_1, output_units=decoder_size, name='decoder2')
    l_reshape = Reshape(l_dec_2, (-1, [2]), name="l_reshape")
    fc_layer = FC(l_reshape, char_vocab_size, act=softmax, name="l_fc")

    final_layer = Reshape(fc_layer, (input_var.shape[0], -1, char_vocab_size))
    prediction = calculate_output(final_layer, deterministic=False)
    # total_cost = T.nnet.categorical_crossentropy(
    #     T.reshape(prediction, (-1, char_vocab_size)), y.flatten())
    # probs = T.nnet.softmax(prediction.reshape([prediction.shape[0]*prediction.shape[1], prediction.shape[2]]))
    y_flat = T.flatten(y,1)
    y_flat_idx = T.arange(y_flat.shape[0]) * char_vocab_size + y_flat
    costf = -T.log(prediction.flatten()[y_flat_idx]+1e-8)
    costf = costf.reshape([y.shape[0],y.shape[1]])
    costf = (costf * y_mask).sum(0)
    loss = costf.sum()

    all_parameters = collect_all_conf_params(final_layer, train=True)
    grads=T.grad(loss,all_parameters)
    if grad_clip > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(T.switch(g2 > (grad_clip**2),
                                      g / T.sqrt(g2) * grad_clip,
                                      g))
        grads = new_grads
    all_grads = grads
    if norm_const is not None:
        all_grads,all_norm = ful_norm_const(all_grads,norm_const,return_norm=True)
    updates = adam(all_grads, all_parameters, learning_rate=learning_rate)
    adam_items = updates.items()
    adam_updates = OrderedDict(adam_items)
    print 'Creating Training Function...'
    adam_train_fun = theano.function([input_var, y, y_mask],
                                     [loss],
                                     updates=adam_updates)
    return input_var,input_layer,encoding_layer,final_layer,adam_train_fun,all_parameters




if __name__ == '__main__':
    build_model()
