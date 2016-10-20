import argparse
from logger import setup_logging
from model.data_manager import DataManager
from project_settings import LOG_PATH, MODELS, TWEET2VEC, OUTPUTS
from utils import Utils

__author__ = 'pralav'
from collections import OrderedDict
from model_core.activations import softmax, relu

from model_core.layers.basic_layers import FC, Input, Repeat, Reshape
from model_core.layers.common_utils import calculate_output, set_conf_param_values
from model_core.layers.conv_cuda import Convolution1D, MaxPool1D
from model_core.layers.conv_op import ConvOps
from model_core.layers.fuse import FuseLayer, Dropout
from model_core.layers.common_utils import collect_all_conf_params
from model_core.layers.rnn import LSTM
import theano
import theano.tensor as T
from model_core.loss import categorical_crossentropy
from model_core.update_fns import adam, ful_norm_const
import numpy as np

ADAM_LR = theano.shared(np.float32(0.001), 'Adam LR')
logging = setup_logging(save_path=LOG_PATH + "/model_embedding.log")
utils = Utils(logging)
logger=logging.getLogger(__name__)


def build_model(num_filters=256, encoder_size=256,char_vocab_size=75,max_chars=150):
    input_var = T.tensor3('input_var')
    convops = ConvOps()
    input_layer = Input((None, char_vocab_size, max_chars), theano_sym=input_var, name='l_in')
    conv_1 = Convolution1D(input_layer, num_filters, 7, convolution=convops)
    pool_1 = MaxPool1D(conv_1, 3)
    conv_2 = Convolution1D(pool_1, num_filters, 7, convolution=convops)
    pool_2 = MaxPool1D(conv_2, 3)
    conv_3 = Convolution1D(pool_2, num_filters, 3, convolution=convops)
    conv_4 = Convolution1D(conv_3, num_filters, 3, convolution=convops)
    encoding_layer = LSTM(conv_4, encoder_size, no_return_seq=True,name='encoder')

    return input_var,input_layer,encoding_layer


def load_model_values(func_params, model_path):
    enc_model = build_model(**func_params)
    model_param_values = utils.load_file_joblib(model_path)
    set_conf_param_values(enc_model, model_param_values)


def get_embeddings(tweets, model_name, epoch, output_file):
    out_path = utils.get_path(MODELS, TWEET2VEC) + "/%s/" % model_name
    enc_params = utils.load_file_joblib("%s/encoding_%s_%s.npy" % (out_path, model_name, epoch))
    (max_chars, num_filters, encoder_size, decoder_size, char_vocab_size, grad_clip, norm_constraint,
     learning_rate) = utils.load_file_joblib("%s/encoding_%s_%s_global.npy" % (out_path, model_name, epoch))

    input_var,input_layer,lstm_enc=build_model(num_filters=num_filters,encoder_size=encoder_size,char_vocab_size=char_vocab_size,max_chars=max_chars)
    set_conf_param_values(lstm_enc,enc_params)
    data_manager=DataManager(logging)
    tweet_reps=[]
    for i,xx,yy in data_manager.batch_generator_test(tweets,tweets,batch_size=256):
        logger.info(" %s,%s"%(xx.shape,len(yy)))
        reps= calculate_output(lstm_enc, inputs={input_layer: input_var}).eval({input_var: xx})
        tweet_reps.extend(reps)

    utils.save_file_joblib(tweet_reps,output_file)

def process_request(input_file,model_name, epoch, output_file=None):
    tweets=utils.load_file(input_file)
    def_out_path = utils.get_path(OUTPUTS, TWEET2VEC) + "/%s/res/results.npy" % model_name
    if output_file is None:
        output_file=def_out_path
    get_embeddings(tweets,model_name,epoch,output_file)


if __name__ == '__main__':
    logging = setup_logging(save_path=LOG_PATH + "/emb.log")
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('-m', '--model_name', help='Model Name', dest='model_name',
                        default='tweet_word')
    parser.add_argument('-i', '--input_tweet_file', help='Tweet file input', dest='input_file',
                        default='/res/tweet.json')
    parser.add_argument('-o', '--output_file', help='Tweet Embeddings file output', dest='output_file',
                        default=None)
    parser.add_argument('-e', '--trained_epoch', help='Trained Epoch of Model', dest='epoch_t', type=int,
                        default=1)
    parser.add_argument('-R', '--emb', help='Representations/Embeddings', action='store_true')
    args = parser.parse_args()
    if args.emb:
        process_request(args.input_file,args.model_name,args.epoch_t,args.output_file)





