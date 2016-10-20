from collections import Counter
import random
import argparse
from model.data_manager import DataManager
from model.t2v import  build_model
from model_core.layers.common_utils import calculate_output, get_conf_param_values

__author__ = 'pralav'
from logger import setup_logging
from project_settings import LOG_PATH, DATASETS, TWEET2VEC, OUTPUTS, MODELS
from utils import Utils
import theano
import numpy as np

ADAM_LR = theano.shared(np.float32(0.001), 'Adam LR')

class Trainer(object):
    def __init__(self, logging, num_filters=256, encoder_size=256, decoder_size=128, max_chars=150, char_vocab_size=75,
                 grad_clip=15., norm_const=25, learning_rate=0.001):
        self.logging = logging
        self.utils = Utils(logging)
        self.logger = logging.getLogger(__name__)
        self.data_manager = DataManager(logging)
        self.max_chars = max_chars
        self.num_filters = num_filters
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.char_vocab_size = char_vocab_size
        self.grad_clip = grad_clip
        self.norm_constraint = norm_const
        self.learning_rate = theano.shared(np.float32(learning_rate), 'Adam LR')

    def load_data(self, train_filename):
        train_file = train_filename[train_filename.rfind("/")+1:]#'/res/train.json'

        data = self.utils.load_file(train_filename)
        tweets, decoding_tweets = [], []
        for d in data:
            tweets.append(d['tweet1'])
            decoding_tweets.append(d['tweet2'])
        self.utils.save_file((tweets, decoding_tweets),
                             self.utils.get_path(DATASETS, TWEET2VEC) + "/%s.json" % train_file)
        return tweets,decoding_tweets

    def get_result(self, layer, input_layer, input_var, input_data):
        theano_res = calculate_output(layer, inputs={input_layer: input_var})
        return theano_res.eval({input_var: input_data})

    def test_val_data(self, val_fn, val_data):
        self.logger.info("=" * 50)
        try:
            self.logger.info("Test Validation data")
            val_err = 0
            val_acc = 0
            val_batches = 0
            for i, x_, y_ in self.data_manager.batch_generator(val_data, batch_size=1000):
                if random.random() < 0.5:
                    err, acc = val_fn(x_, y_)
                    val_err += err
                    val_acc += acc

                    val_batches += 1
                if val_batches > 50:
                    break
        except Exception, e:
            self.logger.info(e)
            self.logger.exception(e)
            self.logger.error(e)
            raise e
        self.logger.info("Validation acc: {:.6f}".format(val_acc / val_batches))

    def train_enc_dec(self, train_file, model_name, num_epochs=5,out_dir=None):
        if out_dir is None:
            out_path = self.utils.get_path(MODELS, TWEET2VEC) + "/%s/" % model_name
        else:
            out_path=out_dir
        tweets, decoding_tweets=self.load_data(train_file)
        # tweets, decoding_tweets = self.utils.load_file(train_file)
        # decoding_tweets = tweets
        # print Counter(decoding_tweets), Counter(test_decoding_tweets)
        # train_data = zip()

        input_var, input_layer, encoding_layer, final_layer, train_fun, all_params = build_model(
            self.num_filters, self.encoder_size, self.decoder_size, self.max_chars, self.char_vocab_size,
            self.grad_clip, self.norm_constraint, self.learning_rate)
        for e in xrange(1, num_epochs + 1):
            self.logger.info("Starting epoch {}".format(e))
            self.logger.info("Training with the Adam update function")

            mb = 0
            for i, x_, y_, mask_ in self.data_manager.batch_generator_test(tweets, decoding_tweets, batch_size=256):
                loss = train_fun(x_, y_, mask_)
                if mb % 5 == 0:
                    self.logger.info("Epoch: {}, Minibatch: {}, Total Loss: {}".format(e, mb, loss))
            mb += 1
            self.logger.info("Training epoch {}.".format(e))
            self.logger.info("Epoch {} results".format(e))
            self.logger.info("Saving model parameters for epoch {0} at path {1}".format(e, ))
            params = get_conf_param_values(final_layer)
            enc_params = get_conf_param_values(encoding_layer)
            self.utils.save_file_joblib(params, "%s/%s_%s.npy" % (out_path, model_name, e))
            self.utils.save_file_joblib(enc_params, "%s/encoding_%s_%s.npy" % (out_path, model_name, e))
            all_params=(self.max_chars,self.num_filters,self.encoder_size ,self.decoder_size,self.char_vocab_size,self.grad_clip,self.norm_constraint,self.learning_rate)
            self.utils.save_file_joblib(all_params, "%s/encoding_%s_%s_global.npy" % (out_path, model_name, e))




if __name__ == '__main__':
    logging = setup_logging(save_path=LOG_PATH + "/training.log")
    parser = argparse.ArgumentParser(description='Trainer')

    parser.add_argument('-m', '--model_name', help='Model Name', dest='model_name',
                        default='tweet_word')
    parser.add_argument('-f', '--num_filters', help='No. of filters', dest='num_filters',
                        default=256, type=int)
    parser.add_argument('-L', '--encoder_len', help='Encoder length', dest='enc_len', type=int,
                        default=256)
    parser.add_argument('-d', '--decoder_len', help='Decoder length', dest='dec_len', type=int,
                        default=128)

    parser.add_argument('-c', '--max_char_len', help='Max Char Length', dest='max_char_len', type=int,
                        default=150)
    parser.add_argument('-g', '--grad_clip', help='Gradient Clip', dest='grad_clip',
                        default=5, type=float)

    parser.add_argument('-N', '--norm_constraint', help='Norm Constraint', dest='norm_constraint', type=float,
                        default=25)
    parser.add_argument('-E', '--epochs', help='Epochs', dest='epochs', type=int,
                        default=3)
    parser.add_argument('-l', '--learning_rate', help='Learning rate', dest='learning_rate', type=float,
                        default=0.001)
    parser.add_argument('-i', '--train_file', help='Training File', dest='train_file',
                        default='/res/train.json')

    parser.add_argument('-o', '--out_dir', help='Output Directory', dest='out_dir',
                        default=None)

    parser.add_argument('-t', '--train', help='Train', action='store_true')

    args = parser.parse_args()
    if args.train:
        trainer = Trainer(logging, num_filters=args.num_filters, encoder_size=args.enc_len, decoder_size=args.dec_len,
                          max_chars=args.max_char_len,grad_clip=args.grad_clip,norm_const=args.norm_constraint,learning_rate=args.learning_rate)
        trainer.train_enc_dec(train_file=args.train_file, model_name=args.model_name, num_epochs=args.epochs,out_dir=args.out_dir)

