import math

__author__ = 'pralav'
import re
import numpy as np
class DataManager(object):
    def __init__(self,logging,max_char_size=150):
        self.logger=logging.getLogger(__name__)
        self.alphabets = "Pabcdefghijklmnopqrstuvwxyz0123456789-,;.!?:~'\"/\\|_@#$%^&*+-=<>()[]{}BESHU"
        self.alpha_idx = {alpha: i for i, alpha in enumerate(self.alphabets)}
        self.max_char_size=max_char_size
        # self.cand_mapper=CandidateTweetMapper(self.logging)


    def clean_tweet(self, tweet):
        try:
            tweet=tweet.encode('ascii', 'ignore').lower()
        except:
            pass
        # tweet = self.cand_mapper.replace_tweet_c(tweet=tweet.lower())
        tweet = re.sub(r"can'?t", ' can not', tweet)
        tweet = re.sub(r"n't", ' not', tweet)
        tweet = re.sub(r"'s", ' is', tweet)
        tweet = re.sub(r"i'm", ' i am ', tweet)
        tweet = re.sub(r"'ll", ' will', tweet)
        tweet = re.sub(r"'ve", ' have', tweet)
        tweet = re.sub(r"'d", ' would', tweet)
        tweet = re.sub(r'\&amp;|\&gt;|&lt;|\&', ' and ', tweet)
        url = re.compile(r'(https?[^\s]*)')
        smile = re.compile(r'[8:=;][\'`\-]?[\)d]+|[)d]+[\'`\-][8:=;]')
        sad = re.compile(r'[8:=;][\'`\-]?\(+|\)+[\'`\-][8:=;]')
        lol = re.compile(r'[8:=;][\'`\-]?p+')
        tweet = re.sub(r'\@[^\s]+', ' U ', tweet)
        tweet = url.sub( ' ', tweet)
        tweet = re.sub(r'\/', ' ', tweet)
        tweet = smile.sub( ' H ', tweet)
        tweet = lol.sub(' H ', tweet)
        tweet = sad.sub( ' S ', tweet)
        tweet = re.sub(r'([\!\?\.]){2,}', '\g<1>', tweet)
        tweet = re.sub(r'\b(\S*?)([^\s])\2{2,}\b', '\g<1>\g<2>', tweet)
        tweet = re.sub(r'\#', ' #', tweet)
        tweet = re.sub(r'[^\w\#\s\?\<\>]+', ' ', tweet)
        tweet = re.sub('\s+', ' ', tweet)
        return 'B '+tweet.strip()+' E'

    def text_to_one_hot_char(self, text):
        x = np.zeros((self.max_char_size, len(self.alphabets) + 1))
        for i, char in enumerate(list(text.lower())[:self.max_char_size]):
            if char not in self.alpha_idx:
                x[i, -1] = 1
            else:
                x[i, self.alpha_idx[char]] = 1
        return x

    def text_to_char_idx(self, text):
        alpha_len = len(self.alpha_idx)-1
        x=[]
        for i in range(self.max_char_size):
            if i < len(text):
                x.append(self.alpha_idx.get(text[i],alpha_len))
            else:
                x.append(0)


        x=np.array(x)
        return x

    def text_to_char_seq(self,text):
        x=[]
        for i, char in enumerate(list(text.lower())[:self.max_char_size]):

            if char not in self.alpha_idx:
                x.append(self.alpha_idx[len(self.alphabets)-1])
            else:
                x.append(self.alpha_idx[char])
        return x

    def batch_generator(self, data, batch_size=64):

        n_batches = int(math.ceil(len(data) / batch_size))
        for i in range(n_batches):
            labels = []
            act_matrix = []
            start = i * batch_size
            end = i * batch_size + batch_size
            batch_data = data[start:end]
            for x,y in batch_data:
                label = y[0] if isinstance(y, list) else y
                labels.append(label)
                act_matrix.append(self.text_to_one_hot_char(self.clean_tweet(x)).T)

            yield (i, np.array(act_matrix).astype('float32'), np.array(labels))

    def batch_generator_test(self, tweets, decoding_tweets,batch_size=64):

        n_batches = int(math.ceil(len(tweets) / batch_size))
        for i in range(n_batches):
            act_matrix = []
            decoding_matrix = []
            y_mask=np.zeros((batch_size,self.max_char_size),dtype='int8')
            start = i * batch_size
            end = i * batch_size + batch_size
            original_tweets = tweets[start:end]
            label_tweets = decoding_tweets[start:end]

            for i,(x,y) in enumerate(zip(original_tweets,label_tweets)):
                act_matrix.append(self.text_to_one_hot_char(self.clean_tweet(x)).T)
                clean_y=self.clean_tweet(y)
                len_y=len(clean_y) if len(clean_y)<self.max_char_size else self.max_char_size
                decoding_matrix.append(self.text_to_char_idx(clean_y))
                # decoding_matrix.append(self.text_to_one_hot_char(self.clean_tweet(x)))
                y_mask[i,:len_y]=1


            yield (i, np.array(act_matrix).astype('float32'),np.array(decoding_matrix).astype('int32'),np.array(y_mask).astype('int8'))
