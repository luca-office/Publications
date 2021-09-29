import re
import string
import numpy as np
import tensorflow as tf

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


def process_mail(mail):
    """Process mail function.
    Input:
        mail: a string containing a mail
    Output:
        mail_clean: a list of words containing the processed mail

    """
    stemmer = PorterStemmer()
    stopwords_german = stopwords.words('german')
    # remove hyperlinks
    mail = re.sub(r'https?:\/\/.*[\r\n]*', '', mail)
    # remove hashtags
    # only removing the hash # sign from the word
    mail = re.sub(r'#', '', mail)
    # tokenize mails
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    mail_tokens = tokenizer.tokenize(mail)

    mail_clean = []
    for word in mail_tokens:
        if (word not in stopwords_german and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # mails_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            mail_clean.append(stem_word)

    return mail_clean

def build_freqs(mails, ys):
    """Build frequencies.
    Input:
        mails: a list of mails
        ys: an m x 1 array with the politeness label of each mail
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, politeness) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all mails
    # and over all processed words in each mail.
    freqs = {}
    for y, mail in zip(yslist, mails):
        for word in process_mail(mail):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

def extract_features(tweet, freqs):
    '''
    Input: 
        mail: a list of words for one mail
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,3)
    '''
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_mail(tweet)
    
    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 2)) 
    
    # loop through each word in the list of words
    for word in word_l:
        
        # increment the word count for the positive label 1
        x[0,0] += freqs.get((word, 1.0),0.0)
        
        # increment the word count for the negative label 0
        x[0,1] += freqs.get((word, 0.0),0.0)
        
    assert(x.shape == (1, 2))
    return x

class F1_metric(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        # Initialize our metric by initializing the two metrics it's based on:
        # Precision and Recall
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Update our metric by updating the two metrics it's based on
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

    def result(self):   
        # To get the F1 result, we compute the harmonic mean of the current
        # precision and recall
        return 2 / ((1 / self.precision.result()) + (1 / self.recall.result())) 