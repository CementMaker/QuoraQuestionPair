#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
import datetime
import pickle
import random
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s')
logger = logging.getLogger(__name__)


def remove_stop_words(sentence, stop_words_set):
    ans = []
    for word in sentence.split():
        if word.lower() not in stop_words_set:
            ans.append(word)

    return " ".join(ans)


def remove_sample_shorter_than_ten(filePath="./data/csv/train_train.csv"):
    data = pd.read_csv(filePath).values

    number = 0
    length, range_index = len(data), []
    for index in range(length):
        if len(data[index][1].split()) < 4 and len(data[index][2].split()) < 4:
            range_index.append(index)
            number += 1

    data = np.delete(data, range_index, axis=0)
    print(number, len(data))

    pd.DataFrame(data=data).to_csv(filePath)


class data(object):
    def __init__(self, data_path, maxlen=30):
        print("获取数据，开始时间:", datetime.datetime.now().isoformat())
        self.path = os.path.dirname(__file__)

        # 获取数据, 数据来源于data_path
        self.maxlen = maxlen
        self.df = pd.read_csv(data_path).dropna()
        self.data = self.df[['question1', 'question2']].values
        self.label = self.df[['is_duplicate']].values
        logger.info('self.path: %s.' % (self.path))
        logger.info('self.data shape: (%d, %d)' % (self.data.shape))
        logger.info('self.label shape: (%d, %d)' % (self.label.shape))

        self.label = np.squeeze(self.label, axis=-1)
        self.one_hot_label = np.zeros(shape=(len(self.label), 2))
        self.one_hot_label[np.arange(0, len(self.label)), self.label] = 1
        logger.info('one hot label shape: (%d, %d)' % (self.one_hot_label.shape))

    @staticmethod
    def text_to_wordlist(text):
        text = re.sub(r"[^A-Za-z0-9]", " ", text)
        text = re.sub(r"what's", "", text)
        text = re.sub(r"What's", "", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"I'm", "I am", text)
        text = re.sub(r" m ", " am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"60k", " 60000 ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e-mail", "email", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"quikly", "quickly", text)
        text = re.sub(r" usa ", " America ", text)
        text = re.sub(r" USA ", " America ", text)
        text = re.sub(r" u s ", " America ", text)
        text = re.sub(r" uk ", " England ", text)
        text = re.sub(r" UK ", " England ", text)
        text = re.sub(r"india", "India", text)
        text = re.sub(r"switzerland", "Switzerland", text)
        text = re.sub(r"china", "China", text)
        text = re.sub(r"chinese", "Chinese", text)
        text = re.sub(r"imrovement", "improvement", text)
        text = re.sub(r"intially", "initially", text)
        text = re.sub(r"quora", "Quora", text)
        text = re.sub(r" dms ", "direct messages ", text)
        text = re.sub(r"demonitization", "demonetization", text)
        text = re.sub(r"actived", "active", text)
        text = re.sub(r"kms", " kilometers ", text)
        text = re.sub(r"KMs", " kilometers ", text)
        text = re.sub(r" cs ", " computer science ", text)
        text = re.sub(r" upvotes ", " up votes ", text)
        text = re.sub(r" iPhone ", " phone ", text)
        text = re.sub(r"\0rs ", " rs ", text)
        text = re.sub(r"calender", "calendar", text)
        text = re.sub(r"ios", "operating system", text)
        text = re.sub(r"gps", "GPS", text)
        text = re.sub(r"gst", "GST", text)
        text = re.sub(r"programing", "programming", text)
        text = re.sub(r"bestfriend", "best friend", text)
        text = re.sub(r"dna", "DNA", text)
        text = re.sub(r"III", "3", text)
        text = re.sub(r"the US", "America", text)
        text = re.sub(r"Astrology", "astrology", text)
        text = re.sub(r"Method", "method", text)
        text = re.sub(r"Find", "find", text)
        text = re.sub(r"banglore", "Banglore", text)
        text = re.sub(r" J K ", " JK ", text)
        return text

    def get_one_hot(self):
        if not os.path.exists(os.path.join(self.path, "./data/pkl/test.pkl")) \
           or not os.path.exists(os.path.join(self.path, "./data/pkl/length.pkl"))\
           or not os.path.exists(os.path.join(self.path, "./data/pkl/train.pkl")):
            self.context = self.data.reshape(-1)


            # 清洗数据
            self.context = [self.text_to_wordlist(line) for line in tqdm(self.context)]
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
                num_words=20000,  # 词表大小为20000，词表的提取根据TF的计算结果排序
                filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ',
                lower=True,
                split=' ',
                char_level=False,
                oov_token=None,
                document_count=0)

            self.tokenizer.fit_on_texts(tqdm(self.context))
            self.context = np.array(self.tokenizer.texts_to_sequences(tqdm(self.context)))

            self.length = np.array([self.maxlen if(len(sentence) > self.maxlen) else len(sentence) \
                    for sentence in tqdm(self.context)])
            self.context = pad_sequences(tqdm(self.context), maxlen=self.maxlen, padding='post')

            self.length = self.length.reshape((int(len(self.context) / 2), 2))
            self.context = self.context.reshape((int(len(self.context) / 2), 2, self.maxlen))
            logger.info("self.context.shape: (%d, %d, %d)" % (self.context.shape))

            self.train_data, self.test_data, \
            self.train_label, self.test_label, \
            self.train_data_length, self.test_data_length = \
                train_test_split(self.context, self.one_hot_label, self.length, random_state=0, test_size=0.01)

            # 保存相关数据
            pickle.dump(obj=(self.train_data, self.train_label), file=open("./data/pkl/train.pkl", "wb"))
            pickle.dump(obj=(self.test_data, self.test_label), file=open("./data/pkl/test.pkl", "wb"))
            pickle.dump(obj=(self.train_data_length, self.test_data_length), file=open("./data/pkl/length.pkl", "wb"))

            logger.info('self.train_data shape: (%d, %d, %d)' % (self.train_data.shape))
            logger.info('self.train_label shape: (%d, %d)' % (self.train_label.shape))
            logger.info('self.test_data shape: (%d, %d, %d)' % (self.test_data.shape))
            logger.info('self.test_label shape: (%d, %d)' % (self.test_label.shape))
            logger.info('self.train_data_length shape: (%d, %d)' % (self.train_data_length.shape))
            logger.info('self.test_data_length shape: (%d, %d)' % (self.test_data_length.shape))
        else:
            self.train_data, self.train_label = pickle.load(open("./data/pkl/train.pkl", "rb"))
            self.test_data, self.test_label = pickle.load(open("./data/pkl/test.pkl", "rb"))
            self.train_data_length, self.test_data_length = pickle.load(open("./data/pkl/length.pkl", "rb"))
            logger.info('self.train_data shape: (%d, %d, %d)' % (self.train_data.shape))
            logger.info('self.train_label shape: (%d, %d)' % (self.train_label.shape))
            logger.info('self.test_data shape: (%d, %d, %d)' % (self.test_data.shape))
            logger.info('self.test_label shape: (%d, %d)' % (self.test_label.shape))
            logger.info('self.train_data_length shape: (%d, %d)' % (self.train_data_length.shape))
            logger.info('self.test_data_length shape: (%d, %d)' % (self.test_data_length.shape))
        return self

    @staticmethod
    def get_batch(epoches, batch_size, data, label, sentence_length):
        data = list(zip(data, label, sentence_length))
        for epoch in range(epoches):
            random.shuffle(data)
            for batch in range(0, len(data), batch_size):
                if batch + batch_size >= len(data):
                    yield data[batch: len(data)]
                else:
                    yield data[batch: (batch + batch_size)]


if __name__ == '__main__':
    data_file = "./data/csv/train.csv"
    Data = data(data_path=data_file, maxlen=70).get_one_hot()
