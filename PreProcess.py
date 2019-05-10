#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
import pickle
import random
import statistics

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

from collections import Counter
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer

path = os.getcwd()


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
    def __init__(self, data_path):
        print("获取数据，开始时间:", datetime.datetime.now().isoformat())
        self.path = os.path.dirname(__file__)


        # 获取数据, 数据来源于data_path
        self.df = pd.read_csv(data_path).dropna()
        self.data = self.df[['question1', 'question2']].values
        self.label = self.df[['is_duplicate']].values

        self.train_data, self.test_data, self.train_label, self.test_label = train_test_split(self.data,
                                                                                              self.label,
                                                                                              random_state=0,
                                                                                              test_size=0.01)
        # print(datetime.datetime.now().isoformat())
        # print("当前文件路径 :", self.path)
        # print("self.data.shape :", self.data.shape)
        # print("self.label.shape :", self.label.shape)
        # print("self.train_data.shape :", self.train_data.shape)
        # print("self.test_data.shape :", self.test_data.shape)
        # print("self.train_label.shape :", self.test_label.shape)
        # print("self.test_label.shape :", self.test_feature.shape)


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
        if not os.path.exists(os.path.join(self.path, "./data/pkl/test.pkl"))\
           or not os.path.exists(os.path.join(self.path, "./data/pkl/train.pkl")) \
           or not os.path.exists(os.path.join(self.path, "./data/pkl/vocab.model")):
            x_text = np.append(self.data, self.test_data).reshape(2 * len(self.data) +\
                                                                  2 * len(self.test_data))

            # reshape 数据
            self.train_data = self.train_data.reshape(2 * len(self.train_data))
            self.test_data = self.test_data.reshape(2 * len(self.test_data))

            # 清洗数据
            self.train_data = [self.text_to_wordlist(line) for line in self.train_data]
            self.test_data = [self.text_to_wordlist(line) for line in self.test_data]

            # 转化成数据，将词汇进行编号
            vocab_processor = learn.preprocessing.VocabularyProcessor(70, min_frequency=5)
            vocab_processor = vocab_processor.fit(self.train_data)
            print("vocab_processor 训练结束")

            # 训练数据和测试数据进行编号
            self.vec_train = list(vocab_processor.transform(self.train_data))
            self.vec_test = list(vocab_processor.transform(self.test_data))

            # 编号
            self.vec_train = [(self.vec_train[index], self.vec_train[index + 1])\
                              for index in range(0, len(self.vec_train), 2)]
            self.vec_test = [(self.vec_test[index], self.vec_test[index + 1])\
                             for index in range(0, len(self.vec_test), 2)]
            print("vocab_processor 转化结束", "number of words :", len(vocab_processor.vocabulary_))

            # 保存vocab_processor转换模型
            pickle.dump(obj=vocab_processor, file=open("./data/pkl/vocab.model", "wb"))

            # 保存转换之后的数据
            pickle.dump(obj=(self.vec_train, self.label), file=open("./data/pkl/train.pkl", "wb"))
            pickle.dump(obj=(self.vec_test, self.test_label), file=open("./data/pkl/test.pkl", "wb"))
            print("dump 结束")
        else:
            self.vec_train, self.label = pickle.load(open("./data/pkl/train.pkl", "rb"))
            self.vec_test, self.test_label = pickle.load(open("./data/pkl/test.pkl", "rb"))
        return self

    @staticmethod
    def get_batch(epoches, batch_size, data, label):
        data = list(zip(data, label))
        for epoch in range(epoches):
            random.shuffle(data)
            for batch in range(0, len(data), batch_size):
                if batch + batch_size >= len(data):
                    yield data[batch: len(data)]
                else:
                    yield data[batch: (batch + batch_size)]


if __name__ == '__main__':
    data_file = "./data/csv/train.csv"
    Data = data(data_path=data_file)
    Data.get_one_hot()
