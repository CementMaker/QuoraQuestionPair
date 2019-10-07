#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import pickle
import datetime
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from PreProcess import data
from sklearn.metrics import classification_report
from model.SiameseDynamicRnn import Bilstm
from model.DecomposableAttentionBase import Decomposable

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s')
logger = logging.getLogger(__name__)


def dev_data(x_test, y_test):
    test_a = [a for (a, b) in x_test]
    test_b = [b for (a, b) in x_test]
    return test_a, test_b, y_test


def get_batch(epoches, batch_size, data, label, length, mask_a, mask_b):
    data = list(zip(data, label, length, mask_a, mask_b))
    for epoch in range(epoches):
        random.shuffle(data)
        for batch in range(0, len(data), batch_size):
            if batch + batch_size >= len(data):
                yield data[batch: len(data)]
            else:
                yield data[batch: (batch + batch_size)]


class DecomposableTrain(object):
    def __init__(self):
        # 定义CNN网络，对话窗口以及optimizer
        self.sess = tf.Session()
        self.Model = Decomposable(seq_length=70,
                                  n_vocab=20005,
                                  embedding_size=128,
                                  hidden_size=128,
                                  n_classes=2)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.Model.loss, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())

        self.train_data, self.train_label = pickle.load(open("./data/pkl/train.pkl", "rb"))
        self.test_data, self.test_label = pickle.load(open("./data/pkl/test.pkl", "rb"))
        self.train_data_length, self.test_data_length = pickle.load(open("./data/pkl/length.pkl", "rb"))

        lambda_fuction = lambda lena, lenb: [[1] * lena + [0] * (70 - lena), [1] * lenb + [0] * (70 - lenb)]
        self.test_data_mask = [lambda_fuction(lena, lenb) for lena, lenb in self.test_data_length]
        self.train_data_mask = [lambda_fuction(lena, lenb) for lena, lenb in self.train_data_length]
        self.train_data_mask_a = np.array([sample[0] for sample in self.train_data_mask])
        self.train_data_mask_b = np.array([sample[1] for sample in self.train_data_mask])

        logger.info('self.train_data shape: (%d, %d, %d)' % (self.train_data.shape))
        logger.info('self.train_label shape: (%d, %d)' % (self.train_label.shape))
        logger.info('self.test_data shape: (%d, %d, %d)' % (self.test_data.shape))
        logger.info('self.test_label shape: (%d, %d)' % (self.test_label.shape))
        logger.info('self.train_data_length shape: (%d, %d)' % (self.train_data_length.shape))
        logger.info('self.test_data_length shape: (%d, %d)' % (self.test_data_length.shape))
        logger.info('self.train_data_mask_a shape: (%d, %d)' % (self.train_data_mask_a.shape))
        logger.info('self.train_data_mask_b shape: (%d, %d)' % (self.train_data_mask_b.shape))

        # 获取训练数据迭代器并且获取测试数据（用于神经网络验证）
        self.batches = get_batch(1, 100, self.train_data, self.train_label,
                                 self.train_data_length, self.train_data_mask_a, self.train_data_mask_b)
        self.test_a, self.test_b, self.y_test = dev_data(self.test_data, self.test_label)

        # tensorboard
        tf.summary.scalar("loss", self.Model.loss)
        tf.summary.scalar("accuracy", self.Model.accuracy)
        self.merged_summary_op_train = tf.summary.merge_all()
        self.merged_summary_op_test = tf.summary.merge_all()
        self.summary_writer_train = tf.summary.FileWriter("./summary/train", graph=self.sess.graph)
        self.summary_writer_test = tf.summary.FileWriter("./summary/test", graph=self.sess.graph)

    def train_step(self, a_batch, b_batch, label, length_a, length_b, mask_a, mask_b):
        '''
        神经网路的训练过程
        :param a_batch: 网络: input_sentence_a
        :param b_batch: 网络: input_sentence_b
        :param label: 标签
        :return: 训练网络，没有返回值
        '''
        feed_dict = {
            self.Model.premise: a_batch,
            self.Model.hypothesis: b_batch,
            self.Model.premise_mask: mask_a,
            self.Model.hypothesis_mask: mask_b,
            self.Model.y: label,
            self.Model.dropout_keep_prob: 0.5
        }
        _, summary, step, loss, accuracy = self.sess.run(
            fetches=[self.optimizer, self.merged_summary_op_train, self.global_step,
                     self.Model.loss, self.Model.accuracy],
            feed_dict=feed_dict)
        self.summary_writer_train.add_summary(summary, step)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, accuracy {}".format(time_str, step, loss, accuracy))

    def dev_step(self, a_batch, b_batch, label, length_a, length_b, mask_a, mask_b):
        '''
        神经网路的验证过程，查看网络是否收敛
        :param a_batch: 网络: input_sentence_a
        :param b_batch: 网络: input_sentence_b
        :param label: 标签
        :return: 验证网络，没有返回值
        '''

        feed_dict = {
            self.Model.premise: a_batch,
            self.Model.hypothesis: b_batch,
            self.Model.premise_mask: mask_a,
            self.Model.hypothesis_mask: mask_b,
            self.Model.y: label
        }

        loss, summary, step, acc = self.sess.run(
            fetches=[self.Model.loss, self.merged_summary_op_test,
                     self.global_step, self.Model.accuracy],
            feed_dict=feed_dict
        )
        self.summary_writer_test.add_summary(summary, step)
        # print(classification_report(y_true=label, y_pred=predict))

    def main(self):
        ''' 神经网络的入口，整个网络的运行过程 '''
        for batch in self.batches:
            x, y, length, mask_a, mask_b = zip(*batch)
            batch_a = np.array([a for (a, b) in x])
            batch_b = np.array([b for (a, b) in x])
            length_a = np.array([a for (a, b) in length])
            length_b = np.array([b for (a, b) in length])
            self.train_step(batch_a, batch_b, y, length_a, length_b, mask_a, mask_b)
            current_step = tf.train.global_step(self.sess, self.global_step)

            # if current_step % 20 == 0:
            #     print("\nEvaluation:")
            #     self.dev_step(self.test_a, self.test_b, self.y_test, length_a, length_b, )


if __name__ == '__main__':
    Net = DecomposableTrain()
    Net.main()
