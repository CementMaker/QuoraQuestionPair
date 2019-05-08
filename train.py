#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from PreProcess import data
from sklearn.metrics import classification_report

from model.SiameseCNN import SiameseCNN
from model.SiameseLSTM import SiameseLSTM
from model.SiameseBiLSTM import SiameseBiLSTM

class function(object):
    def __init__(self):
        pass

    @staticmethod
    def dev_data(x_test, y_test):
        test_a = [a for (a, b) in x_test]
        test_b = [b for (a, b) in x_test]
        y_test = np.squeeze(y_test, 1)
        return test_a, test_b, y_test

    @staticmethod
    def test_result(sess, model, df, ans):
        vocab_model = pickle.load(open("./data/vocab.model", "rb"))
        batch_a = list(vocab_model.transform([data.text_to_wordlist(text) for text in df['question1'].values]))
        batch_b = list(vocab_model.transform([data.text_to_wordlist(text) for text in df['question2'].values]))

        batch_y = df['test_id'].values
        feed_dict = {
            model.input_sentence_a: batch_a,
            model.input_sentence_b: batch_b,
            model.label: batch_y,
            model.dropout_keep_prob: 0.5
        }

        answer = sess.run([model.ans], feed_dict=feed_dict)
        print(np.squeeze(np.array(answer), [0]).shape)
        return np.append(ans, answer)


class siamese_cnn_train(object):
    def __init__(self):
        # 定义CNN网络，对话窗口以及optimizer
        # sequence_length, vocab_size, embedding_size, filter_sizes, num_filters
        self.sess = tf.Session()
        self.Model = SiameseCNN(sequence_length=70,
                                vocab_size=33300,
                                embedding_size=200,
                                filter_sizes=[1, 2, 3, 4, 5, 6],
                                num_filters=5)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.Model.loss, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())

        self.data = data(data_path='./data/csv/train.csv').get_one_hot()
        self.train_data, self.train_label = self.data.vec_train, self.data.train_label
        self.test_data, self.test_label = self.data.vec_test, self.data.test_label
        # self.test_feature, self.train_feature = self.data.test_feature, self.data.train_feature

        print("*****************************************************************************")
        print("self.train_data.shape :", np.array(self.train_data).shape)
        print("self.train_label.shape :", np.array(self.train_label).shape)
        print("self.test_data.shape :", np.array(self.test_data).shape)
        print("self.test_label.shape :", np.array(self.test_label).shape)
        print("*****************************************************************************")

        # 获取训练数据迭代器并且获取测试数据（用于神经网络验证）
        self.batches = data.get_batch(1, 500, self.train_data, self.train_label)
        self.test_a, self.test_b, self.y_test = function.dev_data(self.test_data, self.test_label)

        # 定义数据变量，用来记录网络训练过程中的数据
        # self.ans = []
        # self.train_loss, self.train_accuracy = [], []
        # self.test_loss, self.test_accuracy = [], []

        # tensorboard
        tf.summary.scalar("loss", self.Model.loss)
        tf.summary.scalar("accuracy", self.Model.accuracy)
        self.merged_summary_op_train = tf.summary.merge_all()
        self.merged_summary_op_test = tf.summary.merge_all()
        self.summary_writer_train = tf.summary.FileWriter("./summary/SiameseCNN/train", graph=self.sess.graph)
        self.summary_writer_test = tf.summary.FileWriter("./summary/SiameseCNN/test", graph=self.sess.graph)

    def train_step(self, a_batch, b_batch, label):
        '''
        神经网路的训练过程
        :param a_batch: Siamese网络左边的输入
        :param b_batch: Siamese网络右边的输入
        :param label: 标签
        :return: 训练网络，没有返回值
        '''
        feed_dict = {
            self.Model.input_sentence_a: a_batch,
            self.Model.input_sentence_b: b_batch,
            self.Model.dropout_keep_prob: 1.0,
            self.Model.label: label
        }
        _, summary, step, loss, accuracy = self.sess.run(
            fetches=[self.optimizer, self.merged_summary_op_train, self.global_step,
                     self.Model.loss, self.Model.accuracy],
            feed_dict=feed_dict)
        self.summary_writer_train.add_summary(summary, step)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, accuracy {}".format(time_str, step, loss, accuracy))

    def dev_step(self, a_batch, b_batch, label):
        '''
        神经网路的验证过程，查看网络是否收敛
        :param a_batch: Siamese网络左边的输入
        :param b_batch: Siamese网络右边的输入
        :param label: 标签
        :return: 验证网络，没有返回值
        '''

        feed_dict = {
            self.Model.input_sentence_a: a_batch,
            self.Model.input_sentence_b: b_batch,
            self.Model.dropout_keep_prob: 1.0,
            self.Model.label: label
        }
        loss, summary, step, acc, predict = self.sess.run(
            fetches=[self.Model.loss, self.merged_summary_op_test,
                     self.global_step, self.Model.accuracy, self.Model.predict],
            feed_dict=feed_dict
        )
        self.summary_writer_test.add_summary(summary, step)
        print(classification_report(y_true=label, y_pred=predict))
        print("\n**********\tlog_loss：", loss, "**********\n")

    def main(self, flag=False):
        ''' 神经网络的入口，整个网络的运行过程 '''
        for batch in self.batches:
            x, y = zip(*batch)
            batch_a = np.array([a for (a, b) in x])
            batch_b = np.array([b for (a, b) in x])
            self.train_step(batch_a, batch_b, np.squeeze(y, axis=1))
            current_step = tf.train.global_step(self.sess, self.global_step)

            if current_step % 50 == 0:
                print("\nEvaluation:")
                self.dev_step(self.test_a, self.test_b, self.y_test)
                print("")

        print("Run the command line:\n"
              "--> tensorboard --logdir=summary"
              "\nThen open http://0.0.0.0:6006/ into your web browser")


if __name__ == '__main__':
    Net = siamese_cnn_train()
    Net.main()

