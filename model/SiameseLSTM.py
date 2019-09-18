#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn


class SiameseLSTM(object):
    def lstm(self, input, rnn_size, num_layers, scope):
        stack_lstm = []
        for _ in range(num_layers):
            cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
            # stm_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
            stack_lstm.append(cell)
        lstm_cell_m = rnn.MultiRNNCell(stack_lstm, state_is_tuple=True)

        outputs, _ = tf.nn.static_rnn(lstm_cell_m, input, dtype=tf.float32, scope=scope)
        return outputs[-1]

    def __init__(self, num_layers, seq_length, embedding_size, vocab_size, rnn_size):
        # 输入数据以及数据标签
        self.label = tf.placeholder(tf.float32, [None, ], name="label")
        self.labels = tf.concat((tf.expand_dims(1 - self.label, axis=-1),
                                 tf.expand_dims(self.label, axis=-1)), axis=1)
        self.input_sentence_a = tf.placeholder(tf.int32, [None, seq_length], name="input_a")
        self.input_sentence_b = tf.placeholder(tf.int32, [None, seq_length], name="input_b")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope('embeddingLayer'):
            # W : 词表（embedding 向量），后面用来训练.
            w = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1), name="W")
            embedded_a = tf.nn.embedding_lookup(w, self.input_sentence_a)
            embedded_b = tf.nn.embedding_lookup(w, self.input_sentence_b)

            inputs_a = tf.unstack(embedded_a, axis=1)
            inputs_b = tf.unstack(embedded_b, axis=1)

        # outputs是最后一层每个节点的输出
        # last_state是每层最后一个节点的输出。
        with tf.name_scope("output"):
            # lstm 共享权重
            self.out1 = self.lstm(inputs_a, rnn_size, num_layers, scope="a")
            self.out2 = self.lstm(inputs_b, rnn_size, num_layers, scope="b")

        with tf.name_scope("scope"):
            self.diff = self.out1 - self.out2
            self.mul = tf.multiply(self.out1, self.out1)

            self.feature = tf.concat([self.diff, self.mul, self.out1, self.out1], axis=1)
            self.weight = tf.Variable(tf.truncated_normal(shape=[rnn_size * 4, 2],
                                                          stddev=0.1,
                                                          mean=0.0))
            self.bias = tf.Variable(tf.truncated_normal(shape=[2], stddev=0.1, mean=0.0))
            self.result = tf.nn.xw_plus_b(self.feature, self.weight, self.bias)
            self.logits = tf.nn.softmax(self.result, axis=1)

        print(self.logits, self.labels)

        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels,
                                                                     logits=self.logits)
            self.loss = tf.reduce_mean(self.losses)

        with tf.name_scope("accuracy"):
            self.predict = tf.argmax(self.logits, axis=1)
            self.equal = tf.equal(self.predict, tf.argmax(self.labels, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(self.equal, tf.float32))


if __name__ == '__main__':
    lstm = SiameseLSTM(seq_length=60,
                       vocab_size=73300,
                       embedding_size=128,
                       rnn_size=128,
                       num_layers=1)

