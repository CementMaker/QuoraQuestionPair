'''
Created on August 1, 2018
@author : hsiaoyetgun (yqxiao)
Reference : A Decomposable Attention Model for Natural Language Inference (EMNLP 2016)
'''

# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Decomposable(object):
    # 前馈神经网络，前向传播词向量
    def _feedForwardBlock(self, inputs, num_units, scope, isReuse=False, initializer=None):
        with tf.variable_scope(scope, reuse = isReuse):
            # if initializer is None:
            #     initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope('feed_foward_layer1'):
                inputs = tf.nn.dropout(inputs, self.dropout_keep_prob)
                # 全连接层, 相当于添加一个层
                outputs = tf.layers.dense(inputs, num_units, tf.nn.relu, kernel_initializer=initializer)

            with tf.variable_scope('feed_foward_layer2'):
                outputs = tf.nn.dropout(outputs, self.dropout_keep_prob)
                # 全连接层, 相当于添加一个层
                resluts = tf.layers.dense(outputs, num_units, tf.nn.relu, kernel_initializer=initializer)
                return resluts

    def __init__(self, seq_length, n_vocab, embedding_size, hidden_size, n_classes):
        # model init
        self.premise = tf.placeholder(tf.int32, [None, seq_length], 'premise')
        self.hypothesis = tf.placeholder(tf.int32, [None, seq_length], 'hypothesis')
        self.y = tf.placeholder(tf.float32, [None, n_classes], 'y_true')

        # premise的长度标记，[1,...,1,0,...0]
        self.premise_mask = tf.placeholder(tf.float32, [None, seq_length], 'premise_mask')
        self.hypothesis_mask = tf.placeholder(tf.float32, [None, seq_length], 'hypothesis_mask')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.variable_scope('embedding'):
            self.Embedding = tf.get_variable('Embedding', [n_vocab, embedding_size], tf.float32)
            self.embeded_left = tf.nn.embedding_lookup(self.Embedding, self.premise)
            self.embeded_right = tf.nn.embedding_lookup(self.Embedding, self.hypothesis)

        with tf.variable_scope('attend'):
            self.FNN_left = self._feedForwardBlock(self.embeded_left, hidden_size, 'F')
            self.FNN_right = self._feedForwardBlock(self.embeded_right, hidden_size, 'F', isReuse=True)

            # 相似度矩阵，矩阵元素为词向量点乘
            self.similarity_matrix = tf.matmul(self.FNN_left,
                                               tf.transpose(self.FNN_right, [0, 2, 1]),
                                               name='similarity_matrix')

            # 没有的词向量点乘之后的mask
            self.mask = tf.multiply(tf.expand_dims(self.premise_mask, 2),
                                    tf.expand_dims(self.hypothesis_mask, 1),
                                    name='mask')

            # 将相似度矩阵中没有的词向量点乘之后的值设为0
            self.e = tf.multiply(self.similarity_matrix, self.mask, name='e')

            self.attentionSoft_a = tf.exp(self.e - tf.reduce_max(self.e, axis=2, keepdims=True))
            print("self.attentionSoft_a: ", self.attentionSoft_a)
            print("tf.reduce_max(self.e, axis=2, keepdims=True) : ", tf.reduce_max(self.e, axis=2, keepdims=True))
            self.attentionSoft_a = tf.multiply(self.attentionSoft_a, tf.expand_dims(self.hypothesis_mask, 1))
            self.attentionSoft_a = tf.divide(self.attentionSoft_a, tf.reduce_sum(self.attentionSoft_a, axis=2, keepdims=True))
            self.attentionSoft_a = tf.multiply(self.attentionSoft_a, self.mask)

            self.attentionSoft_b = tf.exp(self.e - tf.reduce_max(self.e, axis=1, keepdims=True))
            self.attentionSoft_b = tf.multiply(self.attentionSoft_b, tf.expand_dims(self.premise_mask, 2))
            self.attentionSoft_b = tf.divide(self.attentionSoft_b, tf.reduce_sum(self.attentionSoft_b, axis=1, keepdims=True))
            self.attentionSoft_b = tf.transpose(tf.multiply(self.attentionSoft_b, self.mask), [0, 2, 1])

            print('att_soft_a', self.attentionSoft_a)
            print('att_soft_b', self.attentionSoft_b)

        with tf.variable_scope('AGGREATE'):
            self.beta = tf.matmul(self.attentionSoft_b, self.embeded_left)
            self.alpha = tf.matmul(self.attentionSoft_a, self.embeded_right)
            print('alpha', self.alpha)
            print('beta', self.beta)

            self.a_beta = tf.concat([self.embeded_left, self.beta], axis=2)
            self.b_alpha = tf.concat([self.embeded_right, self.alpha], axis=2)
            print('a_beta', self.a_beta)
            print('b_alpha', self.b_alpha)

        with tf.variable_scope('FNN'):
            self.v_1 = self._feedForwardBlock(self.a_beta, hidden_size, 'G')
            self.v_2 = self._feedForwardBlock(self.b_alpha, hidden_size, 'G', isReuse=True)
            self.v1_sum = tf.reduce_sum(self.v_1, axis=1)
            self.v2_sum = tf.reduce_sum(self.v_2, axis=1)
            self.v = tf.concat([self.v1_sum, self.v2_sum], axis=1)
            self.ff_outputs = self._feedForwardBlock(self.v, hidden_size, 'H')
            # self.y_hat = tf.clip_by_value(tf.layers.dense(self.ff_outputs, n_classes), 1e-10, 1)
            self.y_hat = tf.layers.dense(self.ff_outputs, n_classes)
            print('y_hat', self.y_hat)

        with tf.variable_scope('loss'):
            self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.y_hat)
            self.loss = tf.reduce_mean(self.losses, name='loss_val')

        with tf.variable_scope('accuracy'):
            self.label_pred = tf.argmax(self.y_hat, 1, name='label_pred')
            self.label_true = tf.argmax(self.y, 1, name='label_true')
            self.correct_pred = tf.equal(tf.cast(self.label_pred, tf.int32), tf.cast(self.label_true, tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name='Accuracy')


if __name__ == '__main__':
    decomposable_model = Decomposable(
        seq_length=10,
        n_vocab=20,
        embedding_size=30,
        hidden_size=40,
        n_classes=2)
