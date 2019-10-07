#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn


class SiameseAttentionLSTM(object):
    def __init__(self, num_layers, seq_length, embedding_size, vocab_size, rnn_size, attention_size):
        # 输入数据以及数据标签
        self.labels = tf.placeholder(tf.float32, [None, 2], name="label")
        self.input_sentence_a = tf.placeholder(tf.int32, [None, seq_length], name="input_a")
        self.input_sentence_b = tf.placeholder(tf.int32, [None, seq_length], name="input_b")
        self.length_sentence_a = tf.placeholder(tf.int32, [None, ], name="length_sentence_a")
        self.length_sentence_b = tf.placeholder(tf.int32, [None, ], name="length_sentence_b")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.variable_scope('Embedding'):
            # W : 词表（embedding 向量），后面用来训练.
            w = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1), name="W")
            self.embedded_a = tf.nn.embedding_lookup(w, self.input_sentence_a)
            self.embedded_b = tf.nn.embedding_lookup(w, self.input_sentence_b)

            # self.inputs_a = tf.unstack(embedded_a, axis=1)
            # self.inputs_b = tf.unstack(embedded_b, axis=1)

        # outputs是最后一层每个节点的输出
        # last_state是每层最后一个节点的输出。
        with tf.variable_scope("RNN"):
            with tf.variable_scope("Rnn_Forward"):
                stacked_rnn_fw = []
                for _ in range(num_layers):
                    fw_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
                    stacked_rnn_fw.append(fw_cell)
                self.lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

            with tf.variable_scope("Rnn_backward"):
                stacked_rnn_bw = []
                for _ in range(num_layers):
                    bw_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
                    stacked_rnn_bw.append(bw_cell)
                self.lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)

            with tf.variable_scope("BiLSTM_OUTPUT"):
                self.output1, self.states1 = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.lstm_fw_cell_m,
                    cell_bw=self.lstm_bw_cell_m,
                    inputs=self.embedded_a,
                    sequence_length=self.length_sentence_a,
                    dtype=tf.float32
                )

                self.output2, self.states2 = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.lstm_fw_cell_m,
                    cell_bw=self.lstm_bw_cell_m,
                    inputs=self.embedded_b,
                    sequence_length=self.length_sentence_b,
                    dtype=tf.float32
                )
                self.out1 = tf.concat(self.output1, axis=2)
                self.out2 = tf.concat(self.output2, axis=2)
                print(self.out1, self.out2)

        with tf.variable_scope("Attention"):
            def attention(attention_inputs):
                W_omega = tf.Variable(tf.random_normal(shape=[rnn_size * 2, attention_size], stddev=0.1, dtype=tf.float32))
                b_omega = tf.Variable(tf.random_normal(shape=[attention_size], stddev=0.1, dtype=tf.float32))
                u_omega = tf.Variable(tf.random_normal(shape=[attention_size], stddev=0.1, dtype=tf.float32))
                input_reshape = tf.reshape(attention_inputs, [-1, rnn_size * 2])
                v = tf.tanh(tf.matmul(input_reshape, W_omega) + tf.reshape(b_omega, [1, -1]))
                exps = tf.reshape(tf.exp(tf.matmul(v, tf.reshape(u_omega, [-1, 1]))), shape=[-1, seq_length])
                attention_alpha = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
                atten_outs = tf.reduce_sum(attention_inputs * tf.reshape(attention_alpha, [-1, seq_length, 1]), 1)
                return atten_outs, attention_alpha

            self.out_attention_a, _ = attention(self.out1)
            self.out_attention_b, _ = attention(self.out2)
            print(self.out_attention_a)
            print(self.out_attention_b)

        with tf.variable_scope("FNN"):
            self.diff = self.out_attention_a - self.out_attention_b
            self.mul = tf.multiply(self.out_attention_a, self.out_attention_b)

            self.feature = tf.concat([self.diff, self.mul, self.out_attention_a, self.out_attention_b], axis=1)
            self.weight = tf.Variable(tf.truncated_normal(shape=[rnn_size * 8, 2],
                                                          stddev=0.1,
                                                          mean=0.0))
            self.bias = tf.Variable(tf.truncated_normal(shape=[2], stddev=0.1, mean=0.0))
            self.logits = tf.nn.xw_plus_b(self.feature, self.weight, self.bias)

        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels,
                                                                     logits=self.logits)
            self.loss = tf.reduce_mean(self.losses)

        with tf.name_scope("accuracy"):
            self.predict = tf.argmax(self.logits, axis=1)
            self.equal = tf.equal(self.predict, tf.argmax(self.labels, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(self.equal, tf.float32))


if __name__ == '__main__':
    AttentionLstm = SiameseAttentionLSTM(
        num_layers=1,
        seq_length=70,
        embedding_size=128,
        vocab_size=20005,
        rnn_size=128,
        attention_size=128)

