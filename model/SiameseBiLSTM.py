import tensorflow as tf
import numpy as np


class SiameseBiLSTM(object):
    def BiRNN(self, x, scope, embedding_size, hidden_units, sequence_length):
        n_input, n_steps, n_hidden, n_layers = embedding_size, sequence_length, hidden_units, 3
        x = tf.split(tf.reshape(tf.transpose(x, [1, 0, 2]), [-1, n_input]), n_steps, 0)

        with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                stacked_rnn_fw.append(fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            stacked_rnn_bw = []
            for _ in range(n_layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                stacked_rnn_bw.append(bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)

        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell_m,
                                                           lstm_bw_cell_m,
                                                           x,
                                                           dtype=tf.float32)

        # feature = tf.concat(outputs, axis=1)
        return outputs[-1]

    def contrastive_loss(self, y, d, batch_size):
        tmp = y * tf.square(d)
        # tmp= tf.mul(y,tf.square(d))
        tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
        return tf.reduce_sum(tmp + tmp2) / batch_size / 2

    def __init__(self, sequence_length, vocab_size, embedding_size, hidden_units):
        # Placeholders for input, output and dropout
        self.label = tf.placeholder(tf.float32, [None, ], name="label")
        self.input_sentence_a = tf.placeholder(tf.int32, [None, sequence_length], name="input_a")
        self.input_sentence_b = tf.placeholder(tf.int32, [None, sequence_length], name="input_b")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                trainable=True, name="W")
            self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.input_sentence_a)
            self.embedded_chars2 = tf.nn.embedding_lookup(self.W, self.input_sentence_b)

        with tf.name_scope("output"):
            self.feature_a = self.BiRNN(self.embedded_chars1, "side1", embedding_size, hidden_units, sequence_length)
            self.feature_b = self.BiRNN(self.embedded_chars2, "side2", embedding_size, hidden_units, sequence_length)
            print(self.feature_a, self.feature_b)

            self.diff = self.feature_a - self.feature_b
            self.mul = tf.multiply(self.feature_a, self.feature_b)

            self.feature = tf.concat(values=[self.diff, self.mul, self.feature_a, self.feature_b], axis=1)
            self.weight = tf.Variable(tf.truncated_normal(shape=[hidden_units * 2 * 4, 2],
                                                          stddev=0.1,
                                                          mean=0.0))
            self.bias = tf.Variable(tf.truncated_normal(shape=[2], stddev=0.1, mean=0.0))
            self.logits = tf.nn.xw_plus_b(self.feature, self.weight, self.bias)

        with tf.name_scope("loss"):
            self.labels = tf.concat((tf.expand_dims(1 - self.label, axis=-1),
                                     tf.expand_dims(self.label, axis=-1)), axis=1)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels,
                                                                                  logits=self.logits))

        with tf.name_scope("accuracy"):
            self.predict = tf.argmax(self.logits, axis=1)
            self.equal = tf.equal(self.predict, tf.argmax(self.labels, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(self.equal, tf.float32))


if __name__ == '__main__':
    Model = siameseBiLSTM(
        sequence_length=10,
        vocab_size=1000,
        embedding_size=128,
        hidden_units=28
    )
