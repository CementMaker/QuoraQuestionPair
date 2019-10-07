import tensorflow as tf


class Bilstm(object):
    def __init__(self, embedding_size, vocab_size, rnn_size, max_length):
        # 输入数据以及数据标签
        self.label = tf.placeholder(tf.float32, [None, 2], name="label")
        self.input_sentence_a = tf.placeholder(tf.int32, [None, max_length], name="input_a")
        self.sequence_length_a = tf.placeholder(tf.int32, [None, ], name="sequence_length_a")
        self.input_sentence_b = tf.placeholder(tf.int32, [None, max_length], name="input_b")
        self.sequence_length_b = tf.placeholder(tf.int32, [None, ], name="sequence_length_a")

        with tf.name_scope('embeddingLayer'):
            W = tf.Variable(tf.truncated_normal(shape=[vocab_size, embedding_size]))
            self.embedding_a = tf.nn.embedding_lookup(W, self.input_sentence_a)
            self.embedding_b = tf.nn.embedding_lookup(W, self.input_sentence_b)

        with tf.name_scope('lstm_layer'):
            # 使用 MultiRNNCell 类实现深层循环网络中每一个时刻的前向传播过程
            with tf.variable_scope('lstm_input_a'):
                self.lstm_forward_cell = tf.nn.rnn_cell.LSTMCell(rnn_size)
                self.lstm_backward_cell = tf.nn.rnn_cell.LSTMCell(rnn_size)

                self.outputs_a, _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.lstm_forward_cell,
                    cell_bw=self.lstm_backward_cell,
                    inputs=self.embedding_a,
                    sequence_length=self.sequence_length_a,
                    dtype=tf.float32
                )
                self.outputs_a = tf.concat(self.outputs_a, 2)

                self.outputs_b, _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.lstm_forward_cell,
                    cell_bw=self.lstm_backward_cell,
                    inputs=self.embedding_b,
                    sequence_length=self.sequence_length_b,
                    dtype=tf.float32
                )
                self.outputs_b = tf.concat(self.outputs_b, 2)
            print(self.outputs_a)
            print(self.outputs_b)

        with tf.name_scope('softmaxLayer'):
            self.outputs = tf.concat((self.outputs_a, self.outputs_b), axis=2)
            self.feature = tf.unstack(self.outputs, axis=1)[-1]
            self.weight = tf.Variable(tf.truncated_normal(shape=[rnn_size * 4, 2],
                                                          stddev=0.1,
                                                          mean=0.0))
            self.bias = tf.Variable(tf.truncated_normal(shape=[2], stddev=0.1, mean=0.0))
            self.logits = tf.nn.xw_plus_b(self.feature, self.weight, self.bias)

            print(self.outputs_a)
            print(self.outputs_b)
            print(self.logits)

        # 损失函数，采用softmax交叉熵函数
        with tf.name_scope('loss'):
            self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.label)
            self.loss = tf.reduce_mean(self.losses)

        with tf.name_scope('predict'):
            self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            self.predict = tf.argmax(self.logits, axis=1)


# model = Bilstm(embedding_size=100,
#                vocab_size=20005,
#                rnn_size=128,
#                max_length=50)
