import tensorflow as tf


class MatchPyramid(object):
    def __init__(self, sequence_length, vocab_size, embedding_size, filter_sizes, batch_size):
        self.label = tf.placeholder(tf.float32, [batch_size, ], name="label")
        self.input_sentence_a = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_a")
        self.input_sentence_b = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_b")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.labels = tf.concat((tf.expand_dims(1 - self.label, axis=-1),
                                 tf.expand_dims(self.label, axis=-1)), axis=1)

        with tf.name_scope("embedding_layer"):
            self.W = tf.Variable(tf.truncated_normal(shape=[vocab_size, embedding_size],
                                                     dtype=tf.float32,
                                                     stddev=0.1,
                                                     mean=0.0), name="W")
            self.embedded_a = tf.nn.embedding_lookup(params=self.W, ids=self.input_sentence_a)
            self.embedded_b = tf.nn.embedding_lookup(params=self.W, ids=self.input_sentence_b)

            # 构建相似性矩阵，并且使用CNN对齐分类
            self.picture = tf.matmul(self.embedded_a, self.embedded_b, transpose_b=True)
            self.picture = tf.expand_dims(self.picture, axis=-1)
            print(self.picture)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-max-pool-%s" % filter_size):
                filter_shape = [filter_size, filter_size, 1, 1]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                conv = tf.nn.conv2d(input=self.picture,
                                    filter=W,
                                    strides=[1, 1, 1, 1],
                                    padding="SAME")

                pooled = tf.nn.max_pool(
                    value=conv,
                    ksize=[1, 4, 4, 1],
                    strides=[1, 4, 4, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(tf.layers.flatten(tf.squeeze(pooled, axis=3)))
                print(conv, pooled)

        self.h_pool = tf.concat(pooled_outputs, 1)
        print(self.h_pool)

        # 句子的特征向量表示
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool, self.dropout_keep_prob)

        with tf.name_scope("full_connected_layer"):
            full_connected_layer_size = int((sequence_length / 4)) * int((sequence_length / 4)) * len(filter_sizes)
            W = tf.get_variable(
                name="W",
                shape=[full_connected_layer_size, 2],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b)

        with tf.name_scope("loss"):
            self.real = tf.argmax(self.labels, axis=1, name="real_label")
            self.predictions = tf.argmax(self.logits, axis=1, name="predictions")

            losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits)
            self.losses = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.real)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


if __name__ == '__main__':
    Model = MatchPyramid(sequence_length=70,
                         vocab_size=1000,
                         embedding_size=50,
                         filter_sizes=[1, 2, 3, 4, 5],
                         batch_size=100)
