import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class SiameseCNN(object):
    def __init__(self, sequence_length, vocab_size, embedding_size, filter_sizes, num_filters):
        self.label = tf.placeholder(tf.float32, [None, ], name="label")
        self.input_sentence_a = tf.placeholder(tf.int32, [None, sequence_length], name="input_a")
        self.input_sentence_b = tf.placeholder(tf.int32, [None, sequence_length], name="input_b")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("embedding_layer"):
            self.W = tf.Variable(tf.truncated_normal(shape=[vocab_size, embedding_size],
                                                     dtype=tf.float32,
                                                     stddev=0.1,
                                                     mean=0.0), name="W")
            self.embedded_a = tf.nn.embedding_lookup(params=self.W, ids=self.input_sentence_a)
            self.embedded_b = tf.nn.embedding_lookup(params=self.W, ids=self.input_sentence_b)
            self.embedded_a_expand = tf.expand_dims(input=self.embedded_a, axis=-1)
            self.embedded_b_expand = tf.expand_dims(input=self.embedded_b, axis=-1)

        pooled_outputs_a = []
        pooled_outputs_b = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % i):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                w = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1, mean=0.0), name="W")

                conv_a = tf.nn.conv2d(input=self.embedded_a_expand,
                                      filter=w,
                                      strides=[1, 1, 1, 1],
                                      padding="VALID")
                pooled_a = tf.nn.max_pool(value=conv_a,
                                          ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                          strides=[1, 1, 1, 1],
                                          padding='VALID')

                conv_b = tf.nn.conv2d(input=self.embedded_b_expand,
                                      filter=w,
                                      strides=[1, 1, 1, 1],
                                      padding="VALID")
                pooled_b = tf.nn.max_pool(value=conv_b,
                                          ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                          strides=[1, 1, 1, 1],
                                          padding='VALID')
                relu_a, relu_b = tf.nn.relu(pooled_a), tf.nn.relu(pooled_b)
                pooled_outputs_a.append(relu_a)
                pooled_outputs_b.append(relu_b)

        with tf.name_scope("result"):
            self.h_pool_a = tf.concat(pooled_outputs_a, 3)
            self.h_pool_b = tf.concat(pooled_outputs_b, 3)
            self.h_pool_flat_a = tf.squeeze(self.h_pool_a, axis=[1, 2])
            self.h_pool_flat_b = tf.squeeze(self.h_pool_b, axis=[1, 2])

            self.diff = self.h_pool_flat_a - self.h_pool_flat_b
            self.mul = tf.multiply(self.h_pool_flat_a, self.h_pool_flat_b)

            self.feature = tf.concat(values=[self.diff, self.mul, self.h_pool_flat_a, self.h_pool_flat_b], axis=1)
            self.weight = tf.Variable(tf.truncated_normal(shape=[num_filters * len(filter_sizes) * 4, 2],
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
    cnn = Cnn(sequence_length=35,
              vocab_size=1000,
              embedding_size=50,
              filter_sizes=[1, 2, 3, 4, 5],
              num_filters=20)

