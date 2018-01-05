import tensorflow as tf
import os


# The network:
# Input nodes:
#   - 9 * 4 holes (big pins)
#     For each hole 6 nodes
#   - 9 times a count of black and a count of white small pins, count in [0,4]
#     2 nodes: one for black, one for white, in that specific order.
#   - So 9*4*6 + 9*2 = 234 input nodes in total
#     The order: 4*6+2 nodes for the first row, 4*6+2 for the second row, and so on.
#       So [row1hole1yellow,row1hole1blue,...,row1hole4black,row1black,row1white, row2hole1yellow, ..., row9white]
# Output nodes:
#   - 4 holes, so 6 * 4 = 24 output nodes
# Hidden layers:
#   - layer 1: 80 nodes
# Placeholders:
#   - x:  dimension None x 234 (input)
#   - y_: dimension None x 4  (correct output, NOT ONE-HOT but arrays of indices.)
# Parameters:
#   - weights_1: dimension 234 x 80
#   - weights_2: dimension 80 x 24
# Activation functions:
#   - tanh
#   - softmax
# Network propagation:
#   - x --> *w_1 --> tanh --> *w_2 --> softmax --> y
# Cost function:
#   - cross entropy
#   - we use the function tf.nn.sparse_softmax_cross_entropy_with_logits because the labels are one-hot.
# Training algorithm:
#   - Stochastic gradient descent
#
# TODO
#   - Add biases?
#   - Hidden layer size tweaken?
#   - AdaGrad ipv SGD?
#
# https://stackoverflow.com/questions/34240703/whats-the-difference-between-softmax-and-softmax-cross-entropy-with-logits
# https://theblog.github.io/post/neural-networks-multiple-class-softmax/
# https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits


class MastermindNN:
    def __init__(self):
        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, 234], name="x")  # Input  TODO dtype tf.int32?
        self.y_ = tf.placeholder(tf.int32, shape=[None, 4], name="y_")  # Correct output (labels)
        # NOTE: The actual output of the network is of shape [None, 24]. But because we are
        # using tf.nn.sparse_softmax_cross_entropy_with_logits, we need the labels in a different shape.
        # We practically have 4 seperate classifiers for each hole. Each label entry is a number (a 0-D Tensor)
        # in [0,6) designating the color of the pin that should be in the hole of that classifier. The function
        # tf.nn.sparse_softmax_cross_entropy_with_logits expects the labels to be of shape [None] (where None =
        # the batch size). Later on, we need to extract the columns of shape [None], we need to transpose the array
        # to shape [4, None] for that.

        # Weights
        self.weights_1 = tf.Variable(tf.random_uniform([234, 80]), name='weights_1')
        self.weights_2 = tf.Variable(tf.random_uniform([80, 24]), name='weights_2')

        # Compute output
        # First compute unprocessed output logits. Dimension of output logits: None x 24
        #   Expanded version:
        #   hidden_layer_1 = tf.tanh(tf.matmul(self.x, self.weights_1))
        #   hidden_layer_2 = tf.tanh(tf.matmul(hidden_layer_1, self.weights_2))
        #   y_logits = tf.matmul(hidden_layer_2, self.weights_3)
        # Compressed version:
        y_logits = tf.matmul(tf.tanh(tf.matmul(self.x, self.weights_1)),
                             self.weights_2, name='y_logits')

        # Split output logits into four classifiers with 6 classes each.
        # y_logits has dimension [None, 24]. Splitting it into four along dimension 1 will give four
        # Tensors of shape [None, 6]. THESE TENSORS ARE NOT YET SOFTMAXED
        prediction0, prediction1, prediction2, prediction3 = tf.split(y_logits, num_or_size_splits=4, axis=1)

        # ---------- The following can be used for making predictions (not while training).

        # Softmax that shit
        self.prediction0_softmaxed = tf.nn.softmax(prediction0)  # Shape [None, 6]
        self.prediction1_softmaxed = tf.nn.softmax(prediction1)
        self.prediction2_softmaxed = tf.nn.softmax(prediction2)
        self.prediction3_softmaxed = tf.nn.softmax(prediction3)
        self.predictions = [tf.argmax(self.prediction0_softmaxed, axis=1),
                            tf.argmax(self.prediction1_softmaxed, axis=1),
                            tf.argmax(self.prediction2_softmaxed, axis=1),
                            tf.argmax(self.prediction3_softmaxed, axis=1)]
        # self.predictions is of shape [4, None] and all its elements are in the range [0,6)
        # indicating the color of the pin.

        # ---------- The following is to be used during training.

        # Let's also split the labels (y_), to be able to compare them to the predictions seperately.
        # Therefore we need to transpose y_ first. The rank of each labels tensor should be 1.
        y__tranpose = tf.transpose(self.y_)
        labels0, labels1, labels2, labels3 = y__tranpose[0], y__tranpose[1], y__tranpose[2], y__tranpose[3]

        # Compute losses for each classifier.
        self.loss0 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels0, logits=prediction0))
        self.loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels1, logits=prediction1))
        self.loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels2, logits=prediction2))
        self.loss3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels3, logits=prediction3))

        # Add to obtain the total loss. (Need to use at least one tf.add function because it lets me provide a name
        # for the Op.)
        self.loss = tf.add(self.loss0 + self.loss1 + self.loss2, self.loss3, name='overall_loss')

        # Create Optimizer.
        self.train = tf.train.GradientDescentOptimizer(.2).minimize(self.loss)

        # Create summaries for TensorBoard
        self.init_summaries()

        self.global_step = tf.Variable(0, trainable=False)

        self.saver = tf.train.Saver()

    # noinspection PyAttributeOutsideInit
    def init_summaries(self):
        with tf.name_scope('summaries'):
            # Losses
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('loss 0', self.loss0)
            tf.summary.scalar('loss 1', self.loss1)
            tf.summary.scalar('loss 2', self.loss2)
            tf.summary.scalar('loss 3', self.loss3)
            # This is the average loss over one eval period during training. Is calculated in main.py
            self.avg_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('avg_loss', self.avg_loss)

            # Weights
            mean1 = tf.reduce_mean(self.weights_1)
            mean2 = tf.reduce_mean(self.weights_2)
            stddev1 = tf.square(tf.reduce_mean(tf.square(self.weights_1 - mean1)))
            stddev2 = tf.square(tf.reduce_mean(tf.square(self.weights_2 - mean2)))
            max1, min1 = tf.reduce_max(self.weights_1), tf.reduce_min(self.weights_1)
            max2, min2 = tf.reduce_max(self.weights_2), tf.reduce_min(self.weights_2)
            tf.summary.scalar('weights_1 mean', mean1)
            tf.summary.scalar('weights_2 mean', mean2)
            tf.summary.scalar('weights_1 stddev', stddev1)
            tf.summary.scalar('weights_2 stddev', stddev2)
            tf.summary.scalar('weights_1 max', max1)
            tf.summary.scalar('weights_2 max', max2)
            tf.summary.scalar('weights_1 min', min1)
            tf.summary.scalar('weights_2 min', min2)

            # Merge summaries and create FileWriter
            self.merged_summaries = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(
                '/home/teinvdlugt/Documents/Computer Science/Machine Learning/Mastermind/logs',
                tf.get_default_graph())

    def predict(self, session, x):
        """ Propagate the input x through the network and return an array containing the next move.
        :param session: The TensorFlow session to use.
        :param x: The input of the board, must be of shape [None, 234] and contain only 1s and 0s.
        :returns: 1-D array of shape [None, 4]. """

        return session.run([self.predictions, self.prediction0_softmaxed,
                            self.prediction1_softmaxed, self.prediction2_softmaxed, self.prediction3_softmaxed],
                           feed_dict={self.x: x})

    def train_step(self, session, inputs, outputs):
        loss, _ = session.run([self.loss, self.train], feed_dict={self.x: inputs, self.y_: outputs})
        return loss

    def write_summary(self, session, x, y_, global_step, avg_loss):
        summary = session.run(self.merged_summaries, feed_dict={self.x: x, self.y_: y_, self.avg_loss: avg_loss})
        self.summary_writer.add_summary(summary, global_step)

    def restore_or_initialize(self, session, checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Restoring variables from', ckpt.model_checkpoint_path)
            self.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint found -- initializing variables.')
            session.run(tf.global_variables_initializer())

    def save(self, session, checkpoint_dir, global_step):
        print("Saving checkpoint...")
        self.saver.save(session, os.path.join(checkpoint_dir, 'model.ckpt'), global_step=global_step)
        print("Checkpoint saved!")
