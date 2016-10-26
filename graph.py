import tensorflow as tf

BATCH_SIZE = 128


def inference(images):
    with tf.variable_scope('conv1') as scope:
        kernel = tf.get_variable("weights", shape=[5, 5, 3, 64],
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d()) # defines weight matrix fr convolution layer
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME') # does the convolution and stores it in conv
        biases = tf.Variable(tf.zeros([64])) # defines biases as 0
        bias = tf.nn.bias_add(conv, biases) # adds bias to conv
        conv1 = tf.nn.relu(bias)  # relu nonlinearity function - like sigmoid or tanh

        tf.histogram_summary("activations", conv1)
        tf.scalar_summary("sparsity", tf.nn.zero_fraction(conv1))
        tf.scalar_summary("conv1_weights", tf.reduce_mean(kernel))

    # pool 1   -  max pooling
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool1 = tf.nn.dropout(pool1, 0.5) # dropout for regularization

    with tf.variable_scope('conv2') as scope:
        kernel = tf.get_variable("weights2", shape=[5, 5, 64, 64],
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.zeros([64]))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias)

        tf.histogram_summary("activations2", conv1)
        tf.scalar_summary("sparsity2", tf.nn.zero_fraction(conv1))
        tf.scalar_summary("conv2_weights", tf.reduce_mean(kernel))

    # pool 2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool2 = tf.nn.dropout(pool2, 0.5)

    # local
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, [BATCH_SIZE, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable("l2weights", [dim, BATCH_SIZE], initializer=tf.contrib.layers.xavier_initializer(),
                                  regularizer=tf.nn.l2_loss)
        # weight_decay = tf.mul(tf.nn.l2_loss(weights), 0.001, name='weight_loss')
        # tf.add_to_collection('losses', weight_decay)
        # weights = tf.Variable(tf.truncated_normal([dim, BATCH_SIZE], stddev=0.04))
        biases = tf.Variable(tf.zeros([BATCH_SIZE]))
        local2 = tf.nn.relu(tf.matmul(reshape, weights) + biases)
        tf.scalar_summary("local2_weights", tf.reduce_mean(weights))

    # softmax
    with tf.variable_scope('softmax') as scope:
        # weights = tf.get_variable("l2weights", [BATCH_SIZE, 10], initializer=tf.contrib.layers.xavier_initializer())
        weights = tf.Variable(tf.truncated_normal([BATCH_SIZE, 10], stddev=0.04))

        biases = tf.Variable(tf.zeros([10]))
        softmax_linear = tf.matmul(local2, weights) + biases
        tf.scalar_summary("softmax_weights", tf.reduce_mean(weights))

    return softmax_linear

# generate graphs
def _add_loss_summaries(total_loss):
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op

# calculate cross-entropy loss
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
    xentropy_mean = tf.reduce_mean(xentropy, name='xentropy_mean')
    tf.add_to_collection('losses', xentropy_mean)
    # tf.scalar_summary("losses", labels)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(total_loss, learning_rate):
    _add_loss_summaries(total_loss)
    optim = tf.train.AdamOptimizer(learning_rate) # adam optimizer is an improvement on graadient descent
    global_step = tf.Variable(0, name='global_step', trainable=False)
    return optim.minimize(total_loss, global_step=global_step)

# calculate accuracy
def evaluate(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
