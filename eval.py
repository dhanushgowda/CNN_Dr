import tensorflow as tf
import time
import numpy as np
from tensorflow.models.image.cifar10 import cifar10
from tensorflow.models.image.cifar10.cifar10 import MOVING_AVERAGE_DECAY

import graph
import input

BATCH_SIZE = 1  # 100
# TODO:FIND TRAINING SET SIZE
DS_SIZE = 4000  # 49000
N_EPOCH = 25


#
# BATCH_SIZE = 2  # 100
# DS_SIZE = 6    # 49000
# N_EPOCH = 25
def do_eval(sess, eval_correct, images_pl, labels_pl, data_set):
    true_count = 0
    steps_per_epoch = data_set.num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE

    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_pl, labels_pl)
        true_count += sess.run(eval_correct, feed_dict)
    accuracy = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Accuracy @ 1: %0.04f' %
          (num_examples, true_count, accuracy))


def fill_feed_dict(data_set, images_pl, labels_pl):
    images, labels = data_set.next_batch(BATCH_SIZE)
    # print images[0].shape
    return {
        images_pl: images,
        labels_pl: labels
    }


def run_training(data):
    ret = []
    with tf.Graph().as_default():
        images_pl = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 512, 512, 3])
        labels_pl = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

        logits = graph.inference(images_pl)
        # loss = graph.loss(logits, labels_pl)
        # train_op = graph.train(loss, 0.0001)
        # eval_correct = graph.evaluate(logits, labels_pl)

        # summary_op = tf.merge_all_summaries()

        # variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, name='avg')
        # variable_averages = variable_averages.variables_to_restore()

        saver = tf.train.Saver(tf.trainable_variables())
        init = tf.initialize_all_variables()
        sess = tf.Session()
        saver.restore(sess, "model.ckpt")
        print("Model restored.")
        sess.run(init)

        # summary_writer = tf.train.SummaryWriter("summary", sess.graph)
        images, labels = (data.images, data.labels)
        for example in range(data.num_examples):
            # start_time = time.time()
            # images, labels = data.next_batch(BATCH_SIZE)

            # print images[0].shape
            d =  {
                images_pl: [images[example]],
                labels_pl: [labels[example]]
            }
            # feed_dict = fill_feed_dict(data, images_pl, labels_pl)
            # feed_dict = {images_pl: data.images}
            logits_op, lab, imgs = sess.run([logits, labels_pl, images_pl], feed_dict=d)
            print logits_op[0], lab, data.labels[example]
            ret.append([logits_op[0], lab])
            # return logits_op, lab
            # break
            # duration = time.time() - start_time

            # assert not np.isnan(loss_val), 'Model diverged with loss = NaN'

            # if step % 10 == 0 or step == N_EPOCH * (DS_SIZE // BATCH_SIZE) - 1:
            #     print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_val, duration))
            #     if step > 0:
            #         summary_str = sess.run(summary_op, feed_dict)
            #         summary_writer.add_summary(summary_str, step)
            #         summary_writer.flush()
            #
            # if step % 100 == 0 or step == N_EPOCH * (DS_SIZE // BATCH_SIZE) - 1:
                #     save_path = saver.save(sess, "model.ckpt")
                #     print("Model saved in file: %s" % save_path)
                #     print('Training Data Eval:')
                # do_eval(sess, eval_correct, images_pl, labels_pl, data.train)
                #
                #     print('Validation Data Eval:')
                #     do_eval(sess, eval_correct, images_pl, labels_pl, data.validation)

        return ret

if tf.gfile.Exists("summary"):
    tf.gfile.DeleteRecursively("summary")
# input.maybe_download_and_extract()
# data = input.get_data()

X_train,  y_train, file_names = input._read_data()
X_train = X_train.reshape(12, 512, 512, 3)
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image.astype(np.uint8)
# X_val -= mean_image.astype(np.uint8)
# X_test -= mean_image.astype(np.uint8)
#
train = input.DataSet(X_train, y_train)
# validation = DataSet(X_val, y_val)
# test = DataSet(X_test, y_test)
#
# return base.Datasets(train=train, validation=validation, test=test)
def begin_here():
    val = run_training(train)
    print val,
    return val, file_names

if __name__=="__main__":
    val, file_names = begin_here()
    print val, file_names
