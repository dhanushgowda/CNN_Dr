import tensorflow as tf
import time
import numpy as np

import graph
import input

BATCH_SIZE = 1  # 100
#TODO:FIND TRAINING SET SIZE
DS_SIZE = 4000    # 49000
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
    with tf.Graph().as_default():
        images_pl = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 512, 512, 3])
        labels_pl = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

        logits = graph.inference(images_pl)

        loss = graph.loss(logits, labels_pl)
        train_op = graph.train(loss, 0.0001)
        eval_correct = graph.evaluate(logits, labels_pl)
        saver = tf.train.Saver(tf.trainable_variables())
        summary_op = tf.merge_all_summaries()

        init = tf.initialize_all_variables()
        sess = tf.Session()

        sess.run(init)
        c,d = data.train.next_batch(BATCH_SIZE)
        a = sess.run(logits,feed_dict={images_pl: c})
        print a
        summary_writer = tf.train.SummaryWriter("summary", sess.graph)

        for step in range(N_EPOCH * (DS_SIZE // BATCH_SIZE)):
            start_time = time.time()
            feed_dict = fill_feed_dict(data.train, images_pl, labels_pl)
            _, loss_val = sess.run([train_op, loss], feed_dict=feed_dict)
            duration = time.time() - start_time

            assert not np.isnan(loss_val), 'Model diverged with loss = NaN'

            if step % 10 == 0 or step == N_EPOCH * (DS_SIZE // BATCH_SIZE) - 1:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_val, duration))
                if step > 0:
                    summary_str = sess.run(summary_op, feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

            if step % 100 == 0 or step == N_EPOCH * (DS_SIZE // BATCH_SIZE) - 1:
                save_path = saver.save(sess, "model.ckpt")
                print("Model saved in file: %s" % save_path)
                print('Training Data Eval:')
                do_eval(sess, eval_correct, images_pl, labels_pl, data.train)
                print('Validation Data Eval:')
                do_eval(sess, eval_correct, images_pl, labels_pl, data.validation)


if tf.gfile.Exists("summary"):
    tf.gfile.DeleteRecursively("summary")
# input.maybe_download_and_extract()
data = input.get_data()

run_training(data)
