import tensorflow as tf
import numpy as np

import graph
import input

BATCH_SIZE = 1  # 100

prediction_X, prediction_y, file_names = input._read_data()
prediction_X = prediction_X.reshape(12, 512, 512, 3)
mean_image = np.mean(prediction_X, axis=0)
prediction_X -= mean_image.astype(np.uint8)
prediction_data = input.DataSet(prediction_X, prediction_y)


def run_prediction(data):
    ret = []
    with tf.Graph().as_default():
        images_pl = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 512, 512, 3])
        labels_pl = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

        logits = graph.inference(images_pl)

        saver = tf.train.Saver(tf.trainable_variables())

        init = tf.initialize_all_variables()
        sess = tf.Session()

        saver.restore(sess, "model.ckpt")
        print("Model restored.")
        sess.run(init)

        images, labels = (data.images, data.labels)

        for example in range(data.num_examples):
            # images, labels = data.next_batch(BATCH_SIZE)

            feed_dict = {
                images_pl: [images[example]],
                labels_pl: [labels[example]]
            }

            logits_op, lab = sess.run([logits, labels_pl], feed_dict=feed_dict)

            # print logits_op[0], logits_op[0].argmax(), lab, data.labels[example]
            ret.append([logits_op[0], lab])

        return ret


def get_predictions():
    prediction_op = run_prediction(prediction_data)
    return prediction_op, file_names


if __name__ == "__main__":
    prediction_op, file_names = get_predictions()
    print prediction_op, file_names
