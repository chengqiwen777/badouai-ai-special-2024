import numpy as np
import tensorflow as tf
import math
import cifar_data
import time

max_steps = 2000
batch_size = 200
num_examples_per_epoch_for_eval = 10000
data_dir = 'cifar-10-batches-bin'

def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weights_loss')
        tf.add_to_collection('losses', weights_loss)
    return var

images_train, labels_train = cifar_data.inputs(data_dir, batch_size, True)
images_test, labels_test = cifar_data.inputs(data_dir, batch_size, None)

x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
y_ = tf.placeholder(tf.int32, [batch_size])

kernel1 = variable_with_weight_loss([5, 5, 3, 64], 5e-2, 0.0)
conv1 = tf.nn.conv2d(x, kernel1, strides=[1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(conv1 + bias1)
pool1 = tf.nn.max_pool(relu1, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

kernel2 = variable_with_weight_loss([5, 5, 64, 64], 5e-2, 0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, strides=[1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.01, shape=[64]))
relu2 = tf.nn.relu(conv2 + bias2)
pool2 = tf.nn.max_pool(relu2, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

reshaped_tensor = tf.reshape(pool2, [batch_size, -1])
dim = reshaped_tensor.get_shape()[1].value

weights1 = variable_with_weight_loss([dim, 384], 0.004, 0.04)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc1 = tf.nn.relu(tf.matmul(reshaped_tensor, weights1) + fc_bias1)

weights2 = variable_with_weight_loss([384, 192], 0.004, 0.04)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
fc2 = tf.nn.relu(tf.matmul(fc1, weights2) + fc_bias2)

weights3 = variable_with_weight_loss([192, 10], 0.04, 0)
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
fc3 = tf.add(tf.matmul(fc2, weights3), fc_bias3)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc3, labels=y_)
weights_l2_loss = tf.add_n(tf.get_collection('losses'))
loss = tf.reduce_mean(cross_entropy) + weights_l2_loss

train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
top1_op = tf.nn.in_top_k(fc3, y_, 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    tf.train.start_queue_runners()

    for i in range(max_steps):
        start_time = time.time()
        image_batch, label_batch = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y_: label_batch})

        duration = time.time() - start_time
        if i % 100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            print('step: %d, loss: %.3f (%.2f examples/sec; %.2f sec/batch)' %
                  (i, loss_value, examples_per_sec, sec_per_batch))

    num_batch = int(math.ceil(num_examples_per_epoch_for_eval / batch_size))
    total = batch_size * num_batch
    correct = 0

    for i in range(num_batch):
        image_batch, label_batch = sess.run([images_test, labels_test])
        pred = sess.run([top1_op], feed_dict={x: image_batch, y_: label_batch})
        correct += np.sum(pred)

    print('Accurancy:', 100 * correct / total)
