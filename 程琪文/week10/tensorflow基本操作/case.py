import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.1, x_data.shape)
y_data = np.square(x_data) + noise


x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 隐藏层
weights_L1 = tf.Variable(tf.random_normal([1, 10]))
# bias_L1 = tf.Variable(tf.random_normal([1, 10]))
z1 = tf.matmul(x, weights_L1)
outputs_L1 = tf.nn.relu(z1)

# 输出层
weights_L2 = tf.Variable(tf.random_normal([10, 1]))
# bias_L2 = tf.Variable(tf.random_normal([1, 1]))
z2 = tf.matmul(outputs_L1, weights_L2)
evaluation = tf.nn.relu(z2)

loss = tf.reduce_mean(tf.square(y-evaluation))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    for i in range(1000):
        sess.run(optimizer, feed_dict={x: x_data, y: y_data})

    evaluation_val = sess.run(evaluation, feed_dict={x: x_data})

    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, evaluation_val, 'r-', lw=3)
    plt.show()

writer = tf.summary.FileWriter('logs', tf.get_default_graph())
writer.close()
