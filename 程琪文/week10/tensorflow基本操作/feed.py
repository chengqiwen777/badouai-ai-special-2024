import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
res = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run([res], feed_dict={input1: [1], input2: [2]}))
