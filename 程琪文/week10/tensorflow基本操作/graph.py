import tensorflow as tf

# matrix1 = tf.constant([[1., 2.]])
# matrix2 = tf.constant([[1.], [2.]])
# product = tf.matmul(matrix1, matrix2)
#
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()

with tf.device('/cpu:0'):
    a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], name='num1')
    b = tf.constant([6, 5, 4, 3, 2, 1], shape=[3, 2], name='num2')
c = tf.matmul(a, b)
sess = tf.Session()
res = sess.run(c)
print(res)
sess.close()
