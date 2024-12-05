import tensorflow as tf

a = tf.constant([1,2,3], name='a')
b = tf.Variable(tf.random_uniform([3]), name='b')
c = tf.add_n([a, b], name='c')

writer = tf.summary.FileWriter('logs', tf.get_default_graph())
writer.close()
