# import tensorflow as tf
#
# state = tf.Variable(0, name='Counter')
#
# one = tf.constant(1)
# new_value = tf.add(state, one)
# update = tf.assign(state, new_value)
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
# print('state', sess.run(state))
#
# for i in range(5):
#     sess.run(update)
#     print('update', sess.run(state))
# sess.close()


import tensorflow as tf

state = tf.Variable(0, name='counter')
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)
# sess.run(state)
for _ in range(5):
    print('update', sess.run(update))
    print('state', sess.run(state))
sess.close()
