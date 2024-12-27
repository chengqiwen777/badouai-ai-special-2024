import tensorflow as tf
from model import vgg16
import utils


img1 = utils.load_image('dog.jpg')
inputs = tf.placeholder(tf.float32, [None, None, 3])
resized_img = utils.resize_image(inputs, (224, 224))

prediction = vgg16.vgg16(resized_img)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
ckpt_file = './logs/vgg_16.ckpt'
saver.restore(sess, ckpt_file)

probs = tf.nn.softmax(prediction)
res = sess.run(probs, feed_dict={inputs: img1})

print(utils.print_prob(res[0], 'synset.txt'))
