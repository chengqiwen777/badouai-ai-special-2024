import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np


def load_image(img_path):
    img = mpimg.imread(img_path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    return crop_img


def resize_image(img, size, method=tf.image.ResizeMethod.BILINEAR, align_corners=False):
    img = tf.expand_dims(img, 0)
    img = tf.image.resize_images(img, size, method, align_corners)
    img = tf.reshape(img, tf.stack([-1, size[0], size[1], 3]))
    return img


def print_prob(probs, file_path):
    with open(file_path, 'r') as f:
        synset = [l.strip() for l in f.readlines()]
    pred = np.argsort(probs)[::-1]
    top1 = synset[pred[0]]
    print(('Top1:', top1, probs[pred[0]]))
    top5 = [(synset[pred[i]], probs[pred[i]]) for i in range(5)]
    print('Top5:', top5)
    return top1
