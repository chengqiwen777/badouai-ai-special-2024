import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import cv2

def load_image(image_path):
    img = mpimg.imread(image_path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    return crop_img


def resize_images(image_list, target_size=(224, 224)):
    with tf.name_scope('resize_images'):
        new_images = []
        for image in image_list:
            img = cv2.resize(image, (target_size[1], target_size[0]))
            new_images.append(img)
        images = np.array(new_images)
        return images


def print_prob(argmax):
    with open('index.txt', 'r', encoding='utf-8') as f:
        synset = [l.split(';')[1][:-1] for l in f.readlines()]
    print(synset[argmax])
    return synset[argmax]
