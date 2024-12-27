from model.AlexNet import AlexNet
import utils
import numpy as np
import cv2
from keras import backend as K

K.image_data_format() == 'channels_first'

if __name__ == '__main__':
    model = AlexNet()
    model.load_weights('last1.h5')

    img = cv2.imread('test.jpg')
    img_nor = img / 255.0
    img_nor = np.expand_dims(img_nor, 0)
    img_resize = utils.resize_images(img_nor, (224, 224))

    pred = model.predict(img_resize)
    utils.print_prob(np.argmax(pred))

    cv2.imshow('show', img)
    cv2.waitKey()
