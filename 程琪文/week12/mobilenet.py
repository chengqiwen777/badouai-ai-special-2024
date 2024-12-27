import numpy as np
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, DepthwiseConv2D
from keras import backend as K
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Reshape, Dropout
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions

def MobileNet(input_shape=(224, 224, 3),
              classes=1000,
              depth_multiplier=1,
              dropout=1e-3):

    img_input = Input(shape=input_shape)

    x = _conv_block(img_input, 32, strides=(2, 2))

    x = _depthwise_conv_block(x, 64,
                              depth_multiplier=depth_multiplier, block=1)

    x = _depthwise_conv_block(x, 128, depth_multiplier=depth_multiplier,
                              strides=(2, 2), block=2)

    x = _depthwise_conv_block(x, 128, depth_multiplier=depth_multiplier,
                              block=3)

    x = _depthwise_conv_block(x, 256, depth_multiplier=depth_multiplier,
                              strides=(2, 2), block=4)

    x = _depthwise_conv_block(x, 256, depth_multiplier=depth_multiplier,
                              block=5)

    x = _depthwise_conv_block(x, 512, depth_multiplier=depth_multiplier,
                              strides=(2, 2), block=6)

    x = _depthwise_conv_block(x, 512, depth_multiplier=depth_multiplier, block=7)
    x = _depthwise_conv_block(x, 512, depth_multiplier=depth_multiplier, block=8)
    x = _depthwise_conv_block(x, 512, depth_multiplier=depth_multiplier, block=9)
    x = _depthwise_conv_block(x, 512, depth_multiplier=depth_multiplier, block=10)
    x = _depthwise_conv_block(x, 512, depth_multiplier=depth_multiplier, block=11)

    x = _depthwise_conv_block(x, 1024, depth_multiplier=depth_multiplier,
                              strides=(2, 2), block=12)

    x = _depthwise_conv_block(x, 1024, depth_multiplier=depth_multiplier, block=13)

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same', name='conv_prediections')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)

    model = Model(img_input, x, name='MobileNet')
    model.load_weights('mobilenet_1_0_224_tf.h5')

    return model


def _conv_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel_size, strides=strides, use_bias=False, name='conv1')(x)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation(relu6, name='conv1_relu')(x)
    return x


def _depthwise_conv_block(inputs, pointwise_conv_filters,
                          strides=(1, 1), depth_multiplier=1, block=1):
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', strides=strides,
                        depth_multiplier=depth_multiplier, use_bias=False,
                        name='conv_dw_%d' % block)(inputs)
    x = BatchNormalization(name='conv_dw_%d_bn' % block)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1), padding='same',
               use_bias=False, strides=(1, 1), name='conv_pw_%d' % block)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block)(x)
    x = Activation(relu6, name='conv_pw_%d_relu' % block)(x)
    return x


def relu6(x):
    return K.relu(x, max_value=6)


def preprocess_input(x):
    x /= 255.0
    x -= 0.5
    x *= 2.0
    return x


if __name__ == '__main__':
    model = MobileNet()
    img = image.load_img('elephant.jpg', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, 0)
    x = preprocess_input(x)
    pred = model.predict(x)
    print(decode_predictions(pred))
