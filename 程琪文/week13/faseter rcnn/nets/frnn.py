from keras.layers import Conv2D, Input, TimeDistributed, Flatten, Dense, Reshape
from keras.models import Model
from nets.resnet import ResNet50, classifier_layers
from nets.RoiPoolingConv import RoiPoolingConv

def get_rpn(base_layers, num_anchors):
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regr')(x)
    x_class = Reshape((-1, 1), name='classification')(x_class)
    x_regr = Reshape((-1, 4), name='regression')(x_regr)

    return [x_class, x_regr, base_layers]
