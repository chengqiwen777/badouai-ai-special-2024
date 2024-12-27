from keras.engine.topology import Layer
import keras.backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf


class RoiPoolingConv(Layer):
    '''
       ROI pooling layer for 2D inputs.
       # Arguments
           pool_size: int
               Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
           num_rois: number of regions of interest to be used
       # Input shape
           list of two 4D tensors [X_img,X_roi] with shape:
           X_img:
           `(1, channels, rows, cols)` if dim_ordering='th'
           or 4D tensor with shape:
           `(1, rows, cols, channels)` if dim_ordering='tf'.
           X_roi:
           `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
       # Output shape
           3D tensor with shape:
           `(1, num_rois, channels, pool_size, pool_size)`
       '''

    def __init__(self, pool_size, num_rois, **kwargs):
        self.dim_ordering = K.image_dim_ordering()
        # assert语句用于确保self.dim_ordering的值只能是'tf'或'th'之一，如果不是，则抛出断言错误那么程序将抛出一个 AssertionError 异常，并显示错误消息 'dim_ordering must be in {tf, th}'。
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    # 重写
    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    # 定义了层的前向传播逻辑。
    def call(self, x, mask=None):
        assert (len(x) == 2)

        img = x[0]
        rois = x[1]

        outputs = []

        for roi_idx in range(self.num_rois):
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            # 在 TensorFlow 中，tf.image.resize_images 函数用于调整图像张量的大小。
            # 这个函数通常接受一个四维张量作为输入，其形状为
            # [batch_size, height, width, channels]，并返回调整大小后的图像张量。
            rs = tf.image.resize_images(img[:, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        # 在 Keras 中，K.permute_dimensions 是一个用于重新排列张量维度的函数。与transpose类似。
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output
