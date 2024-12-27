import tensorflow as tf
import os

num_classes = 10
num_examples_per_epoch_for_train = 50000
num_examples_per_epoch_for_test = 10000


class Cifar10Record:
    pass


def read_cifar(filequeue):
    result = Cifar10Record()

    label_bytes = 1
    result.depth = 3
    result.height = 32
    result.width = 32

    image_bytes = result.depth * result.height * result.width

    record_bytes = image_bytes + label_bytes

    reader = tf.FixedLengthRecordReader(record_bytes)
    result.key, value = reader.read(filequeue)

    record = tf.decode_raw(value, tf.uint8)

    result.label = tf.cast(tf.strided_slice(record, [0], [label_bytes]), tf.int32)

    depth_major = tf.reshape(tf.strided_slice(record, [label_bytes], [record_bytes]),
                             [result.depth, result.height, result.width])

    result.uint8Image = tf.transpose(depth_major, [1, 2, 0])

    return result


def inputs(data_dir, batchsize, distorted):
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]

    filequeue = tf.train.string_input_producer(filenames)

    read_input = read_cifar(filequeue)
    read_label = read_input.label
    read_image_float = tf.cast(read_input.uint8Image, tf.float32)

    if distorted is not None:
        cropped_image = tf.random_crop(read_image_float, [24, 24, 3])
        flipped_image = tf.image.random_flip_left_right(cropped_image)
        adjusted_brightness = tf.image.random_brightness(flipped_image, 0.8)
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness, 0.2, 1.8)
        std_image = tf.image.per_image_standardization(adjusted_contrast)

        std_image.set_shape([24, 24, 3])
        read_label.set_shape([1])

        min_examples_queue = int(num_examples_per_epoch_for_test * 0.4)

        images_train, labels_train = tf.train.shuffle_batch([std_image, read_label],
                                                            batch_size=batchsize,
                                                            capacity=3 * batchsize + min_examples_queue,
                                                            min_after_dequeue=min_examples_queue,
                                                            num_threads=16)

        return images_train, tf.reshape(labels_train, [batchsize])
    else:
        resized_image = tf.image.resize_image_with_crop_or_pad(read_image_float, 24, 24)
        std_image = tf.image.per_image_standardization(resized_image)

        std_image.set_shape([24, 24, 3])
        read_label.set_shape([1])

        min_examples_queue = int(0.4 * num_examples_per_epoch_for_train)

        images_test, labels_test = tf.train.batch([std_image, read_label],
                                                  batch_size=batchsize,
                                                  capacity=3 * batchsize + min_examples_queue,
                                                  num_threads=16)

        return images_test, tf.reshape(labels_test, [batchsize])
