import numpy as np
import keras


def generate_anchors(sizes=None, ratios=None):
    if sizes is None:
        sizes = [128, 256, 512]

    if ratios is None:
        ratios = [[1, 1], [1, 2], [2, 1]]

    num_anchors = len(sizes) * len(ratios)

    anchors = np.zeros([num_anchors, 4])

    anchors[:, 2:] = np.tile(sizes, [2, len(ratios)]).T

    for i in range(len(ratios)):
        anchors[3*i:3*i+3, 2] = anchors[3*i:3*i+3, 2] * ratios[i][0]
        anchors[3*i:3*i+3, 3] = anchors[3*i:3*i+3, 3] * ratios[i][1]

    anchors[:, 0::2] -= np.tile(anchors[:, 2], (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3], (2, 1)).T

    return anchors


def shift(shape, anchors, stride=16):
    shift_x = (np.arange(0, shape[0], dtype=keras.backend.floatx()) + 0.5) * stride
    shift_y = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shift_x = np.reshape(shift_x, [-1])
    shift_y = np.reshape(shift_y, [-1])

    shifts = np.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts = np.transpose(shifts)

    num_anchors = np.shape(anchors)[0]
    k = np.shape(shifts)[0]

    shifted_anchors = (np.reshape(anchors, [1, num_anchors, 4]) +
                       np.array(np.reshape(shifts, [k, 1, 4]), dtype=keras.backend.floatx()))
    shifted_anchors = np.reshape(shifted_anchors, [k*num_anchors, 4])

    return shifted_anchors


def get_anchors(shape, width, height):
    anchors = generate_anchors()

    network_anchors = shift(shape, anchors)
    network_anchors[:, 0::2] = network_anchors[:, 0::2] / width
    network_anchors[:, 1::2] = network_anchors[:, 1::2] / height

    network_anchors = np.clip(network_anchors, 0, 1)
    return network_anchors
