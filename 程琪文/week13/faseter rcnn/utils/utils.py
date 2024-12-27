import tensorflow as tf
import numpy as np


class BBoxUtility(object):
    def __init__(self, priors=None, overlap_threshold=0.7, ignore_threshold=0.3,
                 nms_thresh=0.7, top_k=300):
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self.ignore_threshold = ignore_threshold
        self._nms_thresh = nms_thresh
        self._top_k = top_k
        self.boxes = tf.placeholder(tf.float32, [None, 4])
        self.scores = tf.placeholder(tf.float32, [None,])
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                self._nms_thresh)
        self.sess = tf.Session()

    @property
    def nms_thresh(self):
        return self._nms_thresh

    @nms_thresh.setter
    def nms_thresh(self, val):
        self._nms_thresh = val
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                self._nms_thresh)

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, val):
        self._top_k = val
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k,
                                                self._nms_thresh)

    def iou(self, box):
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])
        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)

        inter_area = inter_wh[:, 0] * inter_wh[:, 1]

        box_area = (box[2] - box[0]) * (box[3] - box[1])
        priors_area = (self.priors[:, 2] - self.priors[:, 0]) * (self.priors[:, 3] - self.priors[:, 1])

        union = priors_area + box_area - inter_area
        iou = inter_area / union

        return iou

    def encode_box(self, box, return_iou=True):
        iou = self.iou(box)
        encoded_box = np.zeros([self.num_priors, 4+return_iou])

        assign_mask = iou > self.overlap_threshold

        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]

        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:4] - box[0:2]

        assigned_priors = self.priors[assign_mask]
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] + assigned_priors[:, 2:4])
        assigned_priors_wh = assigned_priors[:, 2:4] - assigned_priors[:, :2]

        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh
        encoded_box[:, :2][assign_mask] *= 4

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
        encoded_box[:, 2:4][assign_mask] *= 4

        return encoded_box.ravel()

    def ignore_box(self, box):
        iou = self.iou(box)
        ignored_box = np.zeros((self.num_priors, 1))
        assign_mask = (iou > self.ignore_threshold) and (iou < self.overlap_threshold)

        if not assign_mask.any():
            assign_mask[iou.argmax()] = True

        ignored_box[:, 0][assign_mask] = iou[assign_mask]
        return ignored_box.ravel()

    def assign_boxes(self, boxes, anchors):
        self.num_priors = len(anchors)
        self.priors = anchors
        assignment = np.zeros((self.num_priors, 5))
        assignment[:, -1] = 0.0
        if len(boxes) == 0:
            return assignment

        ignored_boxes = np.apply_along_axis(self.ignore_box, 1, boxes[:, :4])
        ignored_boxes = ignored_boxes.reshape([-1, self.num_priors, 1])
        ignore_iou = ignored_boxes[:, :, -1].max(axis=0)
        ignore_iou_mask = ignore_iou > 0
        assignment[:, -1][ignore_iou_mask] = -1

        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, 4])
        encoded_boxes = encoded_boxes.reshape([-1, self.num_priors, 5])
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        assign_num = len(best_iou_idx)
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        assignment[:, 4][best_iou_mask] = 1

        return assignment
