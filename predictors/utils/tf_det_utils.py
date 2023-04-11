import tensorflow as tf


def xywh2yxyx(boxes):  # from (x_center,y_center,w,h) -> (x1,y1,x2,y2)
    """
    boxes:tf.Tensor of shape (num_boxes,4)
    """
    xc, yc, w, h = tf.split(boxes, 4, axis=-1)
    x1 = xc - w / 2
    x2 = xc + w / 2
    y1 = yc - h / 2
    y2 = yc + h / 2
    return tf.concat([y1, x1, y2, x2], axis=-1)


def reorganize(boxes):
    # reorganizing the bboxes
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=-1)
    return tf.concat([x1, y1, x2, y2], axis=-1)


# non-max-suppression
def nms(preds,
        score_th=0.3, iou_th=0.5, max_dets=300):
    """
    preds:tf.Tensor of shape (1,4,num_boxes)
    score_th:float cls_score
    iou_th: float  iou_threshold
    max_dets:int max number of detections
    """
    preds = preds[0]
    preds = tf.transpose(preds, perm=[1, 0])
    sort = tf.argsort(preds[:, 4],
                      direction='DESCENDING')  # sorting the bboxes in descending order according to the cls score
    preds = tf.gather(preds, sort)
    boxes = xywh2yxyx(preds[:, :4])
    scores = preds[:, 4]
    out_indices = tf.image.non_max_suppression(boxes,
                                               scores,
                                               max_dets,
                                               iou_th,
                                               score_th)

    out_boxes = tf.gather(boxes, out_indices)
    out_boxes = reorganize(out_boxes)
    out_scores = tf.gather(scores, out_indices)
    return out_boxes, out_scores


def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    # calculate from img0_shape
    img1_shape = tf.cast(img1_shape, tf.float32)
    img0_shape = tf.cast(img0_shape, tf.float32)
    gain = tf.minimum(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    gain = tf.cast(gain, tf.float32)
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    x1, y1, x2, y2 = tf.split(coords, 4, axis=-1)
    x1 = x1 - pad[0]  # x padding
    x2 = x2 - pad[0]  # x padding
    y1 = y1 - pad[1]  # y padding
    y2 = y2 - pad[1]  # y padding
    coords = tf.concat([x1, y1, x2, y2], axis=-1)
    coords = coords / gain
    coords = clip_boxes(coords, img0_shape)
    return coords


def clip_boxes(boxes, shape):
    """
    It takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the
    shape
    Args:
      boxes (tf.Tensor): the bounding boxes to clip
      shape (tuple): the shape of the image
    """
    x1 = tf.clip_by_value(boxes[:, 0], 0.0, shape[1])
    y1 = tf.clip_by_value(boxes[:, 1], 0.0, shape[0])
    x2 = tf.clip_by_value(boxes[:, 2], 0.0, shape[1])
    y2 = tf.clip_by_value(boxes[:, 3], 0.0, shape[0])
    return tf.stack([x1, y1, x2, y2], axis=-1)

