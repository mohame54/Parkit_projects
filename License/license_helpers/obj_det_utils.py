import numpy as np
import tensorflow as tf
import cv2




def xywh2yxyx(boxes): # from (x_center,y_center,w,h) -> (x1,y1,x2,y2)
    """
    boxes:tf.Tensor of shape (num_boxes,4)
    """
    xc,yc,w,h = tf.split(boxes,4,axis=-1)
    x1 = xc-w/2
    x2 = xc+w/2
    y1 = yc-h/2
    y2 = yc+h/2
    return tf.concat([y1,x1,y2,x2],axis=-1)

def reOrganize(boxes):
    # reorganizing the bboxes
    y1,x1,y2,x2 = tf.split(boxes,4,axis=-1)
    return tf.concat([x1,y1,x2,y2],axis=-1)

#non-max-supression    
def nms(preds,
        score_th=0.3,iou_th=0.5,max_dets=300):
    
    """
    preds:tf.Tensor of shape (1,4,num_boxes)
    score_th:float cls_score
    iou_th: float  iou_threshold
    max_dets:int max number of detections
    """
    preds =preds[0]
    preds = tf.transpose(preds,perm=[1,0]) 
    sort = tf.argsort(preds[:,4],direction='DESCENDING') # sorting the bboxes in descending order according to the cls score
    preds = tf.gather(preds,sort)
    boxes = xywh2yxyx(preds[:,:4])
    scores = preds[:,4]
    out_indices = tf.image.non_max_suppression(boxes,
                                        scores,
                                        max_dets,
                                        iou_th,
                                        score_th)
    
    out_boxes = tf.gather(boxes,out_indices)
    out_boxes = reOrganize(out_boxes)
    out_scores = tf.gather(scores,out_indices)
    return out_boxes,out_scores
    
    
def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    # calculate from img0_shape
    img1_shape = tf.cast(img1_shape,tf.float32)
    img0_shape = tf.cast(img0_shape,tf.float32)
    gain = tf.minimum(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    gain = tf.cast(gain,tf.float32)
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    x1,y1,x2,y2 = tf.split(coords,4,axis=-1)
    x1 = x1 - pad[0]   # x padding
    x2 = x2 - pad[0]  # x padding
    y1 = y1 - pad[1]  # y padding
    y2 = y2 -  pad[1] # y padding
    coords = tf.concat([x1,y1,x2,y2],axis=-1)
    coords = coords/ gain
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
    x1 = tf.clip_by_value(boxes[:,0],0.0,shape[1])
    y1 = tf.clip_by_value(boxes[:,1],0.0,shape[0])
    x2 = tf.clip_by_value(boxes[:,2],0.0,shape[1])
    y2 = tf.clip_by_value(boxes[:,3],0.0,shape[0])
    return tf.stack([x1,y1,x2,y2],axis=-1)


def preprocess(img):
  y = img.copy()
  y = y.astype(np.float32)/255
  return tf.convert_to_tensor(y[None])

def load_preprocess(img_path=None,img=None):
  if img_path is not None:
     img = cv2.imread(img_path)[...,::-1]
  im = LetterBox()(image=img)
  im = preprocess(im)
  return im,img 

class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose"""

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get('img') if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop('rect_shape', self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if labels.get('ratio_pad'):
            labels['ratio_pad'] = (labels['ratio_pad'], (dw, dh))  # for evaluation

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels['img'] = img
            labels['resized_shape'] = new_shape
            return labels
        else:
            return img  