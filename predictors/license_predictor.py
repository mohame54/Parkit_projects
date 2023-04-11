from .modules import Bbox, ArabCharTokenizer
from .utils.license_utils import *
from .utils import load_preprocess


class License_predictor:
    def __init__(self, obj_model_path: str,
                 vocab_path: str,
                 license_model_path: str,
                 **nms_kwargs):
        if obj_model_path.find('complete') != -1:
            self._complete = True
        self.obj_model = load_model(obj_model_path)  # load detection model
        self.lic_det = load_model(license_model_path)  # load recognition model
        self.tokenizer = ArabCharTokenizer.from_vocab_file(vocab_path)  # load the tokenizer
        self.nms_kwargs = nms_kwargs # non-max suppression kwargs

    def _clip_to_boxes(self, img_org, coords):
        signs_list = []
        for coors in coords:
            signs_list.append(clip_img(img_org, coors.attrs))
        return signs_list

    def __call__(self, img_path=None, img=None):
        im, img_org = load_preprocess(img_path, img)  # loading and preprocessing the image for license detection
        im = tf.convert_to_tensor(im)
        outs = self.obj_model(im, img_org.shape[:-1], **self.nms_kwargs)  # the bounding boxes
        del im
        boxes = Bbox.from_array_tensor(outs.numpy())  # converting it to bbox class
        imgs = self._clip_to_boxes(img_org, boxes)  # clip the license from the main image
        imgs = [load_resize_img(img=i) for i in imgs]  # preprocessing the image before feeding it
        for i, im in enumerate(imgs):
            outs = self.lic_det(im)  # the license number encoding
            text = self.tokenizer.decode(outs.numpy().squeeze())  # decoding it to text
            boxes[i].text = text  # storing each text for each bboxes
        del img_org
        return boxes
