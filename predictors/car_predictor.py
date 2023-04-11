import cv2
import time
import numpy as np
import torch
from .modules import Bbox
from .utils import letter_box, load_preprocess, preprocess
from .utils.torch_det_utils import non_max_suppression, scale_coords


class Predictor:
    def __init__(self, model_path, **nmskwargs):
        try:
            self.model = torch.jit.load(model_path)
        except FileNotFoundError:
            raise ValueError(f'the path not found:{model_path}')
        self.device = torch.device('cpu')
        self.model = self.model.to(self.device)
        self.nmskwargs = nmskwargs
        self.resizer = letter_box

    def __call__(self, img_org=None, img_path=None):  # if the image is provided it must be in RGB format
        with torch.no_grad():
            if img_org is None and img_path is None:
                raise ValueError('you must provide img attr or img_path attr')
            if img_path is not None:
                im, img_org = load_preprocess(img_path)
            else:
                im = self.resizer(image=img_org)
                im = preprocess(im)
            im = torch.from_numpy(im).to(self.device)
            preds = self.model(im).cpu()
            preds = non_max_suppression(preds, **self.nmskwargs)[0]
            preds[:4] = scale_coords(im.shape[2:], preds[:4], img_org.shape[:-1])
        return Bbox.from_array_tensor(preds)

    def from_video(self, vid_path, out_path='vid.avi', with_conf=False, fps=20):
        print(f"Making A video in :{out_path}.")
        st = time.process_time()
        cap = cv2.VideoCapture(vid_path)  # main video
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        font = cv2.FONT_HERSHEY_DUPLEX  # the font type
        out = cv2.VideoWriter(f'{out_path}', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                              (frame_width, frame_height))  # making a new video
        while cap.isOpened():
            ret, img = cap.read()
            if ret:
                im = img[..., ::-1]
                im = self.reiszer(image=im)
                im = preprocess(im)
                with torch.no_grad():
                    im = torch.from_numpy(im).to(self.device)
                    preds = self.model(im)
                    preds = non_max_suppression(preds, **self.nmskwargs)[0]
                    preds[:4] = scale_coords(im.shape[2:], preds[:4], img.shape[:-1])
                boxes = Bbox.from_array_tensor(preds)  # list of bboxes
                num_boxes = len(boxes)  # number of boxes
                del im, preds  # delete the unecessary variables
                for box in boxes:  # drawing the boxes
                    attrs = np.array(box.attrs)
                    if with_conf:
                        text = f"Car:{box.conf:.3f}"
                    else:
                        text = 'Car'

                    img = cv2.rectangle(img, attrs[:2], attrs[2:], color=(0, 255, 0), thickness=3)
                    img = cv2.putText(img, text, attrs[:2] - 10, font, 1, (0, 0, 0), 1, cv2.LINE_AA)
                out.write(img)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # terminating the process
                    break
        cap.release()
        out.release()
        print(f"Finished in {(time.process_time() - st):.3f} secs")
