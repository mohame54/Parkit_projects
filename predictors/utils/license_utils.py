import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Sequence, Union, Tuple
import numpy as np
import time


# loading and preprocessing the images before feed it to the model
def load_resize_img(img_path: str = None,
                    img: np.array = None,
                    img_size: Union[Sequence[int], List[int], Tuple[int]] = (64, 128)) -> tf.Tensor:
    """
    img_path:str path of the image
    img:array of image
    img_size: tuple of size 2 for the image size
    """
    if img_path is not None:
        img = cv2.imread(img_path)[..., ::-1]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., None]  # conert to gray
    img = tf.image.resize(img, img_size)  # resizing the image
    img = img[:, ::-1, :]  # invert the width for the image
    img = tf.transpose(img, perm=[1, 0, 2]) / 255.0  # transposing and normalizing the images
    img = img[:, ::-1, :]
    img = tf.convert_to_tensor(img)
    return img


def clip_img(img, coords):
    x1, y1, x2, y2 = coords
    y = img[y1:y2, x1:x2, :]
    return y.copy()


def plot_grid_imgs(imgs: Union[List[np.array], List[tf.Tensor], tf.Tensor],
                   num_cols: int,
                   num_rows: int,
                   texts: List[str] = None,
                   return_axes: bool = False):
    """
    imgs: list of images or batched tensor of images
    num_cols: int num_cols in the grid
    num_rows: int num_rows in the grid
    texts:str text title for every cell if provided
    return_axes:bool to return the axes for further plotting
    """
    assert (num_cols * num_rows) <= len(imgs), "the total cells number is bigger than number of images"
    plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))  # making a figure
    for row in range(num_rows):
        for col in range(num_cols):
            index = row * num_cols + col
            plt.subplot(num_rows, num_cols, index + 1)
            plt.imshow(imgs[index], cmap='gray')
            if texts is not None:
                plt.title(texts[index])
            plt.axis("off")
    plt.tight_layout()
    if return_axes:
        ax = plt.gca()
        return ax
    plt.show()
    plt.close()


def load_model(model_path: str):
    """
    model_path:str the model_path to load
    """
    return tf.saved_model.load(model_path)


def convert_to_tflite(model_path: str
                      , new_path: str):
    """
    model_path:str the saved model path
    new_path:str the destination path for the tflite
    """
    st = time.process_time()
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    m = converter.convert()
    with open(new_path + ".tflite", 'wb') as file:
        file.write(m)
    end = time.process_time()
    print(f"Finished converting in {end - st} secs.")
