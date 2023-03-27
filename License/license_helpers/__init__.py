from .  import license_utils
from .license_tokenizer import *
from . import obj_det_utils
import tensorflow as tf
import time


def load_model(model_path:str):
    """
    model_path:str the model_path to load
    """
    return tf.saved_model.load(model_path)

def convert_to_tflite(model_path:str
                     ,new_path:str):
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
    with open(new_path+".tflite",'wb') as file:
        file.write(m)    
    end = time.process_time()
    print(f"Finished converting in {end-st} secs.")    