from tensorflow.keras.models import model_from_json
import numpy as np 
import tensorflow as tf 

## Optional 3 steps : For limiting GPU Memory allocation 
config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.15
session=tf.compat.v1.Session(config=config)

class FacialExpressionModel(object):
    Expression_list=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
    def __init__(self,model_json_file, model_weights_file):
        with open(model_json_file,"r") as json_file:
            loaded_model_json=json_file.read()
            self.loaded_model=model_from_json(loaded_model_json)
            
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_prediction_function()
        
    def predictEmotion(self,img):
        self.predictions=self.loaded_model.predict(img)
        return FacialExpressionModel.Expression_list[np.argmax(self.predictions)]