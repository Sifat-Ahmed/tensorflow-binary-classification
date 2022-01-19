import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tqdm import tqdm
import cv2
from config import Config
from tensorflow.keras.models import load_model
import numpy as np

class Detect:
    def __init__(self):
        self._cfg = Config()       

        if os.path.isfile(self._cfg.model_path):
            self._model = load_model(self._cfg.model_path)
        else:
            raise("Model path not found")
        
    def _preprocess_image(self, image):
        if self._cfg.resize:
            image = cv2.resize(image, self._cfg.image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self._cfg.test_transform(image = image)["image"]
        image = image / 255.
        return image
    
    
    def is_smoke(self, image):

        image = self._preprocess_image(image)
        image = np.expand_dims(image, 0)

        output = self._model.predict(image)
        output = True if output >= self._cfg.classification_threshold else False
        
        return output
    
    
    def run(self, source):
        self._read_source(source)
    
    
    def _read_source(self, path):
        i, count = 1, 1
        color = (255, 255, 255)
        text = ""

        ret = True

        for imagep in os.listdir(path):
            frame = cv2.imread(os.path.join(path, imagep))
            if frame is None: continue
            
            image = frame.copy()
            
            output = self.is_smoke(image)
            
            if output: 
                color = (0, 0, 255)
                text = "smoke"
            else: 
                color = (0, 255, 0)
                text = "no smoke"
                
            image = cv2.putText(frame.copy(),
                                text, 
                                (10, 600), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                5, 
                                color, 
                                1, 
                                cv2.LINE_AA)
            print(imagep, output)
            #cv2.imshow('output', image)
            count += 1
            cv2.waitKey(100)
        
        

if __name__ == '__main__':
    det = Detect()
    det.run(r"/home/workstaion/workspace/DATASET_ALL/segmented_smoke/smoke")