from yolov5 import detect
from Dataset import Dataset_Descriptor
import os



class Model:
    def __init__(self,model_dir,data):
        self.model_dir = model_dir
        self.data = data
        self.res_dir = Dataset_Descriptor.MODEL_RESULT_LOCATION
    
    def _init_work_directory(self):
        os.system(f"rm -r {self.res_dir}/*")
        os.rmdir(self.res_dir)
    
    def detect(self):
        for i in range(len(self.data.image_dir)):
            img_size = self.data.sample_size[i]
            img_dir = self.data.image_dir[i]

            #classes = None
            #classes = [i for i in range(14,24,1)]

            detect.run(imgsz = img_size,
                        weights = self.model_dir,
                        source = img_dir, 
                        project = self.res_dir,
                        name = f"item{i}",
                        save_txt=True,
                        conf_thres=0.25)
                        #classes=classes)
        
