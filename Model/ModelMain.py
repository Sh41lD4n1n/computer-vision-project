from yolov5 import detect
from Dataset import Dataset_Descriptor
from Model import Model
import os


"""
Before first run install yolov5
It support two type to load data 'file'/'array'
    'file'
        by default it load images from input folder 
        (can be configured with `init_path`)
    'array'
        expect input from array of numpy images (need to be tested before usage!)

Model work with images located in directory Dataset_Descriptor.DATASET_LOCATION and 
DATASET_DEST_LOCATION

Model save result to MODEL_RESULT_LOCATION folder
"""

def run(init_path):
    data = Dataset_Descriptor(init_param=init_path,l_type="file")
    print(data.sample_size)

    model = Model(model_dir="./model_weights/yolov5s.pt",data=data)
    model.detect()
    data.set_result()

    return data.image_processed,data.labels
    print(data.labels)
    print(data.image_processed)

run(init_path="input")