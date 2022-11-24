#import pandas as pd
import numpy as np
import os
from cv2 import cv2

class Dataset_Descriptor:
    DATASET_LOCATION = "./starage"
    DATASET_DEST_LOCATION = {"images":"./starage/images",
                             "labels":"./starage/labels",
                             "images_processed":"./starage/images_processed"
                            }
    MODEL_RESULT_LOCATION = "./results"

    def __init__(self,init_param,l_type):
        #l_type `file`/`array`
        self.init_path = None
        self.images_array = None

        self.image_dir = []
        self.sample_size = []
        
        self.image_processed = []
        self.labels = []
        self.animal_number = []

        self._init_work_directory()
        if l_type=='file':
            self.init_path = init_param
            self._get_images()
        elif l_type=='array':
            self.images_array = init_param
            self._load_images()

        
    #def __del__(self):
    #   os.system("rm -r dataset/images/test/*")

    def _init_work_directory(self):
        dest_loc = Dataset_Descriptor.DATASET_LOCATION
        images = Dataset_Descriptor.DATASET_DEST_LOCATION["images"]
        labels = Dataset_Descriptor.DATASET_DEST_LOCATION["labels"]
        images_processed = Dataset_Descriptor.DATASET_DEST_LOCATION["images_processed"]

        if (os.path.exists(dest_loc) or os.path.exists(images) or 
        os.path.exists(labels) or os.path.exists(images_processed)):
            os.system(f"rm -r {dest_loc}/*")
            os.rmdir(dest_loc)

        os.mkdir(dest_loc)
        os.mkdir(images)
        os.mkdir(labels)
        os.mkdir(images_processed)
    
    def _get_images(self):
        images = Dataset_Descriptor.DATASET_DEST_LOCATION["images"]

        for i in os.listdir(self.init_path):

            if not load_image(f_name_s=i,f_name_dest=i,init_folder=self.init_path):
                raise Exception("file not found")
            
            cur_size = cv2.imread(self.init_path+'/'+i).shape[0:2]
            cur_size = ([cur_size[0],cur_size[1]])
            
            self.image_dir.append(images+'/'+i)
            self.sample_size.append(cur_size)
    
    def _load_images(self):
        #REQUIER TESTING
        for i,img in enumerate(self.images_array):
            images = Dataset_Descriptor.DATASET_DEST_LOCATION["images"]
            cv2.imwrite(f"{images}/{i}.jpg",img)
            self.image_dir.append(f"{images}/{i}.jpg")
            
            cur_size = img.shape[0:2]
            cur_size = ([cur_size[0],cur_size[1]])
            
            self.sample_size.append(cur_size)

    def set_result(self):
        res_path = Dataset_Descriptor.MODEL_RESULT_LOCATION

        for i in os.listdir(res_path):
            
            img_path = res_path + "/" + i
            img_name = os.listdir(img_path)
            img_name.remove('labels')

            img_path = img_path + '/' + img_name[0]
            labels_path = res_path + "/" + i + "/labels"
            labels_path = [labels_path +'/'+i for i in os.listdir(labels_path)]
            
            self.image_processed.append(img_path)
            self.labels.append(get_pred_labels(labels_path)[0])


def load_image(f_name_s,f_name_dest,init_folder):
    dest_folder = Dataset_Descriptor.DATASET_DEST_LOCATION["images"]
    
    if os.path.isfile(f'{init_folder}/{f_name_s}'):# and check_image(f'./{init_folder}/{f_name}'):
        os.system(f'cp {init_folder}/{f_name_s} {dest_folder}/{f_name_dest}')
        return True
    return False


def load_pred_labels(file_name):
    box_pred = []
    conf_pred = []
    
    if os.path.isfile(file_name):
        with open(file_name,'r') as f:
            lns = f.readlines()
        
            for l in lns:
                pred = list(map(float,l.split(' ')))
                box_pred.append(pred[1:])
                conf_pred.append(pred[0])
        return conf_pred,np.array(box_pred)
        
        
def get_pred_labels(files):
    labels = []
    for f in files:
        _,bbox = load_pred_labels(f)
        labels.append(bbox)
    if len(files)==0:
        labels = [np.array([[0,0,0,0]])]
    return labels

#TESTS
"""d = Dataset_Descriptor(init_path='images',l_type='file')
print(d.image_dir)
print(d.sample_size)

d.set_result("results")
print(d.labels)
print(d.labels[0])
print(d.image_processed)"""
