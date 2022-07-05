#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 12:06:30 2022

@author: ahsanjalal
"""





import numpy as np
import cv2
from pylab import *

from os.path import join, isfile
import sys,os,glob
from ctypes import *
import math
import random
from natsort import natsorted, ns

## CHANGE PATHS HERE ACCORDINGLY after cloning yolov4 
sys.path.insert(1, 'yolo_framework') 
## -----------------------------

import darknet


##parameters

sot_save_dir='data/kmeans_optical_dense_24_color_100000' #  directory
sot_classifier_dir='data/optical_kmean_24_classified_10k_085' # saving dir
temp_dir='temp_dir'
rgb_dir='data/original_test_frames'

##Paramters
def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))



def image_classification(image, network, class_names):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
    darknet_image =darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    darknet.free_image(darknet_image)
    return sorted(predictions, key=lambda x: -x[1])


cfg_path="trained/detector_only/efficientnet_b0.cfg"
weight_path="trained/detector_only/efficientnet_b0_last.weights"
data_path="trained/detector_only/fish_detector.data"

network, class_names, class_colors = darknet.load_network(
        cfg_path,
        data_path,
        weight_path,
        batch_size=1
    )

[img_h,img_w,ch]=[1080,1920,3]
if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

a=open('test_list.txt','r')
image_files=a.readlines()
image_files=natsorted(image_files)
count=0
for img_name in image_files:
    count+=1
    print(count)
    blobs=[]
    obj_arr=[]
    img_name=img_name.rstrip()
    filename=img_name.split('/')[-1]
    
    rgb_img=cv2.imread(img_name)
    # rgb_img=cv2.resize(rgb_img,[1920,1080])
    rgb_copy=rgb_img.copy()
    if not os.path.exists(join(sot_save_dir,filename)):
        optical_img=np.zeros(shape=(img_h,img_w))
    else:
        optical_img=cv2.imread(join(sot_save_dir,filename))
        optical_img=cv2.cvtColor(optical_img, cv2.COLOR_BGR2GRAY)
        # optical_img=cv2.resize(optical_img,[1920,1080])
   
    
    
#    ret, threshed_img = cv2.threshold(test_img,55, 255, cv2.THRESH_BINARY)
    contours, _= cv2.findContours(optical_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt)>10000:
            blobs.append(cnt)
    for blb in blobs:
        (x,y,w,h) = cv2.boundingRect(blb)
        rgb_copy=cv2.rectangle(rgb_copy,(x,y),(x+w,y+h),(255,12,0),2)
        
    for blb in blobs:
        (x,y,w,h) = cv2.boundingRect(blb)
        # if not os.path.exists(temp_dir):
        #     os.makedirs(temp_dir)
        
        img_patch=rgb_img[y:y+h,x:x+w]

        predictions=image_classification(img_patch, network, class_names)
        predict=predictions[0]
        conf=predict[1]
        fish_label_det=predict[0]
        if( fish_label_det=='fish' and conf>=0.85):
            
            # predict=predictions[0]
            rgb_copy=cv2.rectangle(rgb_copy,(x,y),(x+w,y+h),(255,12,0),2)
            cv2.putText(rgb_copy,fish_label_det,(int(x+2+w/2),int(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),3,cv2.LINE_AA)
            x = (x+w/2.0) / img_w
            y = (y+h/2.0) / img_h
            w = float(w) / img_w
            h = float(h) / img_h
            fish_specie=0
            tmp = [fish_specie, x, y, w, h]
            
            obj_arr.append(tmp)
    xml_content = ""
    for obj in obj_arr:
        xml_content += "%d %f %f %f %f\n" % (obj[0], obj[1], obj[2], obj[3], obj[4])
    if not os.path.exists(sot_classifier_dir):
        os.makedirs(sot_classifier_dir)
    f = open(join(sot_classifier_dir,filename).split('.png')[0]+'.txt', "w")
    f.write(xml_content)
    f.close()
    cv2.imwrite(join(sot_classifier_dir,filename),rgb_copy)
        
print('GMM_optical combination and classification is done ')
