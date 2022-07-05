#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:48:57 2021

@author: ahsan-jalal
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



data_paths=['data/classifier_data/val_153_list.list','data/classifier_data/test_153_list.list']
data_paths_496=['data/classifier_data/val_496_list.list','data/classifier_data/test_496_list.list']
model_name_list=['efficientnet_b0_875500.weights','efficientnet_b0_2050500.weights']# 153 & 496 class models respectively 
cfg_fish=['data/classifier_data/efficientnet_b0_153.cfg','data/classifier_data/efficientnet_b0_496.cfg']
data_fish=['data/classifier_data/fish_classifier_153.data','data/classifier_data/fish_classifier_496.data']
img_dirs=['data/classifier_data/val_data_153','data/classifier_data/test_data_153']
img_dirs_496=['data/classifier_data/val_data_496','data/classifier_data/test_data_496']



""" darknet functions """


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


def prepare_batch(images, network, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_images = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
    darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    return darknet.IMAGE(width, height, channels, darknet_images)


def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    image_height, image_width, _ = check_batch_shape(images, batch_size)
    darknet_images = prepare_batch(images, network)
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_obj(detections, num, len(class_names), nms)
        predictions = darknet.remove_negatives(detections, class_names, num)
        images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
        batch_predictions.append(predictions)
    darknet.free_batch_detections(batch_detections, batch_size)
    return images, batch_predictions


def image_classification(image, network, class_names):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    darknet.free_image(darknet_image)
    return sorted(predictions, key=lambda x: -x[1])


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = name.split(".")[:-1][0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


def batch_detection_example():
    args = parser()
    check_arguments_errors(args)
    batch_size = 3
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=batch_size
    )
    image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
    images = [cv2.imread(image) for image in image_names]
    images, detections,  = batch_detection(network, images, class_names,
                                           class_colors, batch_size=batch_size)
    for name, image in zip(image_names, images):
        cv2.imwrite(name.replace("data/", ""), image)
    print(detections)





count_model=0
count_data=0


# print(model_name)
model_fish='data/classifier_data/'+model_name_list[count_model]


network, class_names, class_colors = darknet.load_network(
    cfg_fish[count_model],data_fish[count_model],model_fish,
    batch_size=1
)

true_count=0
classification_accuracy=[]
random.seed(3)  # deterministic bbox colors

# Add path to valid dataset

for path1 in data_paths:
    f_scores=[]
    mean_ap=[]
    gt_list=[]
    predicted_list=[]
    true_count=0
    a=open(os.getcwd()+'/'+path1,'r')
    val_images=a.readlines()
    count=0
    img_dir=img_dirs[count_data]
    print('Processing {} split out of {} for 153 classes'.format(str(count_data+1),str(len(data_paths))))
    for val_img_path in val_images:
        # print(count)
        count+=1
        val_img_path=val_img_path.rstrip()
        
        path_split1=val_img_path.split('.png')[0]
        gt_name=path_split1.split('_')[1]

        gt_list.append(class_names.index(gt_name))
        #images=load_images(val_img_path)
        #image_name = images[0]
        image = cv2.imread(join(img_dir,val_img_path))
        prediction=image_classification(image, network, class_names)
        predicted_list.append(class_names.index(prediction[0][0]))
        if class_names.index(gt_name) == class_names.index(prediction[0][0]):
            true_count+=1

    # my code for f-score
    count_per_label=[]
    accuracy=float(true_count)/len(val_images)
    classification_accuracy.append(accuracy)
    for i in range(len(np.unique(gt_list))):
        count_per_label.append(gt_list.count(i))
    
        
    # now do confusion matrix
    from sklearn.metrics import confusion_matrix
    cnf_matrix = confusion_matrix(gt_list, predicted_list)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)+1
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    precision= TP/(TP+FP+1)
    recall=TP/(TP+FN+1)
    pr=sum(precision)/len(precision)
    re=sum(recall)/len(recall)
    f_score=2*pr*re/(pr+re)
    f_scores.append(f_score)
    mean_ap.append(pr)
    if count_model==0 and count_data==0:
        data_name='validation_153_classes'
    elif count_model==0 and count_data==1:
        data_name='Test_153_classes'
    
    else:
        print('something wrong')
    print('Results for {} dataset'.format(data_name))
    print('fscore {} and map {} and accuracy {}'.format(f_score,pr,accuracy))
    count_data+=1
count_model+=1
count_data=0
model_fish='data/classifier_data/'+model_name_list[count_model]


network, class_names, class_colors = darknet.load_network(
    cfg_fish[count_model],data_fish[count_model],model_fish,
    batch_size=1
)

true_count=0
classification_accuracy=[]
random.seed(3)  # deterministic bbox colors

for path1 in data_paths_496:
    f_scores=[]
    mean_ap=[]
    gt_list=[]
    predicted_list=[]
    true_count=0
    a=open(os.getcwd()+'/'+path1,'r')
    val_images=a.readlines()
    count=0
    img_dir=img_dirs_496[count_data]
    print('Processing {} split out of {} from 496 classes'.format(str(count_data+1),str(len(data_paths))))
    for val_img_path in val_images:
        # print(count)
        count+=1
        val_img_path=val_img_path.rstrip()
        
        path_split1=val_img_path.split('.png')[0]
        gt_name=path_split1.split('_')[1]

        gt_list.append(class_names.index(gt_name))
        #images=load_images(val_img_path)
        #image_name = images[0]
        image = cv2.imread(join(img_dir,val_img_path))
        prediction=image_classification(image, network, class_names)
        predicted_list.append(class_names.index(prediction[0][0]))
        if class_names.index(gt_name) == class_names.index(prediction[0][0]):
            true_count+=1

    # my code for f-score
    count_per_label=[]
    accuracy=float(true_count)/len(val_images)
    classification_accuracy.append(accuracy)
    for i in range(len(np.unique(gt_list))):
        count_per_label.append(gt_list.count(i))
    
        
    # now do confusion matrix
    from sklearn.metrics import confusion_matrix
    cnf_matrix = confusion_matrix(gt_list, predicted_list)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)+1
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    precision= TP/(TP+FP+1)
    recall=TP/(TP+FN+1)
    pr=sum(precision)/len(precision)
    re=sum(recall)/len(recall)
    f_score=2*pr*re/(pr+re)
    f_scores.append(f_score)
    mean_ap.append(pr)
    if count_model==1 and count_data==0:
        data_name='validation_496_classes'
    elif count_model==1 and count_data==1:
        data_name='Test_496_classes'
    else:
        print('something wrong')
    print('Results for {} dataset'.format(data_name))
    print('fscore {} and map {} and accuracy {}'.format(f_score,pr,accuracy))
    count_data+=1






