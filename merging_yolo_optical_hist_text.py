#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 11:49:12 2022

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


# yolo_hist='/home/ahsanjalal/ozfish/yolo_results_hist_test'
yolo_hist='data/test_yolo_images'
optical_kmeans_hist='data/optical_kmean_24_classified_10k_085' 
save_dir='data/gmm_optical_yolo_merge_text'
a=open('test_list.txt','r')
video_used=a.readlines()

count=0
for vids in video_used:
    count+=1
    print(count)
    video_name=vids.rstrip()
    video_name=video_name.split('/')[-1]
    v_split=video_name.split('.')
    frame_from_vid=int(v_split[2])
    # frame_on_gt=frame_from_vid+60
    frame_on_gt=frame_from_vid
    img_name=v_split[0]+'.'+v_split[1]+'.'+str(frame_on_gt)+'.txt'
    # frame=cv2.imread(join(rgb_dir,v_split[0]+'.'+v_split[1]+'.'+str(frame_on_gt)+'.png'))
    yolo_flag=0
    optical_flag=0
    
    if not os.path.exists(join(yolo_hist,img_name)):
        yolo_txt=[]
    else:
        a=open(join(yolo_hist,img_name),'r')
        yolo_txt=a.readlines()
        a.close()
        yolo_flag=1
    if not os.path.exists(join(optical_kmeans_hist,img_name)):
        optical_txt=[]
    else:
        b=open(join(optical_kmeans_hist,img_name),'r')
        optical_txt=b.readlines()
        b.close()
        optical_flag=1
    
    if yolo_flag==1 and optical_flag==1:
        for i in optical_txt:
            yolo_txt.append(i)
    elif yolo_flag==1 and optical_flag==0:
        f=1
    elif yolo_txt==0 and optical_txt==1:
        yolo_txt=optical_txt
        
    xml_content = ""
    for obj in yolo_txt:
        obj=obj.rstrip()
        obj=obj.split(' ')
        xml_content += "%d %f %f %f %f\n" % (int(obj[0]), float(obj[1]), float(obj[2]), float(obj[3]), float(obj[4]))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f = open(join(save_dir,video_name).split('.png')[0]+'.txt', "w")
    f.write(xml_content)
    f.close()
        
    
print('done')
    