#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:53:15 2022

@author: ahsanjalal
"""


from PIL import Image
import numpy as np
import cv2,os
from pylab import *
from os.path import join, isfile
import sys,os,glob
from ctypes import *
import math
import random
from sklearn.cluster import KMeans
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
n_colors =24


# img_name='/home/ahsanjalal/ozfish/G000370_L.avi.90124.png'
# image=Image.open(img_name)
# sample_img=imread(img_name)
# result = image.convert('P', palette=Image.ADAPTIVE, colors=10)
gmm_save_dir='data/kmeans_optical_dense_24_color_100000'
if not os.path.exists(gmm_save_dir):
    os.makedirs(gmm_save_dir)

vid_counter=0
data_dir='data/optical_dense_results_hist'
a=open('test_list.txt','r')
video_used=a.readlines()
for vids in video_used:
    video_name=vids.rstrip()
    video_name1=video_name.split('/')[-1]
    v_split=video_name1.split('.')
    frame_on_gt=int(v_split[2])
    counter=0
    if not os.path.exists(join(gmm_save_dir,v_split[0]+'.'+v_split[1]+'.'+str(frame_on_gt)+'.png')):
        vid_counter+=1
        print(str(vid_counter)+'/'+str(len(video_used)))
        
            
        sample_img=imread(join(data_dir,video_name1))
        w,h,_ = sample_img.shape
        sample_img = sample_img.reshape(w*h,3)
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(sample_img)
        
        # find out which cluster each pixel belongs to.
        labels = kmeans.predict(sample_img)
        n_bins=max(labels)+1
        hist1,ranges,o_info=plt.hist(labels,bins=n_bins)
        list1=list(hist1)
        max_val=max(list1)
        max_index = list1. index(max_val) 
        # the cluster centroids is our color palette
        identified_palette = np.array(kmeans.cluster_centers_).astype(int)
        new_list=[]
        # here we are removing all color blobs greater thatn 100000
        for i in list1:
            if i >100000:
                new_list.append(0)
            else:
                new_list.append(1)
        for j in range(len(new_list)):
            identified_palette[j,:]=identified_palette[j,:]*new_list[j]
        
        iden_ori=np.copy(identified_palette)
        # identified_palette[max_index,:]=0
        # recolor the entire image
        recolored_img = np.copy(sample_img)
        recolored_img_ori = np.copy(sample_img)
        for index in range(len(recolored_img)):
            recolored_img[index] = identified_palette[labels[index]]
            recolored_img_ori[index]=iden_ori[labels[index]]
            
        # reshape for display
        recolored_img = recolored_img.reshape(w,h,3)
        recolored_img_ori = recolored_img_ori.reshape(w,h,3)
        a=open(join(gmm_save_dir,v_split[0]+'.'+v_split[1]+'.'+str(frame_on_gt)+'.txt'),'w')
        for vals in hist1:
            a.write(str(vals)+'\n')
        a.close()
        cv2.imwrite(join(gmm_save_dir,v_split[0]+'.'+v_split[1]+'.'+str(frame_on_gt)+'.png'),recolored_img)
    else:
        vid_counter+=1
        # cv2.imwrite(join(gmm_save_dir,v_split[0]+'.'+v_split[1]+'.'+str(frame_on_gt)+'_original.png'),recolored_img_ori)
# imsave('kmeans_color_q.jpg', recolored_img)