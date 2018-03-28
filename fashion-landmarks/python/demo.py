'''
###########################################################################
# step1:
# use
#     pipeline = pipeline_init('upper')
# for initiating a fashion landmark detection pipeline
# 
# step2:
# use
#     prediction = pipeline_forword(img,pipeline)
# for getting landmarks of images
# 
###########################################################################
'''
import caffe
import os

import cv2
from pipeline_init import pipeline_init
from pipeline_forword import pipeline_forword
from pipeline_show_results import pipeline_show_results


poj_path = '/home/cjl/fashionai'
pipeline = pipeline_init('upper', poj_path)

image_path = poj_path + '/fashion-landmarks/data/FLD_upper/'

name_list = os.listdir(image_path)

for i in range(len(name_list)):
    
    img_name = image_path + name_list[i]
    if not os.path.exists(img_name):
        continue
    img = cv2.imread(img_name)#img in B-G-R mode
    
    # forward
    prediction = pipeline_forword(img, pipeline)
    
    
    # show result 
    pipeline_show_results(img, prediction)
    


# TODO: Python? release pipeline
#pipeline_release
