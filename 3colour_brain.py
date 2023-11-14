#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 12:57:38 2021

@author: Mathew
"""

from skimage.io import imread
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import filters,measure
from skimage.filters import threshold_local



# Details of what to look for in the filenames - i.e. what the filenames contain

image_one="PSD93"
image_two="PSD95"
image_three="SAP102"

# Paths to the files

pathlist=[]

pathlist.append(r"/Users/Mathew/Documents/Current analysis/Triple_mouse_example_images/1/")
pathlist.append(r"/Users/Mathew/Documents/Current analysis/Triple_mouse_example_images/2/")



def load_image(toload):
    
    image=imread(toload)
    
    return image

def z_project(image):
    
    mean_int=np.mean(image,axis=0)
    
    return mean_int

# Subtract background:
def subtract_bg(image):
    background = threshold_local(image, 11, offset=np.percentile(image, 1), method='median')
    bg_corrected =image - background
    return bg_corrected

def threshold_image_otsu(input_image):
    threshold_value=filters.threshold_otsu(input_image)  
    # threshold_value=filters.threshold_triangle(input_image)  
 
    print(threshold_value)
    binary_image=input_image>threshold_value
    
    return threshold_value,binary_image


def threshold_image_std(input_image):
   
    
    threshold_value=input_image.mean()+2*input_image.std()
    print(threshold_value)
    binary_image=input_image>threshold_value
    
    return threshold_value,binary_image

def threshold_image_standard(input_image,thresh):
    
    binary_image=input_image>thresh
    
    return binary_image

# Threshold image using otsu method and output the filtered image along with the threshold value applied:

def threshold_image_fixed(input_image,threshold_number):
    threshold_value=threshold_number   
    binary_image=input_image>threshold_value
    
    return threshold_value,binary_image

# Label and count the features in the thresholded image:
def label_image(input_image):
    labelled_image=measure.label(input_image)
    number_of_features=labelled_image.max()
    
    return number_of_features,labelled_image

# Function to show the particular image:
def show(input_image,color=''):
    if(color=='Red'):
        plt.imshow(input_image,cmap="Reds")
        plt.show()
    elif(color=='Blue'):
        plt.imshow(input_image,cmap="Blues")
        plt.show()
    elif(color=='Green'):
        plt.imshow(input_image,cmap="Greens")
        plt.show()
    else:
        plt.imshow(input_image)
        plt.show() 


# Take a labelled image and the original image and measure intensities, sizes etc.
def analyse_labelled_image(labelled_image,original_image):
    measure_image=measure.regionprops_table(labelled_image,intensity_image=original_image,properties=('area','perimeter','centroid','orientation','major_axis_length','minor_axis_length','mean_intensity','max_intensity'))
    measure_dataframe=pd.DataFrame.from_dict(measure_image)
    return measure_dataframe

# This is to look at coincidence purely in terms of pixels

def coincidence_analysis_pixels(binary_image1,binary_image2):
    pixel_overlap_image=binary_image1&binary_image2         
    pixel_overlap_count=pixel_overlap_image.sum()
    pixel_fraction=pixel_overlap_image.sum()/binary_image1.sum()
    
    return pixel_overlap_image,pixel_overlap_count,pixel_fraction

# Look at coincidence in terms of features. Needs binary image input 

def feature_coincidence(binary_image1,binary_image2):
    number_of_features,labelled_image1=label_image(binary_image1)          # Labelled image is required for this analysis
    coincident_image=binary_image1 & binary_image2        # Find pixel overlap between the two images
    coincident_labels=labelled_image1*coincident_image   # This gives a coincident image with the pixels being equal to label
    coinc_list, coinc_pixels = np.unique(coincident_labels, return_counts=True)     # This counts number of unique occureences in the image
    
    
    # coinc_list = np.delete(coinc_list, 0, axis=0)
    # coinc_pixels = np.delete(coinc_pixels, 0, axis=0)
    
    # Now for some statistics
    total_labels=labelled_image1.max()
    total_labels_coinc=len(coinc_list)
    fraction_coinc=total_labels_coinc/total_labels
    
    # Now look at the fraction of overlap in each feature
    # First of all, count the number of unique occurances in original image
    label_list, label_pixels = np.unique(labelled_image1, return_counts=True)

    # label_list = np.delete(label_list, 0, axis=0)
    # label_pixels = np.delete(label_pixels, 0, axis=0)
        
    fract_pixels_overlap=[]
    for i in range(len(coinc_list)):
        overlap_pixels=coinc_pixels[i]
        label=coinc_list[i]
        total_pixels=label_pixels[label]
        fract=1.0*overlap_pixels/total_pixels
        fract_pixels_overlap.append(fract)
    
    
    # Generate the images
    coinc_list[0]=1000000   # First value is zero- don't want to count these. 
    coincident_features_image=np.isin(labelled_image1,coinc_list)   # Generates binary image only from labels in coinc list
    coinc_list[0]=0
    non_coincident_features_image=~np.isin(labelled_image1,coinc_list)  # Generates image only from numbers not in coinc list.
    
    return coinc_list,coinc_pixels,fraction_coinc,coincident_features_image


# Go through all files

for path in pathlist:
    
    # This looks for the files:
        
    for root, dirs, files in os.walk(path):
        for name in files:
             if image_one in name:
                    name1=name
             if image_two in name:
                    name2=name
             if image_three in name:
                    name3=name
    
    # Load images
    image1=load_image(path+name1)
    image2=load_image(path+name2)
    image3=load_image(path+name3)
    
    # Just extract the image:
    image1_dims=image1[0,0,0,:,:]
    image2_dims=image2[0,0,0,:,:]
    image3_dims=image3[0,0,0,:,:]
    
    # Subtract bg
    
    image1_bg=subtract_bg(image1_dims)
    image2_bg=subtract_bg(image2_dims)
    image3_bg=subtract_bg(image3_dims)
    
    # Threshold each image:
    
    thresh1,image1_binary=threshold_image_otsu(image1_bg)
    thresh2,image2_binary=threshold_image_otsu(image2_bg)
    thresh3,image3_binary=threshold_image_otsu(image3_bg)
    
    # Save the thresholded images
    
    imsr1 = Image.fromarray(image1_binary)
    imsr1.save(path+image_one+'_binary.tif')
    
    imsr2 = Image.fromarray(image2_binary)
    imsr2.save(path+image_two+'_binary.tif')
    
    imsr3 = Image.fromarray(image3_binary)
    imsr3.save(path+image_three+'_binary.tif')
    
    # Need to label the images
    
    number1,image1_labelled=label_image(image1_binary)
    number2,image2_labelled=label_image(image2_binary)
    number3,image3_labelled=label_image(image3_binary)
    
    # Save the labelled images:
        
    imsr1 = Image.fromarray(image1_labelled)
    imsr1.save(path+image_one+'_labelled.tif')
    
    imsr2 = Image.fromarray(image2_labelled)
    imsr2.save(path+image_two+'_labelled.tif')
    
    imsr3 = Image.fromarray(image3_labelled)
    imsr3.save(path+image_three+'_labelled.tif')
        
        
    # Perform measurements on the images
    
    image1_measurements=analyse_labelled_image(image1_labelled,image1_dims)
    image2_measurements=analyse_labelled_image(image2_labelled,image2_dims)
    image3_measurements=analyse_labelled_image(image3_labelled,image3_dims)
    
    
    # Check for coincidence between image1 and the others:
        
    coinc_list12,coinc_pixels12,fraction_coinc12,coincident_features_image12=feature_coincidence(image1_binary,image2_binary)
    
    # Need to make list with coincident clusters
    
    length_of_list=image1_labelled.max()
    
    coincident12 = [0 for _ in range(length_of_list)]
    overlap12 = [0 for _ in range(length_of_list)]
    
    for i,j in zip(coinc_list12,coinc_pixels12):
        if i>0:
            coincident12[i-1]=1
            overlap12[i-1]=j
    
    
    image1_measurements['Coincident with image 2']=coincident12
    image1_measurements['Overlap with image 2']=overlap12
    
    coinc_list13,coinc_pixels13,fraction_coinc13,coincident_features_image13=feature_coincidence(image1_binary,image3_binary)
    
    # Need to make list with coincident clusters
    
    length_of_list=image1_labelled.max()
    
    coincident13 = [0 for _ in range(length_of_list)]
    overlap13 = [0 for _ in range(length_of_list)]
    
    for i,j in zip(coinc_list13,coinc_pixels13):
        if i>0:
            coincident13[i-1]=1
            overlap13[i-1]=j
    
    image1_measurements['Coincident with image 3']=coincident13
    image1_measurements['Overlap with image 3']=overlap13
    
       
    image1_measurements.to_csv(path + image_one+'_Metrics.csv', sep = '\t')

#  Now image 2


    coinc_list21,coinc_pixels21,fraction_coinc21,coincident_features_image21=feature_coincidence(image2_binary,image1_binary)
      
      # Need to make list with coincident clusters
      
    length_of_list=image2_labelled.max()
      
    coincident21 = [0 for _ in range(length_of_list)]
    overlap21 = [0 for _ in range(length_of_list)]
      
    for i,j in zip(coinc_list21,coinc_pixels21):
          if i>0:
              coincident21[i-1]=1
              overlap21[i-1]=j
      
      
    image2_measurements['Coincident with image 1']=coincident21
    image2_measurements['Overlap with image 1']=overlap21
      
    coinc_list23,coinc_pixels23,fraction_coinc23,coincident_features_image23=feature_coincidence(image2_binary,image3_binary)
      
      # Need to make list with coincident clusters
      
    length_of_list=image2_labelled.max()
      
    coincident23 = [0 for _ in range(length_of_list)]
    overlap23 = [0 for _ in range(length_of_list)]
      
    for i,j in zip(coinc_list23,coinc_pixels23):
          if i>0:
              coincident23[i-1]=1
              overlap23[i-1]=j
      
    image2_measurements['Coincident with image 3']=coincident23
    image2_measurements['Overlap with image 3']=overlap23
      
     
    image2_measurements.to_csv(path + image_two+'_Metrics.csv', sep = '\t')
    
    
    
    #  Now image 3
    
    
    coinc_list31,coinc_pixels31,fraction_coinc31,coincident_features_image31=feature_coincidence(image3_binary,image1_binary)
      
      # Need to make list with coincident clusters
      
    length_of_list=image3_labelled.max()
      
    coincident31 = [0 for _ in range(length_of_list)]
    overlap31 = [0 for _ in range(length_of_list)]
      
    for i,j in zip(coinc_list31,coinc_pixels31):
          if i>0:
              coincident31[i-1]=1
              overlap31[i-1]=j
      
      
    image3_measurements['Coincident with image 1']=coincident31
    image3_measurements['Overlap with image 1']=overlap31
      
    coinc_list32,coinc_pixels32,fraction_coinc32,coincident_features_image32=feature_coincidence(image3_binary,image2_binary)
      
      # Need to make list with coincident clusters
      
    length_of_list=image3_labelled.max()
      
    coincident32 = [0 for _ in range(length_of_list)]
    overlap32 = [0 for _ in range(length_of_list)]
      
    for i,j in zip(coinc_list32,coinc_pixels32):
          if i>0:
              coincident32[i-1]=1
              overlap32[i-1]=j
      
    image3_measurements['Coincident with image 2']=coincident32
    image3_measurements['Overlap with image 2']=overlap32
      
     
    image3_measurements.to_csv(path + image_three+'_Metrics.csv', sep = '\t')
    





