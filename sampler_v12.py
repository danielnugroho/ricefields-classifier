# -*- coding: utf-8 -*-
"""
Created on Sun April 18 2021

@author: Daniel

Minimum requirements:
    Python version     : 3.7.7
    TensorFlow version : 2.1.0
    Keras version      : 2.3.1

USAGE:
--class TRAINING_CLASS.TIF --dataset TRAIN --samples SAMPLES_2018 --step 4
--class TRAINING_CLASS.TIF --dataset TRAIN_2019 --samples SAMPLES_2019 --step 4

13 Nov 2021 (v10):
    - udpate to speed up array stacking process, much faster now
      (replacing np.vstack with np.append)
    
26 Dec 2021 (v11):
    - update with normalization (data range is from 0 dB to -30 dB)
    
27 Dec 2021 (v12):
    - numpy compressed archive format (NPZ) is used instead of CSV
      much smaller size and faster process.

"""

import rasterio
import argparse
import os
import spectral as sp
import glob
import numpy as np
import pandas as pd


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--class", required=True,
	help="path to trained model model")
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
ap.add_argument("-m", "--samples", required=True,
	help="path to CSV file samples")
ap.add_argument("-s", "--step", required=True,
	help="step size")

args = vars(ap.parse_args())

classpath = args["class"]
datasetpath = args["dataset"]
samplefile = args['samples']
stepsize = int(args['step'])

curpath = os.getcwd()
fullpath_VH = curpath +  "//" + datasetpath + "//" + "*VH.hdr"
fullpath_VV = curpath +  "//" + datasetpath + "//" + "*VV.hdr"

# grab the image paths and randomly shuffle them
imagePaths_VH = sorted(list(glob.glob(fullpath_VH)))
imagePaths_VV = sorted(list(glob.glob(fullpath_VV)))

# number of files
series_length = len(imagePaths_VH)

# open the class image to ensure coverage/dimension of the image
dataset_class = rasterio.open(classpath)
image_class = dataset_class.read(1)
imgheight = dataset_class.height
imgwidth = dataset_class.width

print("Image size: ", imgheight, imgwidth)
print("Timeseries: ", str(series_length) + " frames.")

# construct arrays
timeseries_VH = np.empty(series_length+1)
timeseries_VH.fill(0)
timeseriescol_VH = []
timeseries_VV = np.empty(series_length+1)
timeseries_VV.fill(0)
timeseriescol_VV = []

# 3D arrays for VV & VH images
img_VH = np.empty((imgheight, imgwidth, 0),np.float32)
img_VV = np.empty((imgheight, imgwidth, 0),np.float32)
simage = np.empty((imgheight, imgwidth),np.float32)

# load images into array for both VH and VV
print("Reading VH timeseries...")
for imagePath in imagePaths_VH:        
    simage = (sp.open_image(imagePath).read_band(0) + 30)/30 # it's 2D array & normalize
    simage = np.asarray(simage).reshape(imgheight,imgwidth,1) # make it 3D
    img_VH = np.dstack((img_VH, simage))
    print(imagePath)

print("Reading VV timeseries...")
for imagePath in imagePaths_VV:        
    simage = (sp.open_image(imagePath).read_band(0) + 30)/30 # it's 2D array
    simage = np.asarray(simage).reshape(imgheight,imgwidth,1) # make it 3D
    img_VV = np.dstack((img_VV, simage))
    print(imagePath)


print("Extracting classified timeseries...")
for x in range(int(stepsize/2), imgwidth-stepsize, stepsize):
    for y in range(int(stepsize/2), imgheight-stepsize, stepsize):
        
        class_value = image_class[y,x]
        
        if (class_value == 0 or class_value == 1):
            
            bad_records = False
            
            # first entry in timeseries is the class value
            timeseries_VH[0] = class_value
            timeseries_VV[0] = class_value
        
            # extract timeseries from imagestack
            for imagePath in imagePaths_VH:
                index = imagePaths_VH.index(imagePath)
                timeseries_VH[index+1] = img_VH[y,x,index]
                timeseries_VV[index+1] = img_VV[y,x,index]
                
                # check if there's any nulls
                if np.isnan(img_VH[y,x,index]) or np.isnan(img_VV[y,x,index]):
                    bad_records = True
                    break

            if not bad_records:
                timeseriescol_VH.append(timeseries_VH.tolist())
                timeseriescol_VV.append(timeseries_VV.tolist())

# write to CSV file
print("Exporting NPZ files...")
np.savez_compressed(samplefile + "_VH", timeseriescol_VH)
np.savez_compressed(samplefile + "_VV", timeseriescol_VV)

dataset_class.close()
print("Samples preparation completed.")