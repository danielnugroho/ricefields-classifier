# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 11:32:36 2022

@author: dnugr


Sample
--model proto_model_lbs.h5 --train TRAINING_RASTER_D15.tif --class CLASS_NEW_D15_2021 --dataset TRAIN-D15 --step 1 --cores 12 --region 10


np.save(basic_name + "-MARKER.npy", [400])

"""

# PREDICTOR NEW METHOD


import rasterio
import argparse
import multiprocessing
import os
import spectral as sp
import glob
import tensorflow as tf
import numpy as np
from pathlib import Path
from affine import Affine
import sys
import time
import ray
import psutil
import signal

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# this class is a Ray Actor - which basically a stateful worker
# -------------------------------------------------------------------
@ray.remote
class infer():
    
    #import tensorflow as tf
    
    def __init__(self, mj, w):
        with tf.device('/device:CPU:0'):
            self.m = tf.keras.models.model_from_json(mj)
            self.m.set_weights(w)
   
    @ray.method(num_returns=1)
    def predict(self, ts, ids):

        with tf.device('/device:CPU:0'):
            prob = self.m.predict(ts)[0]
            
        return prob, ids


# handling ctrl-C

def handler(signum, frame):
    #res = input("Ctrl-c was pressed. Do you really want to exit? y/n ")
    #if res == 'y':
    #    exit(1)
    print("Ctrl-C was pressed and intercepted. To exit, close the command window.")

# function to write resulting TIFF files 
# -------------------------------------------------------------------
    
def write_results(trainpath, classpath):

    basic_name = Path(classpath).stem

    with rasterio.open(trainpath) as src:

        # affine img transformation to convert pixel coords to world coords
        imgtransform = Affine(gsd * stepsize, 0.0, 
                              left, 0.0, -gsd * stepsize, top)

        '''
        Affine(a, b, c, d, e, f)
        
        a = width of a pixel
        b = row rotation (typically zero)
        c = x-coordinate of the upper-left corner of the upper-left pixel
        d = column rotation (typically zero)
        e = height of a pixel (typically negative)
        f = y-coordinate of the of the upper-left corner of the upper-left pixel
        '''

        class_ras_meta = src.profile
        proba_ras_meta = src.profile

        # 8 bit channel (integer)
        class_ras_meta.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw',
            transform = imgtransform,
            width = cols,
            height = rows)

        # 32 bit channel (float)
        proba_ras_meta.update(
            dtype=rasterio.float32,
            count=1,
            compress='lzw',
            transform = imgtransform,
            width = cols,
            height = rows)

    # save raster files

    print("Writing TIFFs...")
    
    with rasterio.open(basic_name + '-CLASS.tif', 'w', **class_ras_meta) as dst:
        dst.write(dataset_class, indexes=1)

    with rasterio.open(basic_name + '-PROBA1.tif', 'w', **proba_ras_meta) as dst:
        dst.write(dataset_proba1, indexes=1)

    with rasterio.open(basic_name + '-PROBA2.tif', 'w', **proba_ras_meta) as dst:
        dst.write(dataset_proba2, indexes=1) 


# START THE PROGRAM HERE


# show information
print("Python version     : " + sys.version)
print("TensorFlow version : " + tf.__version__)
print("Keras version      : " + tf.keras.__version__)
print("CPU count          : " + str(multiprocessing.cpu_count()))
#print('''This cluster consists of
#    {} nodes in total
#    {} CPU resources in total
#'''.format(len(ray.nodes()), ray.cluster_resources()['CPU']))

# start interrupt handler
signal.signal(signal.SIGINT, handler)

# construct the argument parse and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-t", "--train", required=True,
	help="path to training class")
ap.add_argument("-c", "--class", required=True,
	help="path to class raster")
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
ap.add_argument("-s", "--step", required=True,
	help="step size")
ap.add_argument("-co", "--cores", required=True,
	help="processor cores")
ap.add_argument("-r", "--regions", required=True,
	help="image regions")

args = vars(ap.parse_args())

modelpath = args["model"]
trainpath = args["train"]
classpath = args["class"]
datasetpath = args["dataset"]
stepsize = int(args['step'])
cores = int(args['cores'])
regions = int(args['regions'])

# construct the paths
curpath = os.getcwd()
fullpath_VH = curpath +  "//" + datasetpath + "//" + "*VH.hdr"
fullpath_VV = curpath +  "//" + datasetpath + "//" + "*VV.hdr"

# grab the image paths and randomly shuffle them
imagePaths_VH = sorted(list(glob.glob(fullpath_VH)))
imagePaths_VV = sorted(list(glob.glob(fullpath_VV)))

# basic/core name for the file
basic_name = Path(classpath).stem

# number of files
series_length = len(imagePaths_VH)

# open the class image to ensure coverage/dimension of the image
image_class = rasterio.open(trainpath)
image_class_arr = image_class.read(1)

imgheight = image_class.height
imgwidth = image_class.width

# get pixel resolution & image boundaries
gsd = image_class.res[0]
left, bottom, right, top = image_class.bounds

# initialize class & probability arrays
dataset_class = np.zeros((imgheight, imgwidth), dtype='uint8')
dataset_proba1 = np.zeros((imgheight, imgwidth), dtype='float32')
dataset_proba2 = np.ones((imgheight, imgwidth), dtype='float32')

rows, cols = dataset_class.shape
print(rows, cols)
pixels = cols * rows

print("Image size: ", imgheight, imgwidth)
print("Timeseries: ", str(series_length) + " frames.")
print("Split into: ", str(regions) + "x" + str(regions) + " regions.")


# load and jsonize model so that it can be fetched into a Ray Actor
model = tf.keras.models.load_model(modelpath)

# get both the model and the trained weights
model_json = model.to_json()
model_weight = model.get_weights()

            
# if marker file exists, then continue working from the saved file.
serial_marker = 0
completed_marker = 0
all_markers = regions * regions

if os.path.isfile(basic_name + "-MARKER.npy"):
    coords = np.load(basic_name + "-MARKER.npy")
    
    completed_marker = coords[0]
    
    if (completed_marker == all_markers):
        print("Recovering completed inference.")
    else:
        print("Recovering previous work from region " + str(completed_marker).zfill(3) + "...")    

else:
    serial_marker = 0
    completed_marker = -1
    print("Starting new work...")
    
# 3D arrays for VV & VH images
img_VH = np.empty((imgheight, imgwidth, series_length),np.float16)
img_VV = np.empty((imgheight, imgwidth, series_length),np.float16)
simage = np.empty((imgheight, imgwidth),np.float16)
    
# load images into array for both VH and VV
print("Reading VH timeseries...")
for imagePath in imagePaths_VH:        
    simage = (sp.open_image(imagePath).read_band(0) + 30)/30 # it's 2D array & normalize
    img_VH[:, :, imagePaths_VH.index(imagePath)] = simage
    print(imagePath)

print("Reading VV timeseries...")
for imagePath in imagePaths_VV:        
    simage = (sp.open_image(imagePath).read_band(0) + 30)/30 # it's 2D array
    img_VV[:, :, imagePaths_VV.index(imagePath)] = simage
    print(imagePath)
    

for x in range (regions):   
    for y in range (regions):       
        
        # make sure to jump over completed work units
        
        if (serial_marker > completed_marker): 
            
            t0proc = time.perf_counter()
            pixels_heavy = 0
            pixels_light = 0
            
            imgregx_start = x * imgwidth//regions
            imgregy_start = y * imgheight//regions
            imgregx_end   = (x + 1) * imgwidth//regions
            imgregy_end   = (y + 1) * imgheight//regions
        
            print("Current region: " + str(serial_marker).zfill(3))
            print("X range", imgregx_start, imgregx_end)
            print("Y range", imgregy_start, imgregy_end)
            
            xrange = imgregx_end - imgregx_start
            yrange = imgregy_end - imgregy_start
            
            # init 3D arrays for the subset VV & VH images
            img_VH_reg = np.empty((yrange, xrange, 0),np.float16) # img region size
            img_VV_reg = np.empty((yrange, xrange, 0),np.float16) # img region size
                       
            # initialize class & probability arrays
            dataset_region_class = np.zeros((yrange, xrange), dtype='uint8')
            dataset_region_proba1 = np.zeros((yrange, xrange), dtype='float32')
            dataset_region_proba2 = np.ones((yrange, xrange), dtype='float32')
            
            # cut it to smaller region
            image_class_arr_reg = image_class_arr[imgregy_start:imgregy_end, imgregx_start:imgregx_end]
                        
            # subset image timeseries into array for both VH and VV
            print("Subsetting VH timeseries...")
            for imagePath in imagePaths_VH:        
                img_VH_reg = img_VH[imgregy_start:imgregy_end, imgregx_start:imgregx_end] # cut it to smaller region
                #print(imagePath)
    
            print("Subsetting VV timeseries...")
            for imagePath in imagePaths_VV:        
                img_VV_reg = img_VV[imgregy_start:imgregy_end, imgregx_start:imgregx_end] # cut it to smaller region
                #print(imagePath)
            
            # start working on parallel inference -----------------------------           
            # turn on Ray for parallel processing
            
            if not ray.is_initialized():
                try:
                    ray.init(num_cpus = cores, 
                             log_to_driver=False,
                             include_dashboard=False)
                except:
                    print("Exception during Ray initialization!")
                    print("Program terminated.")
                    sys.exit()
            
            # try-except block is used to recover from this plague:
            # "TimeoutError: Timed out while waiting for node to startup."
            
            # put them into Ray storage for quick access by all workers
            mdl = ray.put(model_json)
            wgt = ray.put(model_weight)
            
            # create stateful workers (actors) so that all processor cores utilized
            print("Creating " + str(cores) + " parallel processes...")
            infer_actors = []

            for c in range(cores):
                infer_actors.append(infer.remote(mdl, wgt))    

            # main loop to process the raster stack
            for i in range(xrange):
                
                # initialize serials
                scount = 0
                futures = []
                ylist = []
            
                for j in range(yrange):
                    
                    # check memory usage to prevent OOM error due to Ray leak
                    memused = psutil.virtual_memory().percent

                    # if memory usage beyond threshold, then reset Ray
                    if (memused > 90):
                        
                        # reset Ray
                        print("Shutting down ray due to memory usage...")
                        
                        if ray.is_initialized():
                            try:
                                ray.shutdown()
                            except:
                                print("Exception during Ray termination!")
                                print("Program terminated.")
                                sys.exit()
                                
                        wait_count = 0

                        while ray.is_initialized():
                            time.sleep(1)
                            wait_count += 1
                            if wait_count > 10:
                                print("Unable to shut down Ray!")
                                print("Program terminated after 10s of wait.")
                                sys.exit()


                        print("Restarting ray...")
                        ray.init(num_cpus = cores, 
                                 log_to_driver=False,
                                 include_dashboard=False)

                        wait_count = 0

                        while not ray.is_initialized():
                            time.sleep(1)
                            wait_count += 1
                            if wait_count > 10:
                                print("Unable to restart Ray!")
                                print("Program terminated after 10s of wait.")
                                sys.exit()

                        # put them into Ray storage for quick access by all workers
                        mdl = ray.put(model_json)
                        wgt = ray.put(model_weight)    
                    
                        # create stateful workers (actors) so that all processor cores utilized
                        print("Creating " + str(cores) + " parallel processes...")
                        infer_actors = []
                        
                        for c in range(cores):
                            infer_actors.append(infer.remote(mdl, wgt))    
                    
            
                    # if memory not an issue, then do this
                    
                    try: 
                        pix_class = image_class_arr_reg[j,i]
    
                        if (pix_class < 2):
                            
                            timeseries = np.empty((1, series_length, 2),np.float16)
                            pixels_heavy += 1
                            
                            # extract timeseries from imagestack
                            for z in range(series_length):
                                timeseries[0, z, 0] = img_VH_reg[j,i,z]
                                timeseries[0, z, 1] = img_VV_reg[j,i,z]
               
                            # this part is to fetch data into each actors equally
                            selector = scount % cores
                    
                            for actor in infer_actors:
                                
                                index = infer_actors.index(actor)
                                
                                if (selector == index):
                                    futures.append(actor.predict.remote(timeseries, j))
                            
                            scount += 1
                        
                        else:
                            pixels_light += 1
                            
                    except OSError as error :
                        print("Generic OS error has occurred!")
                        print(error)
            
                # execute the actors parallely along y axis
                # proba = ray.get(futures) ---> THIS IS HOW WE DID IT IN THE PAST
                # ray.get() --> blocking
                # ray.wait() --> non-blocking
                

                while len(futures): 
                    done_id, futures = ray.wait(futures)
                    
                    proba, ids = ray.get(done_id[0])
            
                    # fetch inferrence results into the array
                    (proba2, proba1) = proba
                    j = ids
                
                    # stepping/stride count
                    stepx = i
                    stepy = j
                    
                    class_value = 1 if proba1 > proba2 else 0
                    
                    # fill in the dataset array for class and both probability
                    dataset_region_class[stepy][stepx] = class_value
                    dataset_region_proba1[stepy][stepx] = proba1
                    dataset_region_proba2[stepy][stepx] = proba2              
            
            print("Saving progress...")
            np.savez_compressed(
                basic_name + "-PROC_" + str(serial_marker).zfill(3), 
                a=dataset_region_class, b=dataset_region_proba1, 
                c=dataset_region_proba2)
            
            # save progress marker
            savecoords = [serial_marker]
            f = np.save(basic_name + "-MARKER.npy", savecoords)
            
            # disable ray shutdown on each column to minimize time overhead
            # ray.shutdown()
            
            t1proc = time.perf_counter()
            timerproc = t1proc - t0proc
            
            progress = (serial_marker+1) * 100.0 / all_markers
            
            avg_speed = 1000.0 * timerproc / (pixels_heavy + pixels_light)
            
            print("Work progress    : " + str(progress).zfill(3) + "%")
            print("Predicted pixels : " + str(pixels_heavy))
            print("Skipped pixels   : " + str(pixels_light))
            print("Elapsed time     : " + str(timerproc) + " seconds.")
            print("Average speed    : " + str(avg_speed) + " ms / pixel")
            print()

        # increment the work unit marker   
        serial_marker += 1


# get ready for the conversion from NPZ files to single TIFF

serial_marker = 0

for x in range (regions):
    for y in range (regions):
        imgregx_start = x * imgwidth//regions
        imgregy_start = y * imgheight//regions
        imgregx_end   = (x + 1) * imgwidth//regions
        imgregy_end   = (y + 1) * imgheight//regions
    
        print("X range", imgregx_start, imgregx_end)
        print("Y range", imgregy_start, imgregy_end)
        
        xrange = imgregx_end - imgregx_start
        yrange = imgregy_end - imgregy_start
        
        
        # load progress dataset
        
        file_path = basic_name + "-PROC_" + str(serial_marker).zfill(3) + ".npz"
        
        if os.path.exists(file_path):    
            dataset_progress_class = np.load(file_path)['a']
            dataset_progress_proba1 = np.load(file_path)['b']
            dataset_progress_proba2 = np.load(file_path)['c']
        else:
            print("Missing data.")
            dataset_progress_class = np.zeros((xrange, yrange))
            dataset_progress_proba1 = np.zeros((xrange, yrange))
            dataset_progress_proba2 = np.zeros((xrange, yrange))

        # convert 3D variable into 2D and insert it to the bigger 2D variable.
        
        dataset_class[imgregy_start:imgregy_end, imgregx_start:imgregx_end] = np.asarray(
            dataset_progress_class).reshape(yrange, xrange)
        dataset_proba1[imgregy_start:imgregy_end, imgregx_start:imgregx_end] = np.asarray(
            dataset_progress_proba1).reshape(yrange, xrange)
        dataset_proba2[imgregy_start:imgregy_end, imgregx_start:imgregx_end] = np.asarray(
            dataset_progress_proba2).reshape(yrange, xrange)

        # increment the work unit marker        
        serial_marker += 1


write_results(trainpath, classpath)

# delete marker file to signify a complete task
if os.path.exists(basic_name + "-MARKER.npy"):
  os.remove(basic_name + "-MARKER.npy")

print("Inference process completed.")

tf.keras.backend.clear_session()

if ray.is_initialized():
    try:
        ray.shutdown()
    except:
        print("Exception during Ray termination!")
        print("Program terminated.")
        sys.exit()

