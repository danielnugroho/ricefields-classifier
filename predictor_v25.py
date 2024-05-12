# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 11:32:36 2022

@author: dnugr


Sample
--model Indramayu_model_U256L3B32_lbs.keras --class CLASS_B4-2022 --datacube B4-2022_DATACUBE.nc --region 1

v0.25 May 12, 2024
    - No longer require WKT text file
    - Instead, this code uses the "bounds" global attribute from NetCDF4 file
    - the "bounds" global attribute is in WKT format

v0.24 May 1, 2024
    - No longer require training raster TIFF file as "template" for output
      GeoTIFF file
    - WKT text file as bounding box is used instead
    - dropped --cores, --steps, and --train parameters.
    - imgheight and imgwidth is now deduced from datacube
    - gsd is inferred from image width (pix) and bound width (degrees)

v0.23 April 15, 2024
    - uses Numpy's vectorized arrays operations to eliminate explicit loops,
      resulting in 100% speed improvement.
    - "heavy" and "light" pixels terminology are no longer used. All pixels are 
      processed, regardless of its classification mask (ocean or land) as this
      is way more efficient.
    - all temporary npz files are deleted afer a successfull operation.

v0.22 April 14, 2024
    - uses S1 processed image stack in NetCDF4 format instead of individual *.img files
    

v0.21 April 10, 2024
  - much faster processing by discarding Ray, and use internal Keras instead
  - using Keras enables the GPU usage, which drastically improve performance
  - trimming unecessary variables assignments
  - what took 78 hrs now only takes 20 minutes

v0.21 April 10, 2024
  - updated to Pytorch instead of Tensorflow
  - Ray runs very painfully slow.
  
np.save(basic_name + "-MARKER.npy", [400])

"""

# PREDICTOR NEW METHOD

import os
import sys
import time
from pathlib import Path
import glob
import multiprocessing
import argparse
import netCDF4 as nc
import rasterio
from rasterio.transform import from_origin

import numpy as np
from shapely.wkt import loads

os.environ["KERAS_BACKEND"] = "torch"

import torch
import keras


# function to write resulting TIFF files
# -------------------------------------------------------------------

def write_results(classpath, bounds, imgheight, imgwidth):

    basic_name = Path(classpath).stem

    # Define the raster size and resolution
    res = (bounds[2] - bounds[0]) / imgwidth  # Calculate resolution based on the width and bounds

    # Define the transform
    transform = from_origin(bounds[0], bounds[3], res, res)  # Top-left corner, resolution

    # Define the number of bands and the data type
    num_bands = 1

    # save raster files
    print("Writing TIFFs...")

    with rasterio.open(basic_name + '-CLASS.tif', 'w',
                       width=imgwidth, height=imgheight, compress='deflate',
                       count=num_bands, dtype=rasterio.uint8,
                       crs='EPSG:4326',  # Define the coordinate reference system
                       transform=transform) as dst:
        dst.write(dataset_class, indexes=1)

    with rasterio.open(basic_name + '-PROBA1.tif', 'w',
                       width=imgwidth, height=imgheight, compress='deflate',
                       count=num_bands, dtype=rasterio.float32,
                       crs='EPSG:4326',  # Define the coordinate reference system
                       transform=transform) as dst:
        dst.write(dataset_proba1, indexes=1)

    with rasterio.open(basic_name + '-PROBA2.tif', 'w',
                       width=imgwidth, height=imgheight, compress='deflate',
                       count=num_bands, dtype=rasterio.float32,
                       crs='EPSG:4326',  # Define the coordinate reference system
                       transform=transform) as dst:
        dst.write(dataset_proba2, indexes=1)


# START THE PROGRAM HERE

# show information
print("Python version     : " + sys.version)
print("Pytorch version    : " + torch.__version__)
print("Keras version      : " + keras.__version__)
print("CPU count          : " + str(multiprocessing.cpu_count()))
#print('''This cluster consists of
#    {} nodes in total
#    {} CPU resources in total
#'''.format(len(ray.nodes()), ray.cluster_resources()['CPU']))


# construct the argument parse and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-c", "--class", required=True,
	help="path to class raster")
ap.add_argument("-d", "--datacube", required=True,
    help="path to input datacube")
ap.add_argument("-r", "--regions", required=True,
	help="image regions")

args = vars(ap.parse_args())

modelpath = args["model"]
classpath = args["class"]
datasetpath = args["datacube"]
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

# Access the variables and print its shape

# Open the NetCDF file
ds = nc.Dataset(datasetpath, 'r')  # 'r' is for read-only mode
series_length = ds.variables['img_VH'].shape[2]  # Replace 'variable_name' with your variable's name

# Check if the 'bounds' attribute exists
if 'bounds' in ds.ncattrs():
    # Retrieve the 'bounds' attribute
    bounds_attr = ds.getncattr('bounds')
    # Print the 'bounds' attribute
    print("Dataset bounds: ", bounds_attr)
else:
    print("The attribute 'bounds' does not exist in this dataset.")
    print("Unable to process.")
    sys.exit(0)

# Use a context manager to open and read the WKT file
#with open(boundpath, 'r') as file:
#    wkt_string = file.read()  # Read the entire contents of the file into a single string

# Load the WKT string into a shapely geometry
geometry = loads(bounds_attr)

# Get the bounds of the geometry
bounds = geometry.bounds  # Returns (minx, miny, maxx, maxy)

# Read the data from the NetCDF file and explicitly cast it to numpy float16
print("Reading datacube file...")
img_VH = np.array(ds.variables['img_VH'][:], dtype=np.float16)
img_VV = np.array(ds.variables['img_VV'][:], dtype=np.float16)

imgheight = img_VH.shape[0]
imgwidth = img_VH.shape[1]

# Optionally print some information about the arrays
print('img_VH shape:', img_VH.shape)
print('img_VV shape:', img_VV.shape)

# Close the dataset after loading the data
print("Closing datacube file...")
ds.close()

# get pixel resolution & image boundaries
gsd = (bounds[2] - bounds[0]) / imgwidth  # Calculate resolution based on the width and bounds

# initialize class & probability arrays
dataset_class = np.zeros((imgheight, imgwidth), dtype='uint8')
dataset_proba1 = np.zeros((imgheight, imgwidth), dtype='float32')
dataset_proba2 = np.ones((imgheight, imgwidth), dtype='float32')

rows, cols = dataset_class.shape
pixels = cols * rows

print("Image size: ", imgheight, imgwidth)
print("Timeseries: ", str(series_length) + " frames.")
print("Split into: ", str(regions) + "x" + str(regions) + " regions.")

# load and jsonize model so that it can be fetched into a Ray Actor
model = keras.models.load_model(modelpath)

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

# placeholder for 2D array for VV & VH images
simage = np.empty((imgheight, imgwidth),np.float16)

# main loop to work across "regions"
for x in range (regions):
    for y in range (regions):

        # make sure to jump over completed work units

        if (serial_marker > completed_marker):

            t0proc = time.perf_counter()

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

            # subset image timeseries into array for both VH and VV
            print("Subsetting VH timeseries...")
            img_VH_reg = img_VH[imgregy_start:imgregy_end, imgregx_start:imgregx_end] # cut it to smaller region

            print("Subsetting VV timeseries...")
            img_VV_reg = img_VV[imgregy_start:imgregy_end, imgregx_start:imgregx_end] # cut it to smaller region

            total_pixels = xrange * yrange
            timeseries = np.empty((total_pixels, series_length, 2),np.float16)

            # main loop to process the raster stack and put the timeseries into array

            print("Preparing timeseries array for inference...")

            # Prepare the timeseries array for storage
            # We need an array of shape (xrange * yrange, series_length, 2)
            timeseries = np.empty((xrange * yrange, series_length, 2), dtype=img_VH_reg.dtype)

            # Reshape img_VH_reg and img_VV_reg from (yrange, xrange, series_length) to (xrange * yrange, series_length)
            reshaped_VH = img_VH_reg.reshape(xrange * yrange, series_length)
            reshaped_VV = img_VV_reg.reshape(xrange * yrange, series_length)

            # Assign values to timeseries, where each slice [scount, :, 0] and [scount, :, 1] corresponds to VH and VV respectively
            timeseries[:, :, 0] = reshaped_VH
            timeseries[:, :, 1] = reshaped_VV

            print("Timeseries array prepared for inference.")

            # run inference on the timeseries array

            print("Running inference on the current region...")
            proba = model.predict(timeseries, batch_size=64000, verbose=0, steps=None, callbacks=None)

            # assign inference result back to the class/proba array
            #scount = 0
            print("Reconstructing result array...")

            # Reshape the proba array from 1D to 2D to match the spatial structure (yrange, xrange, 2)
            reshaped_proba = proba.reshape(yrange, xrange, 2)

            # Assign the class values based on the comparison of probabilities
            # Using np.where to vectorize the conditional assignment
            class_values = np.where(reshaped_proba[:, :, 1] > reshaped_proba[:, :, 0], 1, 0)

            # Fill in the result arrays directly
            dataset_region_class[:, :] = class_values
            dataset_region_proba1[:, :] = reshaped_proba[:, :, 1]
            dataset_region_proba2[:, :] = reshaped_proba[:, :, 0]

            print("Result array reconstructed.")


            print("Saving progress...")
            np.savez_compressed(
                basic_name + "-PROC_" + str(serial_marker).zfill(3),
                a=dataset_region_class, b=dataset_region_proba1,
                c=dataset_region_proba2)

            # save progress marker
            savecoords = [serial_marker]
            f = np.save(basic_name + "-MARKER.npy", savecoords)


            t1proc = time.perf_counter()
            timerproc = t1proc - t0proc

            progress = (serial_marker+1) * 100.0 / all_markers

            avg_speed = 1000.0 * timerproc / total_pixels

            print("Work progress    : " + str(progress).zfill(3) + "%")
            print("Predicted pixels : " + str(total_pixels))
            print("Elapsed time     : " + str(timerproc) + " seconds.")
            print("Average speed    : " + str(avg_speed) + " ms / pixel")
            print()

        # increment the work unit marker
        serial_marker += 1


# get ready for the conversion from NPZ files to single TIFF

print("Reconstituting partial progress...")

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

write_results(classpath, bounds, imgheight, imgwidth)

# delete marker file to signify a complete task
if os.path.exists(basic_name + "-MARKER.npy"):
    os.remove(basic_name + "-MARKER.npy")

    # Also delete all temporary npz files
    pattern = basic_name + "-PROC_*.npz"
    files_to_delete = glob.glob(pattern)

    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except OSError as e:
            print(f"Error: {e.strerror} while deleting {e.filename}")

print("Inference process completed.")

keras.backend.clear_session()

