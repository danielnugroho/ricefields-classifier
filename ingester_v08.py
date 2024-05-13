# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:43:30 2024

@author: Daniel.Nugroho

Usage:
    
    --dataset B4-2022 --datacube B4-2022_DATACUBE.nc

    v0.8 - 240514
      - fixed problem in bounding coordinates parses that caused georeference
        problem (location shift).
        
    v0.7 - 240512
        - this code summarizes the bounding box coordinates from the first
          raster header file (*.hdr) and store this info into the
          "bounds" global attribute of the datacube
        
    v0.6 - 240501
      - no longer require TIFF file "template" to detect imgheight/width
      - instead, it detects them directly from the datacube raw files
    
    v0.4 - 240430
      - capable of ingesting and gapfilling timeseries data
      - parallel processing capability which yielded >5 times speed-up
     

"""
import os, time
import re
import glob
import argparse
from datetime import datetime
import spectral as sp
import numpy as np
import netCDF4 as nc
import multiprocessing
import ray
from scipy.interpolate import UnivariateSpline

cores = int(multiprocessing.cpu_count())

ray.init(num_cpus = cores,
         log_to_driver=False,
         include_dashboard=False)

@ray.remote
class gapfiller():

    def interpolate_slice(self, sliced, i):

        p = sliced.shape[0]
        q = sliced.shape[1]
        interpolated_values = np.zeros((p,q))

        for k in range(p):

            x = np.arange(q)
            valid_indices = np.where(~np.isnan(sliced[k]))[0]
            valid_values = sliced[k][valid_indices]

            # Fit spline on non-null data. The 's' parameter is smoothness factor;
            # s=0 would interpolate through all data points, no smoothing is done.
            # by default the spline requires at least 4 data points (cubic spline)

            if valid_indices.size > 3:  # Only fit spline if there are at least 4 non-NaN values
                spline = UnivariateSpline(valid_indices, valid_values, s=0, ext=3)
                interpolated_values[k] = spline(x)
            else:
                interpolated_values[k] = np.full(sliced[k].shape, np.nan)  # Fill with NaN if all are NaN

        return i, interpolated_values


def interpolate_cube(series_3d):
    n, m, _ = series_3d.shape
    interpolated_series_3d = np.zeros_like(series_3d)

    # create stateful workers (actors) so that all processor cores utilized
    print("Creating " + str(cores) + " parallel processes...")
    gapfiller_actors = []

    for c in range(cores):
        gapfiller_actors.append(gapfiller.remote())

    counter = 0

    # Create a list to hold references to the async results
    results = []

    for i in range(n):

        # this part is to fetch data into each actors equally
        actor_index = counter % cores

        slice_1d = series_3d[i, :, :]

        result = gapfiller_actors[actor_index].interpolate_slice.remote(slice_1d, i)

        results.append(result)

        counter += 1


    while results:
        # Use ray.wait to get the first completed task
        done, results = ray.wait(results, num_returns=1, timeout=None)

        # Retrieve result for the completed task
        result = ray.get(done[0])
        i, interpolated_values = result
        interpolated_series_3d[i, :, :] = interpolated_values

    return interpolated_series_3d


def find_first_hdr_file(directory):
    # List all files in the directory
    for file in os.listdir(directory):
        # Check if the file ends with '.hdr'
        if file.endswith('.hdr'):
            return file  # Return the first matching file
    return None  # Return None if no file is found

def get_image_size(filepath):
    # Load the hyperspectral image
    img = sp.open_image(filepath)
    # Return the spatial and spectral dimensions
    return img.shape


def get_bounds(envi_header):
    
    # Use regular expressions to extract relevant values
    samples = int(re.search(r"samples\s*=\s*(\d+)", envi_header).group(1))
    lines = int(re.search(r"lines\s*=\s*(\d+)", envi_header).group(1))
    map_info_text = re.search(r"map info\s*=\s*\{([^}]+)\}", envi_header).group(1)
    
    # Split the map info text on commas to extract the relevant data.
    map_info_parts = map_info_text.split(",")
    
    # Extract relevant data from map_info_parts
    center_pixel_x = float(map_info_parts[1])
    center_pixel_y = float(map_info_parts[2])
    center_lon = float(map_info_parts[3])
    center_lat = float(map_info_parts[4])
    pixel_width = float(map_info_parts[5])
    pixel_height = float(map_info_parts[6])
    
    # Calculate geographic coordinates of the four corners
    # Top-left corner (min_x, max_y)
    min_x = center_lon - center_pixel_x * pixel_width
    max_y = center_lat + center_pixel_y * pixel_height
    
    # Bottom-right corner (max_x, min_y)
    max_x = center_lon + (samples - center_pixel_x) * pixel_width
    min_y = center_lat - (lines - center_pixel_y) * pixel_height
    
    (min_x, max_y, max_x, min_y)
    print(min_x, max_y, max_x, min_y)
    
    
    # WKT (Well-Known Text) format for polygons requires coordinates of the corners
    # It typically starts and ends at the same point to form a closed loop.
    
    # We already have the min_x, max_y (top-left) and max_x, min_y (bottom-right).
    # We need to calculate the other two corners:
    # Bottom-left corner (min_x, min_y) and top-right corner (max_x, max_y)
    
    # Bottom-left corner
    bottom_left_x = min_x
    bottom_left_y = min_y
    
    # Top-right corner
    top_right_x = max_x
    top_right_y = max_y
    
    # Form the WKT string for a polygon
    wkt_polygon = f"POLYGON(({min_x} {max_y}, {top_right_x} {top_right_y}, {max_x} {min_y}, {bottom_left_x} {bottom_left_y}, {min_x} {max_y}))"

    return wkt_polygon

# construct the argument parse and parse the arguments

ap = argparse.ArgumentParser()

ap.add_argument("-ds", "--dataset", required=True,
	help="path to file dataset")
ap.add_argument("-dc", "--datacube", required=True,
    help="path to output datacube")

args = vars(ap.parse_args())

datacubepath = args["datacube"]
datasetpath = args["dataset"]

# construct the paths
curpath = os.getcwd()
fullpath_VH = curpath +  "//" + datasetpath + "//" + "*VH.hdr"
fullpath_VV = curpath +  "//" + datasetpath + "//" + "*VV.hdr"

first_hdr_file = find_first_hdr_file(curpath +  "//" + datasetpath + "//")
image_size = get_image_size(curpath +  "//" + datasetpath + "//" + first_hdr_file)

imgheight = image_size[0]
imgwidth = image_size[1]

# Open the file and read it into a single string variable
with open((curpath +  "//" + datasetpath + "//" + first_hdr_file), 'r') as file:
    header = file.read()
    
wkt_polygon = get_bounds(header)
print(wkt_polygon)

# grab the image paths and sort them
imagePaths_VH = sorted(list(glob.glob(fullpath_VH)))
imagePaths_VV = sorted(list(glob.glob(fullpath_VV)))

# number of files
series_length = len(imagePaths_VH)

directory = curpath +  "//" + datasetpath

# Pattern to extract date from filename
date_pattern = re.compile(r'(\d{8})')

# List to store dates
datetimes = []

# Loop through each file in the directory
for filename in imagePaths_VH:
    # Search for the date in the filename
    match = date_pattern.search(filename)
    if match:
        # Extract the date
        date_str = match.group(1)
        # Convert string to datetime object
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        # Append datetime object to the list
        datetimes.append(date_obj)

# Convert list of datetime objects to a numpy array
datetime_array = np.array(datetimes, dtype='datetime64')

print(datetime_array)

# Sort the array to find the smallest datetime
sorted_dates = np.sort(datetime_array)

# Smallest datetime
smallest_datetime = sorted_dates[0]


# Create a new datetime64 object for Jan 1st and December 31st of the same year
end_date_year = np.datetime64('{}-12-31T00:00:00.000000'.format(smallest_datetime.astype('datetime64[Y]').astype(int) + 1970))
first_date_year = np.datetime64('{}-01-01T00:00:00.000000'.format(smallest_datetime.astype('datetime64[Y]').astype(int) + 1970))

skipped_days = (smallest_datetime - first_date_year).astype('timedelta64[D]').astype(int)
first_occurence_day = skipped_days % 12
first_proper_acq_day = first_date_year + np.timedelta64(first_occurence_day, 'D')

date_list = [first_proper_acq_day]

#if (smallest_datetime > (first_date_year + np.timedelta64(12, 'D'))):
    # one or more missed acquisition at the beginning of the year
#    date_list = [smallest_datetime - np.timedelta64(12, 'D')]
#else:
    # Initialize the list with the smallest datetime
#    date_list = [smallest_datetime]

# Generate dates adding 12 days iteratively until the end date is reached
current_date = first_proper_acq_day
while current_date <= end_date_year:
    current_date += np.timedelta64(12, 'D')  # Add 12 days
    if current_date <= end_date_year:
        date_list.append(current_date)

# Convert list to a NumPy array
proper_dates = np.array(date_list)
proper_length = len(proper_dates)
missing_frames = proper_length - series_length

print(proper_dates)

# 3D arrays for VV & VH images
img_VH = np.full((imgheight, imgwidth, proper_length), np.nan, dtype=np.float16)
img_VV = np.full((imgheight, imgwidth, proper_length), np.nan, dtype=np.float16)
simage = np.full((imgheight, imgwidth),np.float16)
valid_image = np.zeros(proper_length, bool)

t0proc = time.perf_counter()

# load images into array for both VH and VV
print("Reading VH timeseries...")

for imagePath in imagePaths_VH:
    simage = (sp.open_image(imagePath).read_band(0) + 30)/30 # it's 2D array & normalize

    match = date_pattern.search(imagePath)

    if match:
        # Extract the date
        date_str = match.group(1)
        # Convert string to datetime object
        date_obj = datetime.strptime(date_str, '%Y%m%d')

        indextoput = np.where(proper_dates == date_obj)[0].item()

        img_VH[:, :, indextoput] = simage
        valid_image[indextoput] = True

        print(imagePath + " moved to index " + str(indextoput))


print("Reading VV timeseries...")
for imagePath in imagePaths_VV:
    simage = (sp.open_image(imagePath).read_band(0) + 30)/30 # it's 2D array

    match = date_pattern.search(imagePath)

    if match:
        # Extract the date
        date_str = match.group(1)
        # Convert string to datetime object
        date_obj = datetime.strptime(date_str, '%Y%m%d')

        indextoput = np.where(proper_dates == date_obj)[0].item()

        img_VV[:, :, indextoput] = simage
        valid_image[indextoput] = True

        print(imagePath + " moved to index " + str(indextoput))

t1proc = time.perf_counter()
timerproc = t1proc - t0proc

print("Image size          : " + str(imgheight) + " x " + str(imgwidth))
print("First frame date    : " + str(proper_dates[0]))
print("Last frame date     : " + str(proper_dates[-1]))
print("Proper frame length : " + str(proper_length))
print("Actual frame length : " + str(series_length))
print("Missing frames      : " + str(missing_frames))
print("Elapsed time        : " + str(timerproc) + " seconds.")
print()

t0proc = time.perf_counter()

if (missing_frames > 0):
    print("Interpolating missing frames using cubic spline...")
    intp_img_VH = interpolate_cube(img_VH).astype(np.float16)
    intp_img_VV = interpolate_cube(img_VV).astype(np.float16)
else:
    # no interpolation needed
    intp_img_VH = img_VH.astype(np.float16)
    intp_img_VV = img_VV.astype(np.float16)

t1proc = time.perf_counter()
timerproc = t1proc - t0proc

print('Interpolation completed.')
print("Elapsed time     : " + str(timerproc) + " seconds.")

t0proc = time.perf_counter()

print()
print("Exporting NetCDF files...")
nc_filename = datacubepath
ds = nc.Dataset(nc_filename, 'w', format='NETCDF4')

# Create dimensions
ds.createDimension('height', imgheight)
ds.createDimension('width', imgwidth)
ds.createDimension('frames', proper_length)

# Create variables
print("Creating variables...")
img_VH_var = ds.createVariable('img_VH', "f4", ('height', 'width', 'frames'), zlib=True, complevel=4)
img_VV_var = ds.createVariable('img_VV', "f4", ('height', 'width', 'frames'), zlib=True, complevel=4)
img_dates_var = ds.createVariable('proper_dates', "f8", ('frames'), zlib=True, complevel=4)
img_validity_var = ds.createVariable('valid_image', "i1", ('frames'), zlib=True, complevel=4)

# Write data to variables
print("Writing data to variables...")
img_VH_var[:] = intp_img_VH # put the interpolated arrays
img_VV_var[:] = intp_img_VV # both VV and VH

# Add global attributes (optional)
print("Adding global attributes...")
ds.description = 'Compressed radar images for VH and VV polarizations'
ds.history = 'Created ' + str(nc.date2num(datetime.now(), units='days since 1900-01-01', calendar='gregorian'))
ds.source = 'Generated from satellite imagery'
ds.bounds = wkt_polygon

# Close the dataset
print("Closing dataset...")
ds.close()
t1proc = time.perf_counter()
timerproc = t1proc - t0proc

print('Data saved to  :', nc_filename)
print("Elapsed time   : " + str(timerproc) + " seconds.")

ray.shutdown()
