# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 05:07:08 2024

@author: dnugr
"""
import netCDF4, os

def describe_netcdf_file(file_path):
    """
    Provides a comprehensive description of a NetCDF file, including global attributes (metadata),
    dimensions, variables and their detailed attributes. This function is aimed at giving a deep insight
    into the structure and metadata of the file for thorough analysis and documentation.
    """
    # Open the NetCDF file
    ds = netCDF4.Dataset(file_path, 'r')
       
    print("Detailed File Description:")
    print("-" * 60)
    print(f"File Path: {file_path}")
    print(f"Format: {ds.file_format}")

    # Global attributes (metadata)
    print("\nGlobal Attributes (Metadata):")
    for attr in ds.ncattrs():
        print(f"  {attr}: {getattr(ds, attr)}")
        if attr.lower() in ['description', 'history', 'source', 'bounds']:
            print(f"    Detailed {attr}: {getattr(ds, attr)}")  # Provide more context if specific key attributes are present
    
    # Dimensions information
    print("\nDimensions Information:")
    for name, dim in ds.dimensions.items():
        print(f"  Dimension '{name}': length {dim.size}, {'unlimited' if dim.isunlimited() else 'fixed size'}")

    # Variables and their detailed attributes
    print("\nVariables and Their Attributes:")
    for name, var in ds.variables.items():
        print(f"  Variable '{name}':")
        print(f"    Data type: {var.dtype}")
        print(f"    Dimensions: {var.dimensions}")
        print(f"    Total elements: {var.size}")
        # Display attributes of each variable
        print("    Variable-specific Attributes:")
        for attr_name in var.ncattrs():
            attr_value = getattr(var, attr_name)
            print(f"      {attr_name}: {attr_value}")
            # If specific attributes like 'units' or 'long_name' exist, provide additional context
            if attr_name.lower() in ['units', 'long_name', 'standard_name']:
                print(f"        Explanation: {attr_value} is used to describe the measurement units, detailed naming or standard naming convention of the data.")

    # Check if the 'bounds' attribute exists
    if 'bounds' in ds.ncattrs():
        # Retrieve the 'bounds' attribute
        bounds_attr = ds.getncattr('bounds')
        # Print the 'bounds' attribute
        print("bounds:", bounds_attr)
    else:
        print("The attribute 'bounds' does not exist in this dataset.")

    # Calculate the total uncompressed size
    total_uncompressed_size = sum(var.size * var.dtype.itemsize for var in ds.variables.values())

    # Get the actual file size on disk
    actual_file_size = os.path.getsize(file_path)

    # Calculate and print the compression ratio
    if total_uncompressed_size > 0:
        compression_ratio = total_uncompressed_size / actual_file_size
        print(f"\nOverall Compression Ratio: {compression_ratio:.2f}:1 (Theoretical uncompressed size / Actual file size)")
    else:
        print("\nOverall Compression Ratio: Computation not possible due to zero uncompressed size")

    # Close the dataset to free resources
    ds.close()

# Example usage
file_path = 'E:\\PROCESSING\\JAVALBS\\03_PHASE3\\B3-2023N_DATACUBE.nc'
describe_netcdf_file(file_path)