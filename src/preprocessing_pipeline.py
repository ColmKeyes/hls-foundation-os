# -*- coding: utf-8 -*-
"""
This script provides a class and methods for preprocessing Sentinel-2 imagery data.
The main functionalities include band selection and stacking for further analysis and model input.
"""
"""
@Time    : 7/12/2023 07:49
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : preprocessing_pipeline
"""

##################
## compress images
##################
## gdal_translate -a_nodata 0 -co "COMPRESS=LZW" resampled_radd_alerts.tif resampled_radd_alerts_int16_compressed.tif
##
import numpy as np
import numpy.ma as ma
import os
from matplotlib.pyplot import pause
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import pandas as pd
# import rioxarray
# import xarray
import rasterio
import rasterio.plot
from rasterio.warp import calculate_default_transform ,reproject
from rasterio.windows import get_data_window, transform
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.windows import from_bounds
from rasterio.io import MemoryFile
from rasterio.merge import merge
from shapely.geometry import box,mapping
from shapely.ops import transform as shapely_transform
import pyproj
from functools import partial
# from geocube.api.core import make_geocube
from datetime import datetime
import matplotlib.pyplot as plt
# from meteostat import Daily
# from statsmodels.tsa.seasonal import seasonal_decompose
import re
import warnings
from collections import defaultdict
import json
from functools import partial
import os
import pyproj
import rasterio
from rasterio import MemoryFile
from rasterio.transform import from_bounds, Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box
from shapely.ops import transform as shapely_transform


warnings.simplefilter(action='ignore', category=FutureWarning)

#plt.style.use('dark_background')


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class HLSstacks:
    """
    A class for processing Sentinel-2 imagery data to generate band stacks for further analysis.
    """

    def __init__(self, sentinel2_path, stack_path_list, bands, radd_alert_path, land_cover_path, shp=None):
        """
        Args:
            path (list): List of file paths to Sentinel-2 imagery.
            stack_path_list (str): Path to directory where output raster stacks will be stored.
            bands (list): List of Sentinel-2 band names to include in the stack (e.g., ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']).
            shp (geopandas.GeoDataFrame, optional): GeoDataFrame containing study area polygon.

        Attributes:
            cube (None or xarray.DataArray): Multi-dimensional, named array for storing data.
        """

        # Initialize class variables
        self.sentinel2_path = sentinel2_path
        self.stack_path_list = stack_path_list
        self.bands = bands
        self.shp = shp
        self.cube = None

        self.radd_alert_path = radd_alert_path
        self.land_cover_path = land_cover_path

        # Initialize a list to store titles of processed files (optional)
        self.titles = []


    def resample_radd_alerts(self):
        """
        Resamples 'merged_radd_alerts.tif' to match the resolution of a Sentinel-2 image.
        """

        # Use the first Sentinel-2 image to determine the target resolution and transform
        sentinel_files = [f for f in os.listdir(self.sentinel2_path) if f.endswith('.tif')]
        if not sentinel_files:
            raise ValueError("No Sentinel-2 images found in the specified path.")

        sentinel_path = os.path.join(self.sentinel2_path, sentinel_files[0])
        with rasterio.open(sentinel_path) as sentinel_dataset:
            sentinel_transform = sentinel_dataset.transform
            sentinel_crs = sentinel_dataset.crs
            # You could also use sentinel_dataset.res to get the (x, y) resolution if needed

        # Open the merged RADD alerts image
        merged_radd_path = os.path.join(self.radd_alert_path, 'merged_radd_alerts_qgis_int16_compressed.tif')
        with rasterio.open(merged_radd_path) as merged_radd_dataset:
            # Calculate the transform and dimensions for the new resolution to match the Sentinel-2 image
            transform, width, height = calculate_default_transform(
                merged_radd_dataset.crs, sentinel_crs,
                merged_radd_dataset.width, merged_radd_dataset.height,
                *merged_radd_dataset.bounds,
                dst_transform=sentinel_transform
            )

            # Define the metadata for the resampled dataset
            out_meta = merged_radd_dataset.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": height,
                "width": width,
                "transform": transform,
                "crs": sentinel_crs,
                "count": merged_radd_dataset.count  # Keep the number of bands
            })

            # Perform the resampling
            resampled_radd_path = os.path.join(self.radd_alert_path, 'resampled_radd_alerts.tif')
            with rasterio.open(resampled_radd_path, 'w', **out_meta) as dest:
                for i in range(1, merged_radd_dataset.count + 1):
                    # Reproject and resample each band
                    reproject(
                        source=rasterio.band(merged_radd_dataset, i),
                        destination=rasterio.band(dest, i),
                        src_transform=merged_radd_dataset.transform,
                        src_crs=merged_radd_dataset.crs,
                        dst_transform=transform,
                        dst_crs=sentinel_crs,
                        resampling=Resampling.nearest
                    )

        print(f"Resampled raster saved to {resampled_radd_path}")
        return resampled_radd_path




    def write_hls_rasterio_stack(self):
        """
        Write folder of Sentinel-2 GeoTIFFs, corresponding Fmask, to a GeoTIFF stack file.
        """

        for file in os.listdir(self.sentinel2_path):
            if any(band in file for band in self.bands) and file.endswith('.tif'):
                sentinel_file = os.path.join(self.sentinel2_path, file)

                # Corresponding Fmask file
                fmask_file_name = '.'.join(file.split('.')[:-2]) + '.Fmask.tif'
                fmask_file = os.path.join(self.sentinel2_path, fmask_file_name)

                if not os.path.exists(fmask_file):
                    continue  # Skip if corresponding Fmask file does not exist

                # Files to be stacked (only Sentinel-2 bands and Fmask)
                files = [sentinel_file, fmask_file]

                # Read the first image to setup profile
                with rasterio.open(sentinel_file) as src_image:
                    dst_crs = src_image.crs
                    dst_transform, dst_width, dst_height = calculate_default_transform(
                        src_image.crs, dst_crs, src_image.width, src_image.height, *src_image.bounds)

                    # Create a profile for the stack
                    dst_profile = src_image.profile.copy()
                    dst_profile.update({
                        "driver": "GTiff",
                        "count": len(files),
                        "crs": dst_crs,
                        "transform": dst_transform,
                        "width": dst_width,
                        "height": dst_height
                    })

                    # Create stack directory if it doesn't exist
                    if not os.path.exists(self.stack_path_list):
                        os.makedirs(self.stack_path_list)

                    stack_file = os.path.join(self.stack_path_list, f'{os.path.splitext(file)[0]}_stack.tif')
                    with rasterio.open(stack_file, 'w', **dst_profile) as dst:
                        for i, file_path in enumerate(files, start=1):
                            with rasterio.open(file_path) as src:
                                data = src.read(1)  # Read the first band
                                dst.write(data, i)


    def crop_single_stack(self, sentinel_stack_path, single_image_path, output_path):
        with rasterio.open(sentinel_stack_path) as sentinel_stack:
            sentinel_bounds = sentinel_stack.bounds
            sentinel_crs = sentinel_stack.crs

            with rasterio.open(single_image_path) as image_raster:
                image_bounds = image_raster.bounds
                image_crs = image_raster.crs

                # Reproject the image raster to the CRS of the Sentinel-2 stack
                transformer = partial(
                    pyproj.transform,
                    image_crs,  # source coordinate system (Image CRS)
                    sentinel_crs  # destination coordinate system (Sentinel-2 stack CRS)
                )

                sentinel_box = box(*sentinel_bounds)
                sentinel_box_4326 = sentinel_box.__geo_interface__
                sentinel_box_image_crs = shapely_transform(transformer, sentinel_box)

                ## select only image raster where definite events occur.

                # Get the transform from the original raster dataset
                image_transform = image_raster.transform

                # Mask or clip the image raster to the area of the Sentinel-2 stack, specifying the nodata value and transform
                image_cropped, transform = mask(image_raster, [sentinel_box_image_crs], crop=True, filled=False, pad=False, nodata=0)

                image_cropped[0][image_cropped[0] != 3] = 0

                # Update the profile for the cropped image
                output_profile = image_raster.profile.copy()
                output_profile.update({
                    'height': image_cropped.shape[1],
                    'width': image_cropped.shape[2],
                    'transform': transform
                })

                # Extract the relevant parts of the file name from the sentinel_stack_path
                parts = os.path.basename(sentinel_stack_path).split('.')[0].split('_')
                identifier = f"{parts[0]}_{parts[1]}"

                # Assuming the base name of the single_image_path is 'resampled_radd_alerts_int16_compressed.tif'
                suffix = os.path.basename(single_image_path)

                # Combine the identifier and suffix to form the output file name
                output_file_name = f"{identifier}_{suffix}"
                output_file_path = os.path.join(output_path, output_file_name)


                with rasterio.open(output_file_path, 'w', **output_profile) as dest:
                    dest.write(image_cropped)









