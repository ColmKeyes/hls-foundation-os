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


    def crop_images_to_stacks(self, sentinel_stacks_folder, images_path, output_folder):
        """
        Crops raster images (either a single image or multiple images in a folder) to match the extents of Sentinel-2 image stacks.
        """
        # Check if images_path is a directory or a single file
        is_directory = os.path.isdir(images_path)

        # Define the projection transformer
        transformer = partial(
            pyproj.transform,
            pyproj.Proj(init='epsg:32649'),  # source coordinate system (Sentinel-2 stacks)
            pyproj.Proj(init='epsg:4326')  # destination coordinate system (Images)
        )

        for sentinel_file in os.listdir(sentinel_stacks_folder):
            if sentinel_file.endswith('_stack.tif'):
                sentinel_path = os.path.join(sentinel_stacks_folder, sentinel_file)
                with rasterio.open(sentinel_path) as sentinel_stack:
                    sentinel_bounds = sentinel_stack.bounds
                    sentinel_crs = sentinel_stack.crs
                    sentinel_box = box(*sentinel_bounds)

                    # Reproject Sentinel-2 bounding box to EPSG:4326
                    sentinel_box_4326 = shapely_transform(transformer, sentinel_box)

                    # Iterate over the images (if directory) or just use the single image
                    images_to_process = os.listdir(images_path) if is_directory else [images_path]
                    for image_file in images_to_process:
                        image_path = os.path.join(images_path, image_file) if is_directory else image_file
                        with rasterio.open(image_path) as image_raster:
                            image_box = box(*image_raster.bounds)

                            # Check if the bounding boxes intersect (overlap)
                            if not sentinel_box_4326.intersects(image_box):
                                continue  # Skip if no overlap

                            # Calculate the transformation and new dimensions for reprojection
                            transform, width, height = calculate_default_transform(
                                image_raster.crs, sentinel_crs, image_raster.width, image_raster.height, *image_raster.bounds
                            )

                            # Reproject image raster in memory
                            with MemoryFile() as memfile:
                                with memfile.open(driver='GTiff', height=height, width=width, count=image_raster.count, dtype=image_raster.dtypes[0], crs=sentinel_crs,
                                                  transform=transform) as image_reprojected:
                                    for i in range(1, image_raster.count + 1):
                                        reproject(
                                            source=rasterio.band(image_raster, i),
                                            destination=rasterio.band(image_reprojected, i),
                                            src_transform=image_raster.transform,
                                            src_crs=image_raster.crs,
                                            dst_transform=transform,
                                            dst_crs=sentinel_crs,
                                            resampling=Resampling.nearest
                                        )

                                    # Get the window for cropping
                                    window = from_bounds(*sentinel_bounds, transform=transform, width=width, height=height)
                                    image_cropped = image_reprojected.read(window=window)

                                    # Copy the profile to use it outside the 'with' block
                                    output_profile = image_reprojected.profile.copy()
                                    output_profile.update({
                                        'height': window.height,
                                        'width': window.width,
                                        'transform': image_reprojected.window_transform(window)
                                    })

                                output_file_name = f"{sentinel_file.split('.')[0]}_{os.path.basename(image_file)}"
                                output_file_path = os.path.join(output_folder, output_file_name)

                                # Save the cropped image raster
                                with rasterio.open(output_file_path, "w", **output_profile) as dest:
                                    dest.write(image_cropped)




    def combine_radd_alerts(self, sentinel_base_name):
        """
        Combines multiple RADD alert images for a given Sentinel-2 tile into a single image.

        Args:
            sentinel_base_name (str): The base name of the Sentinel-2 tile.

        Returns:
            str: File path of the combined RADD alert image.
        """

        radd_files = [f for f in os.listdir(self.radd_alert_path) if sentinel_base_name in f]
        if not radd_files:
            return None  # Return None if no corresponding RADD alert files found

        # Initialize an array to hold combined data
        combined_radd = None

        for radd_file in radd_files:
            radd_path = os.path.join(self.radd_alert_path, radd_file)
            with rasterio.open(radd_path) as src:
                radd_data = src.read(1)  # Assuming RADD alert data is in the first band

                if combined_radd is None:
                    # Initialize combined_radd with the shape and type of the first RADD file
                    combined_radd = np.zeros_like(radd_data)

                # Combine by taking the maximum value (useful for binary/categorical data)
                combined_radd = np.maximum(combined_radd, radd_data)

        # Save the combined RADD alert image
        combined_radd_path = os.path.join(self.radd_alert_path, f'{sentinel_base_name}_combined_radd.tif')
        with rasterio.open(radd_files[0]) as src:
            profile = src.profile
            with rasterio.open(combined_radd_path, 'w', **profile) as dst:
                dst.write(combined_radd, 1)

        return combined_radd_path



        def write_rasterio_stack(self):
            """
            Write folder of Sentinel-2 GeoTIFFs, corresponding Fmask, RADD alert labels,
            and land classification labels to a GeoTIFF stack file.
            """

            for file in os.listdir(self.sentinel2_path):
                if any(band in file for band in self.bands) and file.endswith('.tif'):
                    sentinel_file = os.path.join(self.sentinel2_path, file)

                    # Corresponding Fmask file
                    fmask_file_name = '.'.join(file.split('.')[:-2]) + '.Fmask.tif'
                    fmask_file = os.path.join(self.sentinel2_path, fmask_file_name)

                    # RADD alert and Land cover file names
                    base_name = file.split('.')[0]
                    radd_alert_file_name = base_name + '_Radd_Alert_Borneo.tif'
                    land_cover_file_name = base_name + '_Kalimantan_land_cover.tif'
                    radd_alert_file = os.path.join(self.radd_alert_path, radd_alert_file_name)
                    land_cover_file = os.path.join(self.land_cover_path, land_cover_file_name)

                    if not os.path.exists(fmask_file):
                        continue  # Skip if corresponding Fmask file does not exist

                    # Files to be stacked
                    files = [sentinel_file, fmask_file, radd_alert_file, land_cover_file]

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




        # """
        # Write selected bands from Sentinel-2 imagery for each date to separate GeoTIFF stack files.
        # """
        # # Group files by Julian date
        # date_grouped_files = defaultdict(list)
        # for file in os.listdir(self.path):
        #     if file.endswith('.tif'):
        #         # Extract Julian date from filename
        #         julian_date = file.split('.')[3][:7]  # Adjust the slicing if necessary
        #         date_grouped_files[julian_date].append(file)
        #
        # for julian_date, files in date_grouped_files.items():
        #     # Filter files for the specified bands
        #     selected_files = [f for f in files if any(band in f for band in self.bands)]
        #
        #     if not selected_files:
        #         continue  # Skip if no files match the criteria
        #
        #     # Read the first file to setup profile
        #     src_image = rasterio.open(os.path.join(self.path, selected_files[0]), "r")
        #
        #     dst_profile = src_image.profile.copy()
        #     dst_profile.update({
        #         "driver": "GTiff",
        #         "count": len(selected_files),
        #         "height": src_image.height,
        #         "width": src_image.width,
        #         "transform": src_image.transform,
        #         "crs": src_image.crs
        #     })
        #
        #     output_file_path = os.path.join(self.stack_path_list, f'{julian_date}_stack.tif')
        #     if not os.path.exists(os.path.dirname(output_file_path)):
        #         os.makedirs(os.path.dirname(output_file_path))
        #
        #     with rasterio.open(output_file_path, 'w', **dst_profile) as dst:
        #         for i, file in enumerate(selected_files, start=1):
        #             with rasterio.open(os.path.join(self.path, file)) as src:
        #                 data = src.read(1)  # Read the first band
        #                 if write:
        #                     dst.write(data, i)









