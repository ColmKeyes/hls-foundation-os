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
from rasterio.merge import merge
from shapely.geometry import box,mapping
from shapely.ops import transform as shapely_transform
import pyproj
from pyproj import Transformer
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
from osgeo import gdal
from osgeo_utils import gdal_merge as gm



warnings.simplefilter(action='ignore', category=FutureWarning)

#plt.style.use('dark_background')


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

class prep:
    """
    A class for processing Sentinel-2 imagery data in preperation for assimilation by the Prithvi-100m model.
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

    """"""""
    # Core Functionalities
    """"""""

    def resample_radd_alerts(self, merged_radd_alerts):
        """
        Resamples 'merged_radd_alerts.tif' to match the resolution of a Sentinel-2 image.
        """

        # Use the first Sentinel-2 image to determine the target CRS
        sentinel_files = [f for f in os.listdir(self.sentinel2_path) if f.endswith('.tif')]
        if not sentinel_files:
            raise ValueError("No Sentinel-2 images found in the specified path.")

        sentinel_path = os.path.join(self.sentinel2_path, sentinel_files[0])
        with rasterio.open(sentinel_path) as sentinel_dataset:
            sentinel_crs = sentinel_dataset.crs

        # Open the merged RADD alerts image
        with rasterio.open(merged_radd_alerts) as merged_radd_dataset:
            # Manually set the desired output resolution (30m)
            desired_resolution = (30.0, 30.0)

            # Calculate the new transform
            transform, width, height = calculate_default_transform(
                merged_radd_dataset.crs, sentinel_crs,
                merged_radd_dataset.width, merged_radd_dataset.height,
                *merged_radd_dataset.bounds,
                resolution=desired_resolution
            )

            # Define metadata for the resampled dataset
            out_meta = merged_radd_dataset.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": height,
                "width": width,
                "transform": transform,
                "crs": sentinel_crs,
                "compress": "LZW",
                "dtype": 'int16'
            })

            # Perform the resampling
            resampled_radd_path = os.path.join(self.radd_alert_path, 'resampled_merged_radd_alerts_qgis_int16_compressed_30m.tif')
            with rasterio.open(resampled_radd_path, 'w', **out_meta) as dest:
                for i in range(1, merged_radd_dataset.count + 1):
                    reproject(
                        source=rasterio.band(merged_radd_dataset, i),
                        destination=rasterio.band(dest, i),
                        src_transform=merged_radd_dataset.transform,
                        src_crs=merged_radd_dataset.crs,
                        dst_transform=transform,
                        dst_crs=sentinel_crs,
                        resampling=Resampling.nearest
                    )

        return resampled_radd_path

    def crop_single_stack(self, sentinel_stack_path, single_image_path, output_path):

        ##############################
        ## AGB LAND CLASSIFICATION VERSION
        ##############################
        with rasterio.open(sentinel_stack_path) as sentinel_stack, rasterio.open(single_image_path) as image_raster:
            sentinel_bounds = sentinel_stack.bounds
            sentinel_crs = sentinel_stack.crs
            image_bounds = image_raster.bounds
            image_crs = image_raster.crs

            # Create transformers to WGS 84 (EPSG:4326)
            transformer_to_wgs84_sentinel = Transformer.from_crs(sentinel_crs, 'EPSG:4326', always_xy=True)
            transformer_to_wgs84_image = Transformer.from_crs(image_crs, 'EPSG:4326', always_xy=True)

            # Define the functions for coordinate transformation
            def transform_to_wgs84_sentinel(x, y):
                return transformer_to_wgs84_sentinel.transform(x, y)

            def transform_to_wgs84_image(x, y):
                return transformer_to_wgs84_image.transform(x, y)

            # Transform sentinel bounds and image bounds to WGS 84
            sentinel_box_wgs84 = shapely_transform(transform_to_wgs84_sentinel, box(*sentinel_bounds))
            image_box_wgs84 = shapely_transform(transform_to_wgs84_image, box(image_bounds.left, image_bounds.bottom, image_bounds.right, image_bounds.top))

            if not sentinel_box_wgs84.intersects(image_box_wgs84):
                print("doesn't intersect in WGS 84")
                return  # Optionally skip further processing

            transformer_to_image_crs = Transformer.from_crs('EPSG:4326', image_crs, always_xy=True)

            def transform_to_image_crs(x, y):
                return transformer_to_image_crs.transform(x, y)

            # Transform sentinel_box_wgs84 back to the image raster's CRS
            sentinel_box_image_crs = shapely_transform(transform_to_image_crs, sentinel_box_wgs84)

            # Now sentinel_box_image_crs is in the same CRS as the image raster
            # Proceed with masking using sentinel_box_image_crs
            image_cropped, transform = mask(image_raster, [sentinel_box_image_crs], crop=True, filled=False, pad=False, nodata=0)

            # Mask or clip the image raster to the area of the Sentinel-2 stack, specifying the nodata value and transform
            #image_cropped, transform = mask(image_raster, [sentinel_box_wgs84], crop=True, filled=False, pad=False, nodata=0)

            image_cropped[0][image_cropped[0] != 3] = 0

            # Update the profile for the cropped image
            output_profile = image_raster.profile.copy()
            output_profile.update({
                'height': image_cropped.shape[1],
                'width': image_cropped.shape[2],
                'transform': transform,
                'count' : 1
            })

            # Extract the relevant parts of the file name from the sentinel_stack_path
            parts = os.path.basename(sentinel_stack_path).split('.')[0].split('_')
            identifier = f"{parts[0]}_{parts[1]}"

            # Assuming the base name of the single_image_path is 'resampled_radd_alerts_int16_compressed.tif'
            suffix = os.path.basename(single_image_path)

            # Combine the identifier and suffix to form the output file name
            output_file_name = f"{identifier}_{suffix}"
            output_file_path = os.path.join(output_path, output_file_name)

            if os.path.exists(output_file_path):
                print(f"File {output_file_path} already exists. Skipping cropping.")
                return  # Skip the rest of the function


            with rasterio.open(output_file_path, 'w', **output_profile) as dest:
                dest.write(image_cropped[1],1)

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

    def merge_with_agb(self, sentinel_stack_path, agb_path, output_path):
        with rasterio.open(sentinel_stack_path) as sentinel_stack, rasterio.open(agb_path) as agb:
            # Check and reproject AGB data to match Sentinel-2 stack CRS and resolution
            if agb.crs != sentinel_stack.crs or agb.res != sentinel_stack.res:
                # Reproject and resample AGB data
                # (Add reprojection and resampling code here)
                pass

            # Read and stack data
            sentinel_data = sentinel_stack.read()
            agb_data = agb.read(1)

            # Stack AGB data as an additional band to Sentinel-2 data
            stacked_data = np.concatenate((sentinel_data, agb_data[None, :, :]), axis=0)

            # Update the profile for the output file
            output_profile = sentinel_stack.profile.copy()
            output_profile.update(count=sentinel_data.shape[0] + 1)

            # Write the stacked data to a new file
            output_file = os.path.join(output_path, os.path.basename(sentinel_stack_path).replace('.tif', '_with_AGB.tif'))
            with rasterio.open(output_file, 'w', **output_profile) as dest:
                dest.write(stacked_data)

    """"""""
    # Utility Functionalities
    """"""""
    def clip_to_extent(self, src_file, target_file, output_file):
        """
        Clips src_file to the extent of target_file.

        :param src_file: File path of the source image to be clipped.
        :param target_file: File path of the target image for the clipping extent.
        :param output_file: File path for the output clipped image.
        """
        # Open the target image and get its bounding box
        with rasterio.open(target_file) as target_src:
            target_bounds = target_src.bounds

        # Create a bounding box geometry
        bbox = box(*target_bounds)
        geo = [bbox.__geo_interface__]

        # Open the source image and clip it using the bounding box
        with rasterio.open(src_file) as src:
            out_image, out_transform = mask(dataset=src, shapes=geo, crop=True)
            out_meta = src.meta.copy()

            # Update the metadata to match the clipped data
            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})

            # Write the clipped image
            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(out_image)

    def reorder_and_add_blank_band(self, input_file, output_file):
        with rasterio.open(input_file) as src:
            #  metadata for new raster with additional band
            meta = src.meta.copy()
            meta.update(count=7)

            # Create the new raster file
            with rasterio.open(output_file, 'w', **meta) as dst:
                blank_band = src.read(1) * 0
                dst.write(blank_band, 1)

                for band in range(1, src.count + 1):
                    data = src.read(band)
                    dst.write(data, band + 1)

    def apply_fmask(self, sentinel_stack_path, fmask_path, output_file):
        CLOUD_BIT = 1 << 1     # Bit 1 for clouds
        CLOUD_SHADOW_BIT = 1 << 3  # Bit 3 for cloud shadow

        with rasterio.open(sentinel_stack_path) as sentinel_stack, rasterio.open(fmask_path) as fmask:
            fmask_data = fmask.read(1)

            # Cloud and cloud shadow masks
            cloud_mask = (fmask_data & CLOUD_BIT) != 0
            cloud_shadow_mask = (fmask_data & CLOUD_SHADOW_BIT) != 0
            combined_mask = cloud_mask | cloud_shadow_mask

            masked_data = np.empty_like(sentinel_stack.read(), dtype=rasterio.float32)
            for band in range(sentinel_stack.count):
                band_data = sentinel_stack.read(band + 1)
                masked_band = np.where(combined_mask, -9999, band_data)
                masked_data[band] = masked_band

            output_profile = sentinel_stack.profile.copy()
            output_profile.update(dtype=rasterio.float32, nodata=-9999)
            with rasterio.open(output_file, 'w', **output_profile) as dest:
                dest.write(masked_data)


    """"""""
    # Data Wrangling Functionalities
    """"""""

    def warp_rasters(self, input_files, output_file, src_nodata=None, dst_nodata=None):
        # Warp options
        warp_options = gdal.WarpOptions(format='GTiff',
                                        srcNodata=src_nodata,
                                        dstNodata=dst_nodata,
                                        multithread=True)

        # Perform the warp
        gdal.Warp(destNameOrDestDS=output_file,
                  srcDSOrSrcDSTab=input_files,
                  options=warp_options)










