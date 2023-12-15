# -*- coding: utf-8 -*-
"""
This script provides analysis and preprocessing of Sentinel-2 HLS imagery data.
"""
"""
@Time    : 07/12/2023 07:58
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : HLS-Data-Preprocessing-Analysis
"""

import os
# import geopandas as gpd
import warnings
import rasterio
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from src.preprocessing_pipeline import HLSstacks
# from utils import [additional relevant utility functions or classes if any]
if __name__ == '__main__':

    # Set variables
    sentinel2_path = r'E:\Data\Sentinel2_data\30pc_cc\Borneo_June2020_Jan2023_30pc_cc'
    stack_path_list = r'E:\Data\Sentinel2_data\30pc_cc\Borneo_June2020_Jan2023_30pc_cc_stacks'
    bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']  # Sentinel-2 bands to include
    # shp = gpd.read_file('[Path to optional shapefile]')  # Optional geospatial boundary
    radd_alert_path = r"E:\Data\Radd_Alerts_Borneo"
    land_cover_path = r"E:\Data\Sentinel2_data\30pc_cc\Land_Cover_Borneo_Cropped_30pc_cc"
    cropped_radd_alert_path = r"E:\Data\Sentinel2_data\30pc_cc\Radd_Alerts_Borneo_Cropped_30pc_cc"



    # Initialize HLSstacks object
    hls_data = HLSstacks(sentinel2_path, stack_path_list, bands, radd_alert_path, land_cover_path)#, shp)

    # stitch radd alerts together
    #merged_radd_path = hls_data.stitch_radd_alerts()

    # resample radd alerts to 30m
    #hls_data.resample_radd_alerts()

    ## Iterate through Sentinel-2 files in the sentinel2_path directory
    # for sentinel2_file in os.listdir(sentinel2_path):
    #     if sentinel2_file.endswith('.tif'):
    #         sentinel2_file_path = os.path.join(sentinel2_path, sentinel2_file)
    #
    #         # Call crop_images_to_stacks for each Sentinel-2 file
    #         hls_data.crop_images_to_stacks(
    #             stack_path_list, sentinel2_file_path, cropped_radd_alert_path
    #         )



    # combine radd alert images to 1 per sentienl-2 image
    # for sentinel2_file in sentinel2_path:
    #     hls_data.combine_radd_alerts(sentinel2_file)

    # Process and stack Sentinel-2 imagery
    #hls_data.write_hls_rasterio_stack()


    ###########
    ## Crop radd alerts
    ###########
    for sentinel2_file in os.listdir(stack_path_list):
        if sentinel2_file.endswith('.tif'):
            sentinel2_file_path = os.path.join(stack_path_list, sentinel2_file)

            # Call crop_images_to_stacks for each Sentinel-2 file
            hls_data.crop_single_stack(sentinel2_file_path,
                                       r"E:\Data\Radd_Alerts_Borneo\resampled_radd_alerts_int16_compressed.tif", cropped_radd_alert_path)
    #hls_data.crop_images_to_stacks(r"E:\Data\Sentinel2_data\30pc_cc\Borneo_June2020_Jan2023_30pc_cc_stacks", r"E:\Data\Radd_Alerts_Borneo\resampled_radd_alerts_int16_compressed.tif", cropped_radd_alert_path)

    ###########
    ## Crop land cover
    ###########
    #hls_data.crop_images_to_stacks("E:\Data\Sentinel2_data\Borneo_June2020_Jan2023_30pc_cc_stacks", "E:\Data\CMS_AGB_Landcover_Indonesia_1645\CMS_AGB_Landcover_Indonesia_1645\data\land_cover", "E:\Data\Sentinel2_data\Land_Cover_Borneo_Cropped_30pc_cc_stacks")

