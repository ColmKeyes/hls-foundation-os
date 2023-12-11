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
    sentinel2_path = r'E:\Data\Sentinel2_data\Borneo_June2020_Jan2023_30pc_cc'
    stack_path_list = r'E:\Data\Sentinel2_data\Borneo_June2020_Jan2023_30pc_cc_stacks'
    bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']  # Sentinel-2 bands to include
    # shp = gpd.read_file('[Path to optional shapefile]')  # Optional geospatial boundary
    radd_alert_path = r"E:\Data\Sentinel2_data\Radd_Alerts_Borneo_Cropped_30pc_cc_stacks"
    land_cover_path = r"E:\Data\Sentinel2_data\Land_Cover_Borneo_Cropped_30pc_cc_stacks"




    # Initialize HLSstacks object
    hls_data = HLSstacks(sentinel2_path, stack_path_list, bands, radd_alert_path, land_cover_path)#, shp)

    # combine radd alert images to 1 per sentienl-2 image
    for sentinel2_file in sentinel2_path:
        hls_data.combine_radd_alerts(sentinel2_file)

    # Process and stack Sentinel-2 imagery
    #hls_data.write_rasterio_stack()


    ###########
    ## Crop radd alerts
    ###########
    #hls_data.crop_images_to_stacks("E:\Data\Sentinel2_data\Borneo_June2020_Jan2023_30pc_cc_stacks", "E:\Data\Radd_Alerts_Borneo", "E:\Data\Sentinel2_data\Radd_Alerts_Borneo_Cropped_30pc_cc_stacks")

    ###########
    ## Crop land cover
    ###########
    #hls_data.crop_images_to_stacks("E:\Data\Sentinel2_data\Borneo_June2020_Jan2023_30pc_cc_stacks", "E:\Data\CMS_AGB_Landcover_Indonesia_1645\CMS_AGB_Landcover_Indonesia_1645\data\land_cover", "E:\Data\Sentinel2_data\Land_Cover_Borneo_Cropped_30pc_cc_stacks")

