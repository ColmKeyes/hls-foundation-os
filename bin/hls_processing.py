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
    land_cover_path = r"E:\Data\CMS_AGB_Landcover_Indonesia_1645\CMS_AGB_Landcover_Indonesia_1645\data\land_cover"
    radd_alert_path = r"E:\Data\Radd_Alerts_Borneo"
    cropped_land_cover_path = r"E:\Data\Sentinel2_data\30pc_cc\Land_Cover_Borneo_Cropped_30pc_cc"
    cropped_radd_alert_path = r"E:\Data\Sentinel2_data\30pc_cc\Radd_Alerts_Borneo_Cropped_30pc_cc"
    merged_radd_alerts = f"{radd_alert_path}\\merged_radd_alerts_qgis_int16_compressed.tif"
    combined_radd_sen2_stack_path = r"E:\Data\Sentinel2_data\30pc_cc\Borneo_June2020_Jan2023_30pc_cc_stacks_radd"

    # Initialize HLSstacks object
    hls_data = HLSstacks(sentinel2_path, stack_path_list, bands, radd_alert_path, land_cover_path)#, shp)

    ##################
    ## compress images
    ##################
    ## gdal_translate -a_nodata 0 -co "COMPRESS=LZW" resampled_radd_alerts.tif resampled_radd_alerts_int16_compressed.tif
    ##################

    # resample radd alerts to 30m
    #hls_data.resample_radd_alerts(merged_radd_alerts)

    # Process and stack Sentinel-2 imagery
    #hls_data.write_hls_rasterio_stack()

    ##########
    # Crop radd alerts
    ##########
    # for sentinel2_file in os.listdir(stack_path_list):
    #     if sentinel2_file.endswith('.tif'):
    #         sentinel2_file_path = os.path.join(stack_path_list, sentinel2_file)
    #
    #         # Call crop_images_to_stacks for each Sentinel-2 file
    #         hls_data.crop_single_stack(sentinel2_file_path,
    #                                    r"E:\Data\Radd_Alerts_Borneo\resampled_merged_radd_alerts_qgis_int16_compressed_30m.tif", cropped_radd_alert_path)

    ###########
    ## Crop AGB Land Cover Types
    ###########
    ## Have the data already processed, but doesnt quite azs is for land_cover
    # for sentinel2_file in os.listdir(stack_path_list):
    #     if sentinel2_file.endswith('.tif'):
    #         sentinel2_file_path = os.path.join(stack_path_list, sentinel2_file)
    #
    #         # Call crop_images_to_stacks for each Sentinel-2 file
    #         hls_data.crop_single_stack(sentinel2_file_path,
    #                                    os.path.join(land_cover_path, "Kalimantan_land_cover.tif"), cropped_land_cover_path)
    #


    ###########
    ## Crop land cover
    ###########
    #hls_data.crop_images_to_stacks("E:\Data\Sentinel2_data\Borneo_June2020_Jan2023_30pc_cc_stacks", "E:\Data\CMS_AGB_Landcover_Indonesia_1645\CMS_AGB_Landcover_Indonesia_1645\data\land_cover", "E:\Data\Sentinel2_data\Land_Cover_Borneo_Cropped_30pc_cc_stacks")


    ##########
    ## Stacking images
    ##########
    #hls_data.stack_images(stack_path_list, cropped_radd_alert_path, combined_radd_sen2_stack_path,"radd")

    ##########
    ## GDALwarp Stacking
    ##########
    for sentinel2_file in os.listdir(stack_path_list):
        if sentinel2_file.endswith('.tif'):
            sentinel2_file_path = os.path.join(stack_path_list, sentinel2_file)
            date = sentinel2_file.split('_')[0]
            tile = sentinel2_file.split('_')[1]

            # Corresponding RADD alerts file path
            radd_file = f"{date}_{tile}_resampled_merged_radd_alerts_qgis_int16_compressed_30m.tif"
            radd_file_path = os.path.join(cropped_radd_alert_path, radd_file)

            # Output file path for the merged stack
            output_file = os.path.join(combined_radd_sen2_stack_path, f"{tile}_{date}_stacked_radd.tif")

            # Check if the output file already exists, skip if it does
            if os.path.exists(output_file):
                print(f"Output file {output_file} already exists. Skipping...")
                continue

            if os.path.exists(radd_file_path):
                warped_band_files = []

                # Warp each band in Sentinel-2 stack
                for i in range(1, 7):  # Adjust based on the number of bands in your Sentinel-2 stack
                    warped_band_file = f"{output_file.replace('.tif', '')}_warped_band_{i}.tif"
                    hls_data.warp_band(sentinel2_file_path, i, warped_band_file)
                    warped_band_files.append(warped_band_file)

                # Warp the RADD alerts band
                warped_radd_file = f"{output_file.replace('.tif', '')}_warped_radd_band.tif"
                hls_data.warp_band(radd_file_path, 1, warped_radd_file)  # Assuming RADD file has only one band
                warped_band_files.append(warped_radd_file)

                # Merge the warped bands
                hls_data.merge_bands(warped_band_files, output_file)

                # Clean up individual band files
                for file in warped_band_files:
                    os.remove(file)

                print(f"Warped and merged raster saved to {output_file}")