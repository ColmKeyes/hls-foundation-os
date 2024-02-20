"""
@Time    : [Time of Creation, e.g., 07/12/2023 10:00]
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : model_run_input_processor.py
"""

import rasterio
from rasterio.windows import Window
import numpy as np
import os
from src.model_input_processor import Loader

output_folder = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc"
combined_radd_sen2_stack_path = r"E:\Data\Sentinel2_data\30pc_cc\Borneo_June2020_Jan2023_30pc_cc_stacks_radd"
stack_path_list = r'E:\Data\Sentinel2_data\30pc_cc\Borneo_June2020_Jan2023_30pc_cc_stacks_radd'
source_dir = r"E:\Data\Sentinel2_data\30pc_cc\Borneo_June2020_Jan2023_30pc_cc_stacks_radd_fmask_corrected"
train_dir = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\training"
val_dir = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\validation"
output_dir = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc"


tile_size = 512

if __name__ == '__main__':
    model_funcs = Loader(output_folder,  train_dir, val_dir, output_folder,tile_size)

    # for sentinel2_file in os.listdir(combined_radd_sen2_stack_path):
    #     if sentinel2_file.endswith('.tif'):
    #         sentinel2_file_path = os.path.join(stack_path_list, sentinel2_file)
    #         model_funcs.write_hls_rasterio_stack()

    # checkpoint_dir = r'E:\PycharmProjects\hls-foundation-os\Prithvi100m'
    # model_funcs.plot_training_progress(checkpoint_dir)


    #######
    ## Filter images based on minimal labels requirement
    #######
    ### choose stack to filter direction by
    ###
    # alert_stacks = model_funcs.filter_stacks(train_dir, 1,1000)
    # distribution = model_funcs.calculate_alert_distribution(alert_stacks,5)
    # print(distribution)





    #######
    ## plot model variables
    #######

    #model_funcs.plot_metrics_from_log(r"E:\PycharmProjects\hls-foundation-os\Prithvi100m\20240115_183444.log")

    # for folder in [train_dir,val_dir]:
    #     for file in os.listdir(folder):
    #         input_filepath = os.path.join(folder, file)
    #         model_funcs.normalize_single_file_rasterio(input_filepath)




    #######
    ## normalise and determine band mean and stddev for config file
    #######

    # for folder in [train_dir,val_dir]:
    #     model_funcs.normalize_data(folder)
    #
    # band_stats = model_funcs.calculate_band_statistics([train_dir,val_dir])
    # print(band_stats)

    # model_funcs.find_global_min_max(train_dir)
    # model_funcs.normalize_dataset(train_dir)


    #######
    ## Filter by AGB land classification
    #######




    #model_funcs.normalize_single_file_rasterio(r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\training\corrected_HLS.S30.T49MFU.2022083T023551.v2.0.Fmask_3072_1024_sentinel.tif")


    ########
    ## continuously plot bands
    ########
    # pairs = model_funcs.find_pairs(train_dir)
    #
    # for sen_file, radd_file in pairs:
    #     sen_path = os.path.join(train_dir, sen_file)
    #     radd_path = os.path.join(train_dir, radd_file)
    #     model_funcs.plot_bands(sen_path, radd_path)