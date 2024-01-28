# -*- coding: utf-8 -*-
"""
"""
"""
@Time    : 6/12/2024 00:01
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : dataset_run_management
"""


import os
from dataset_management import DatasetManagement
from model_input_processor import Loader
source_dir = r"E:\Data\Sentinel2_data\30pc_cc\Borneo_June2020_Jan2023_30pc_cc_stacks_radd_fmask_corrected"
train_dir = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\training"
val_dir = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\validation"
output_dir = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc"  # Directory to store cropped tiles
tile_size = 512

# Instantiate the DatasetManagement class with an additional output_dir parameter
data_manager = DatasetManagement(source_dir, train_dir, val_dir, output_dir, tile_size)
model_funcs = Loader(output_dir,  train_dir, val_dir, output_dir,tile_size)
if __name__ == '__main__':

    ## remove file with less than 1000 pixels in each file.

    #######
    ## Filter images based on minimal labels requirement
    #######
    ### choose stack to filter direction by
    ###

    # alert_stacks = model_funcs.filter_stacks(source_dir, 1,1000)
    # distribution = model_funcs.calculate_alert_distribution(alert_stacks,5)
    # print(distribution)
    #
    # ## Step 1: Crop original stacks into tiles
    # for file in os.listdir(source_dir):
    #     if file.endswith('_radd_Fmaskapplied.tif'):
    #         image_path = os.path.join(source_dir, file)
    #         data_manager.crop_to_tiles(image_path)
    #         print(f"cropping finished: {file}")
   # alert_stacks = model_funcs.filter_stacks(output_dir, 1,1000)
   #  distribution = model_funcs.calculate_alert_distribution(alert_stacks,5)
   #  print(distribution)

    ## step 4, alter labels
    # model_funcs.alter_radd_data_to_label(output_dir)


    ## before running, tiles should be removed that lack required alert number.
    # run step 2 & 3 together
    ##Step 2: Split cropped tiles into Sentinel-2 and RADD alert pairs
    data_manager.split_tiles(output_dir)

    ## Step 3: Split the dataset into training and validation sets and move the files
    data_manager.split_dataset(output_dir)

    ## Plot all bands
    # output_dir_image = "T50NNK_2021210T022549_radd_Fmaskapplied_1024_0_raddaltered.tif"
    # data_manager.plot_7_bands(output_dir_image)







