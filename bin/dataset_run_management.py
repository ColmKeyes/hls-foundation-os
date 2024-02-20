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
from src.dataset_management import DatasetManagement
from src.model_input_processor import Loader
source_dir = r"E:\Data\Sentinel2_data\30pc_cc\Borneo_June2021_Dec_2023_30pc_cc_stacks_agb_radd_forest_fmask"
train_dir = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\training"
train_dir_narrow = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\training_narrow"
val_dir = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\validation"
val_dir_narrow = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\validation_narrow"
output_dir = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc"  # Directory to store cropped tiles
tile_size = 512

# Instantiate the DatasetManagement class with an additional output_dir parameter
data_manager = DatasetManagement(source_dir, train_dir, val_dir, output_dir, tile_size)
model_funcs = Loader(output_dir,  train_dir, val_dir, output_dir,tile_size)
if __name__ == '__main__':


    #############
    ## Step 9: Filter images based on minimal labels requirements. typical limits: 1000-5000.
    #############
    # alert_stacks = model_funcs.filter_stacks(source_dir, 1,5000)#1000) # set to 0 for current distribution
    # distribution = model_funcs.calculate_alert_distribution(alert_stacks,5)
    # print(distribution)

    #############
    ## Step 10: Crop sen2_agb_radd_forest_stacks into 512x512 tiles for model input
    #############
    # for file in os.listdir(source_dir):
    #     if file.endswith('_radd_fmask_stack.tif'):
    #         image_path = os.path.join(source_dir, file)
    #         data_manager.crop_to_tiles(image_path,output_dir)
    #         print(f"cropping finished: {file}")

    #############
    ## Step 11: Re-Filter tiled images based on minimal labels requirements. typical limits: 1000-5000.
    #############

    alert_stacks = model_funcs.filter_stacks(val_dir_narrow,1,7500)#output_dir, 1,0)#1000)
    distribution = model_funcs.calculate_alert_distribution(alert_stacks,20)
    print(distribution)

    #############
    ## step 12, alter labels
    #############

    # model_funcs.alter_radd_data_to_label(output_dir)


    ## before running, tiles should be removed that lack required alert number.
    # run step 2 & 3 together
    ##Step 2: Split cropped tiles into Sentinel-2 and RADD alert pairs
    # data_manager.split_tiles(output_dir)

    ## Step 3: Split the dataset into training and validation sets and move the files
    # data_manager.split_dataset(output_dir)



    # for file in os.listdir(train_dir):

        ## Plot all bands
    ## nmno need for full path.
    #output_dir_image = os.path.join(output_dir,"2021210_T50NNK_agb_radd_fmask_stack_1024_0_radd_modified_sentinel.tif")
    # data_manager.plot_7_bands("2021210_T50NNK_agb_radd_fmask_stack_1024_0_radd_modified_sentinel.tif")







