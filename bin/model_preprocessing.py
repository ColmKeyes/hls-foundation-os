"""
@Time    : [Time of Creation, e.g., 07/12/2023 10:00]
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : model_functions
"""

import rasterio
from rasterio.windows import Window
import numpy as np
import os
from src.model_functions import ModelFunctions

output_folder =  r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc"
combined_radd_sen2_stack_path = r"E:\Data\Sentinel2_data\30pc_cc\Borneo_June2020_Jan2023_30pc_cc_stacks_radd"
stack_path_list = r'E:\Data\Sentinel2_data\30pc_cc\Borneo_June2020_Jan2023_30pc_cc_stacks_radd'


tile_size =  512


if __name__ == '__main__':

    model_funcs = ModelFunctions(output_folder, tile_size )


    for sentinel2_file in os.listdir(combined_radd_sen2_stack_path):
        if sentinel2_file.endswith('.tif'):
            sentinel2_file_path = os.path.join(stack_path_list, sentinel2_file)
            model_funcs.crop_to_tiles(sentinel2_file_path)



