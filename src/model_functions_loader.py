# -*- coding: utf-8 -*-
"""
This script contains functions for processing and preparing data for the Prithvi-100m model, focusing on tasks such as cropping large satellite images into smaller tiles.
"""
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


class prep:
    def __init__(self,  output_folder=None, tile_size=224):
        self.tile_size = tile_size
        self.output_folder = output_folder

    def crop_to_tiles(self, image_path):
        with rasterio.open(image_path) as src:
            for j in range(0, src.height, self.tile_size):
                for i in range(0, src.width, self.tile_size):
                    window = Window(i, j, self.tile_size, self.tile_size)
                    if src.height - j < self.tile_size or src.width - i < self.tile_size:
                        continue  # Skip incomplete tiles at the edge

                    tile = src.read(window=window)

                    # Save each tile
                    tile_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{i}_{j}.tif"
                    tile_path = os.path.join(self.output_folder, tile_filename)

                    with rasterio.open(
                            tile_path,
                            'w',
                            driver='GTiff',
                            height=tile.shape[1],
                            width=tile.shape[2],
                            count=src.count,
                            dtype=tile.dtype,
                            crs=src.crs,
                            transform=rasterio.windows.transform(window, src.transform)
                    ) as tile_dst:
                        tile_dst.write(tile)














