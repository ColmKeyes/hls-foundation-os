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

import numpy as np
import os
import rasterio.warp
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box
from shapely.ops import transform as shapely_transform
from osgeo import gdal
import json
import re
import matplotlib.pyplot as plt
from mmseg.datasets import CustomDataset
from mmseg.datasets.builder import PIPELINES

class Loader:


    def __init__(self, source_dir, train_dir, val_dir, output_folder, tile_size=512):
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

            self.tile_size = tile_size
            self.source_dir = source_dir
            self.train_dir = train_dir
            self.val_dir = val_dir
            self.output_folder = output_folder

            # Initialize a list to store titles of processed files (optional)


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




    def alter_radd_data_to_label(self, radd_tiles_path):
        # Iterate over files in the training directory
        for filename in os.listdir(radd_tiles_path):
            if filename.endswith('_radd.tif'):
                radd_alert_path = os.path.join(radd_tiles_path, filename)
                output_filename = os.path.splitext(filename)[0] + '_.tif'
                output_path = os.path.join(radd_tiles_path, output_filename)

                # Open the RADD alert raster file
                with rasterio.open(radd_alert_path) as src:
                    # Read the data
                    radd_data = src.read(1)  # Assuming RADD data is in the first band

                    # Convert values greater than 0 to 1
                    radd_data[radd_data > 0] = 1

                    # Save the altered data to the new file
                    with rasterio.open(
                        output_path,
                        'w',
                        driver='GTiff',
                        height=radd_data.shape[0],
                        width=radd_data.shape[1],
                        count=1,
                        dtype=radd_data.dtype,
                        crs=src.crs,
                        transform=src.transform
                    ) as dst:
                        dst.write(radd_data, 1)


    def filter_stacks(self, stack_directory, alert_label_value, min_alert_pixels=0):
        """
        Filters stacks based on the count of RADD alert labels.

        Args:
            alert_label_value (int): Pixel value that represents a RADD alert.
            min_alert_pixels (int): Minimum number of alert pixels required to keep a stack.
        """
        alert_counts = []
        for file in os.listdir(stack_directory):
            if file.endswith('_radd_.tif'):
                file_path = os.path.join(stack_directory, file)

                with rasterio.open(file_path) as src:
                    radd_alerts = src.read(1)  # Assuming the RADD alert band is the first band
                    alert_count = np.count_nonzero(radd_alerts == alert_label_value)
                    print(f"file:{file}, alert count: {alert_count}")
                    alert_counts.append(alert_count)


                    src.close()
                    if alert_count < min_alert_pixels:
                        # Construct base name and delete corresponding files
                        base_name = file.replace('_radd_.tif', '')
                        for suffix in ['_radd_.tif', '_radd.tif', '_sentinel.tif']:
                            file_to_delete = os.path.join(stack_directory, base_name + suffix)
                            if os.path.exists(file_to_delete):
                                os.remove(file_to_delete)
                                print(f"Removed file {file_to_delete}")


                        print(f"Removed stack {file} due to insufficient RADD alerts:{alert_count}.")


        return alert_counts

    def calculate_alert_distribution(self, alert_counts, num_groups=5):
        """
        Calculates the distribution of alert counts into equal groups.

        Args:
            alert_counts (list): List of alert counts from RADD alert bands.
            num_groups (int): Number of groups to divide the alert counts into.

        Returns:
            dict: A dictionary with the range as keys and count of alerts in each range as values.
        """
        max_alert_count = max(alert_counts)
        min_alert_count = min(alert_counts)
        range_size = (max_alert_count - min_alert_count) / num_groups
        distribution = {}

        for i in range(num_groups):
            lower_bound = min_alert_count + i * range_size
            upper_bound = lower_bound + range_size
            if i == num_groups - 1:
                # Ensure the last group includes the max value
                upper_bound = max_alert_count + 1

            # Count how many alerts fall into this range
            count = sum(lower_bound <= count <= upper_bound for count in alert_counts)
            distribution[f"{lower_bound:.2f}-{upper_bound:.2f}"] = count

        return distribution


    def calculate_band_statistics(self, directories, nan_value=-9999):
        """
        Calculates the mean and standard deviation of each Sentinel band,
        excluding specified NaN values.

        Args:
            directories (list): List of directories to process.
            nan_value (int): Value to be treated as NaN and excluded from calculations.

        Returns:
            dict: Dictionary containing means and stds for each band.
        """
        band_sums = []
        band_squares = []
        band_counts = []
        initialized = False

        for directory in directories:
            for file in os.listdir(directory):
                if file.endswith('_sentinel.tif'):
                    file_path = os.path.join(directory, file)

                    with rasterio.open(file_path) as src:
                        bands = src.read()

                        if not initialized:
                            # Initialize sum, sum of squares, and count arrays
                            band_sums = [np.zeros(band.shape) for band in bands]
                            band_squares = [np.zeros(band.shape) for band in bands]
                            band_counts = [0 for _ in bands]
                            initialized = True

                        for i, band in enumerate(bands):
                            valid_mask = band != nan_value
                            band_sums[i] += np.where(valid_mask, band, 0)
                            band_squares[i] += np.where(valid_mask, np.square(band), 0)
                            band_counts[i] += np.count_nonzero(valid_mask)

        means = [np.sum(band_sum) / band_count if band_count > 0 else 0 for band_sum, band_count in zip(band_sums, band_counts)]
        stds = [np.sqrt(max(np.sum(band_square) / band_count - mean ** 2, 0)) if band_count > 0 else 0 for band_square, band_count, mean in zip(band_squares, band_counts, means)]

        return {"means": means, "stds": stds}

    def normalize_data(self, directory, nodata_value=-9999):
        """
        Normalizes each band of the Sentinel-2 data in the given directory to a range of 0-1,
        ignoring specified nodata values.

        Args:
            directory (str): Directory containing Sentinel-2 data files.
            nodata_value (int): Value to be treated as 'no data' and excluded from normalization.
        """
        for file in os.listdir(directory):
            if file.endswith('_sentinel.tif'):
                file_path = os.path.join(directory, file)

                with rasterio.open(file_path, 'r+') as src:
                    bands = src.read()

                    # Normalize each band individually
                    normalized_bands = np.zeros_like(bands, dtype=np.float32)
                    for i, band in enumerate(bands):
                        valid_mask = band != nodata_value
                        valid_band = band[valid_mask]
                        max_value = valid_band.max()
                        if max_value > 1:
                            print("yo")
                        min_value = valid_band.min()
                        normalized_band = np.where(valid_mask, (band - min_value) / (max_value - min_value), nodata_value)
                        normalized_bands[i] = normalized_band

                    # Write the normalized data back to the file
                    src.write(normalized_bands)



    def plot_metrics_from_log(self, log_filename):
        # Regular expression patterns
        performance_pattern = re.compile(
            r"Iter\(val\) \[\d+\]\s+aAcc: ([\d.]+), mIoU: ([\d.]+), mAcc: ([\d.]+), "
            r"IoU.Unburnt land: ([\d.]+), IoU.Burn scar: ([\d.]+), "
            r"Acc.Unburnt land: ([\d.]+), Acc.Burn scar: ([\d.]+)"
        )

        training_pattern = re.compile(
            r"Iter \[\d+/\d+\]\s+lr: [\d.e-]+, .+decode.loss_dice: ([\d.]+), "
            r"decode.acc_seg: ([\d.]+), aux.loss_dice: ([\d.]+), "
            r"aux.acc_seg: ([\d.]+), loss: ([\d.]+)"
        )

        # Initialize lists for performance metrics
        aAccs, mIoUs, mAccs = [], [], []
        IoU_unburnt, IoU_burn, Acc_unburnt, Acc_burn = [], [], [], []

        # Initialize lists for training metrics
        decode_loss_dices, decode_acc_segs = [], []
        aux_loss_dices, aux_acc_segs, total_losses = [], [], []

        # Read the log file and parse the metrics
        if not os.path.exists(log_filename):
            print(f"Log file {log_filename} not found.")
            return

        with open(log_filename, 'r') as file:
            for line in file:
                perf_match = performance_pattern.search(line)
                train_match = training_pattern.search(line)

                if perf_match:
                    aAccs.append(float(perf_match.group(1)))
                    mIoUs.append(float(perf_match.group(2)))
                    mAccs.append(float(perf_match.group(3)))
                    IoU_unburnt.append(float(perf_match.group(4)))
                    IoU_burn.append(float(perf_match.group(5)))
                    Acc_unburnt.append(float(perf_match.group(6)))
                    Acc_burn.append(float(perf_match.group(7)))

                if train_match:
                    decode_loss_dices.append(float(train_match.group(1)))
                    decode_acc_segs.append(float(train_match.group(2)))
                    aux_loss_dices.append(float(train_match.group(3)))
                    aux_acc_segs.append(float(train_match.group(4)))
                    total_losses.append(float(train_match.group(5)))

        # Plotting
        plt.figure(figsize=(15, 20))

        # Plot performance metrics
        plt.subplot(5, 1, 1)
        plt.plot(mIoUs, label='mIoU')


        plt.plot(aAccs, label='aAcc')
        plt.plot(mAccs, label='mAcc')
        plt.plot(IoU_unburnt, label='IoU Unburnt Land')
        plt.plot(IoU_burn, label='IoU Burn Scar')
        plt.plot(Acc_unburnt, label='Acc Unburnt Land')
        plt.plot(Acc_burn, label='Acc Burn Scar')
        plt.xlabel('Iteration')
        plt.ylabel('Performance Metrics')
        plt.title('Performance Metrics Over Iterations')
        plt.legend()

        # Plot training metrics - Decode Loss Dice
        plt.subplot(5, 1, 2)
        plt.plot(decode_loss_dices, label='Decode Loss Dice')
        plt.xlabel('Iteration')
        plt.ylabel('Decode Loss Dice')
        plt.title('Decode Loss Dice Over Iterations')
        plt.legend()

        # Plot training metrics - Decode Accuracy Segment
        plt.subplot(5, 1, 3)
        plt.plot(decode_acc_segs, label='Decode Accuracy Segment')
        plt.xlabel('Iteration')
        plt.ylabel('Decode Accuracy Segment')
        plt.title('Decode Accuracy Segment Over Iterations')
        plt.legend()

        # Plot training metrics - Aux Loss Dice & Aux Accuracy Segment
        plt.subplot(5, 1, 4)
        plt.plot(aux_loss_dices, label='Aux Loss Dice')
        plt.plot(aux_acc_segs, label='Aux Accuracy Segment')
        plt.xlabel('Iteration')
        plt.ylabel('Aux Metrics')
        plt.title('Auxiliary Metrics Over Iterations')
        plt.legend()

        # Plot training metrics - Total Loss
        plt.subplot(5, 1, 5)
        plt.plot(total_losses, label='Total Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Total Loss')
        plt.title('Total Loss Over Iterations')
        plt.legend()

        plt.tight_layout()
        plt.show()




    # def plot_training_progress(self, checkpoint_dir):
    #     metrics = {'loss': [], 'accuracy': []}  # Add more metrics as needed
    #
    #     # Step 1: Collect metrics from checkpoint files
    #     for filename in sorted(os.listdir(checkpoint_dir)):
    #         if filename.endswith('.json'):
    #             with open(os.path.join(checkpoint_dir, filename)) as f:
    #                 for line in f:
    #                     try:
    #                         data = json.loads(line)
    #                         metrics['loss'].append(data['loss'])
    #                         metrics['accuracy'].append(data['accuracy'])
    #                     except json.JSONDecodeError:
    #                         continue  # Skip lines that are not valid JSON
    #
    #                 data = json.load(f)
    #                 # Assuming the JSON structure contains 'loss' and 'accuracy'
    #                 metrics['loss'].append(data['loss'])
    #                 metrics['accuracy'].append(data['accuracy'])
    #
    #     # Step 2: Plot the metrics
    #     fig, ax1 = plt.subplots()
    #
    #     color = 'tab:red'
    #     ax1.set_xlabel('Epoch')
    #     ax1.set_ylabel('Loss', color=color)
    #     ax1.plot(metrics['loss'], color=color)
    #     ax1.tick_params(axis='y', labelcolor=color)
    #
    #     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #
    #     color = 'tab:blue'
    #     ax2.set_ylabel('Accuracy', color=color)
    #     ax2.plot(metrics['accuracy'], color=color)
    #     ax2.tick_params(axis='y', labelcolor=color)
    #
    #
    #





