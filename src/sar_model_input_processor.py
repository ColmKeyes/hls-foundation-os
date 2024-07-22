

import os
import rasterio
import numpy as np
import re
from datetime import datetime


class SARLoader:

    def __init__(self, sen2_stack_path, output_path, data_type):
        self.sen2_stack_path = sen2_stack_path
        self.output_path = output_path
        self.nodata_value = -9999  # Adjust this as needed
        self.data_type = data_type

    def compute_global_min_max(self, input_folder, bands=[6, 7]):
        global_min = np.full(len(bands), np.inf)
        global_max = np.full(len(bands), -np.inf)

        for filename in os.listdir(input_folder):
            if filename.endswith('_sentinel_agb_normalized_sar_masked.tif') and self.data_type in filename:
                filepath = os.path.join(input_folder, filename)
                with rasterio.open(filepath) as src:
                    for band_idx in bands:
                        band = src.read(band_idx).astype(np.float32)
                        valid_mask = band > self.nodata_value  # Assuming self.nodata_value is defined
                        valid_pixels = band[valid_mask]
                        if valid_pixels.size > 0:
                            # Map band 6 to index 0 and band 7 to index 1
                            index = band_idx - 6
                            global_min[index] = min(np.min(valid_pixels), global_min[index])
                            global_max[index] = max(np.max(valid_pixels), global_max[index])

        return global_min, global_max

    # def compute_global_min_max(self, input_folder, bands=[6, 7]):
    #     global_min = np.full(len(bands), np.inf)
    #     global_max = np.full(len(bands), -np.inf)
    #
    #     for filename in os.listdir(input_folder):
    #         if filename.endswith('_sentinel_agb_normalized_sar_masked.tif') and self.data_type in filename:
    #             filepath = os.path.join(input_folder, filename)
    #             with rasterio.open(filepath) as src:
    #                 for i, band_idx in enumerate(bands):
    #                     band = src.read(band_idx).astype(np.float32)
    #                     valid_mask = band > self.nodata_value
    #                     valid_pixels = band[valid_mask]
    #                     if valid_pixels.size > 0:
    #
    #                         index = i - 6  # Subtract 6 to align with the 0-indexed arrays global_min and global_max
    #                         global_min[index] = min(np.min(valid_pixels), global_min[index])
    #                         global_max[index] = max(np.max(valid_pixels), global_max[index])
    #                         # global_min[i - bands[0]] = min(np.min(valid_pixels), global_min[i - bands[0]])
    #                         # global_max[i - bands[0]] = max(np.max(valid_pixels), global_max[i - bands[0]])
    #
    #     return global_min, global_max

    def normalize_images_global(self, input_file, output_file, global_min, global_max, bands=[6, 7]):

        with rasterio.open(input_file) as src:
            meta = src.meta.copy()
            meta.update(dtype=rasterio.float32)
            # src.close()
            with rasterio.open(output_file, 'w', **meta) as dst:
                for i in range(1, src.count + 1):
                    band = src.read(i).astype(np.float32)
                    if i in bands:
                        valid_mask = band > self.nodata_value
                        scaled_band = np.where(valid_mask, (band - global_min[i - bands[0]]) / (global_max[i - bands[0]] - global_min[i - bands[0]]), self.nodata_value)
                        dst.write(scaled_band, i)
                    else:
                        dst.write(band, i)

        # os.remove(input_path)
        print(f"Normalized and masked file saved as: {output_file}")

    def compute_global_mean_std(self, input_folder, bands=[6, 7]):
        sum_means = np.zeros(len(bands))
        sum_variances = np.zeros(len(bands))
        total_pixels = np.zeros(len(bands))

        for filename in os.listdir(input_folder):
            if filename.endswith('_normalized.tif') and self.data_type in filename:
                filepath = os.path.join(input_folder, filename)
                with rasterio.open(filepath) as src:
                    for band_idx in bands:
                        band = src.read(band_idx).astype(np.float32)
                        valid_mask = band > self.nodata_value
                        valid_pixels = band[valid_mask]

                        # Map band 6 to index 0 and band 7 to index 1
                        index = band_idx - 6
                        sum_means[index] += valid_pixels.mean() * valid_pixels.size
                        sum_variances[index] += valid_pixels.var() * valid_pixels.size
                        total_pixels[index] += valid_pixels.size

        # Avoid division by zero for total_pixels
        total_pixels[total_pixels == 0] = 1
        global_means = sum_means / total_pixels
        global_stds = np.sqrt(sum_variances / total_pixels)

        return global_means, global_stds


    def rename_processed_files(self):
        # Define the pattern to match the filenames based on the data type
        if self.data_type == 'bsc':
            pattern = re.compile(rf'{self.data_type}_(\d{{8}})_sen2_(\d{{8}})_(T\d{{2}}\w{{3}})_agb_radd_fmask_stack_(\d+)_(\d+)_sentinel_agb_normalized_sar_masked_normalized\.tif$')
        elif self.data_type == 'coh':
            pattern = re.compile(rf'{self.data_type}_(\d{{8}})_(\d{{8}})_sen2_(\d{{8}})_(T\d{{2}}\w{{3}})_agb_radd_fmask_stack_(\d+)_(\d+)_sentinel_agb_normalized_sar_masked_normalized\.tif$')
        else:
            raise ValueError("Invalid data type. Use 'bsc' or 'coh'.")

        for filename in os.listdir(self.output_path):
            match = pattern.match(filename)
            if match:
                if self.data_type == 'bsc':
                    date, tile, width, height = match.groups()[1], match.groups()[2], match.groups()[3], match.groups()[4]
                elif self.data_type == 'coh':
                    date, tile, width, height = match.groups()[2], match.groups()[3], match.groups()[4], match.groups()[5]

                # Construct the new filename
                new_filename = f"{date}_{tile}_agb_radd_fmask_stack_{width}_{height}_sentinel_agb_normalized_{self.data_type}_masked_normalized.tif"
                new_filepath = os.path.join(self.output_path, new_filename)

                # Rename the file
                old_filepath = os.path.join(self.output_path, filename)
                if os.path.exists(new_filepath):
                    continue
                os.rename(old_filepath, new_filepath)
                print(f"Renamed {filename} to {new_filename}")

    def convert_dates_to_doy(self):
        # pattern = re.compile(r'(\d{4})(\d{2})(\d{2})_(T\d{2}\w{3})_agb_radd_fmask_stack_(\d+)_\d+_sentinel_agb_normalized_(\w+)_masked_normalized\.tif$')
        pattern = re.compile(r'(\d{4})(\d{2})(\d{2})_(T\d{2}\w{3})_agb_radd_fmask_stack_(\d+)_(\d+)_sentinel_agb_normalized_(\w+)_masked_normalized\.tif$')

        for filename in os.listdir(self.output_path):
            match = pattern.match(filename)
            if match:
                year, month, day, tile, width, height, data_type = match.groups()
                date = datetime(int(year), int(month), int(day))
                doy = date.strftime('%Y%j')

                new_filename = f"{doy}_{tile}_agb_radd_fmask_stack_{width}_{height}_sentinel_agb_normalized_{data_type}_masked_normalized.tif"
                new_filepath = os.path.join(self.output_path, new_filename)

                # Rename the file
                old_filepath = os.path.join(self.output_path, filename)
                if os.path.exists(new_filepath):
                    continue
                os.rename(old_filepath, new_filepath)
                print(f"Renamed {filename} to {new_filename}")


    def apply_mask_and_save_to_sar_bands(self, combined_stack_path, mask_band_index=7, output_file_path=None):
        """
        Applies the mask from the Sentinel-2 stack (assumed to be the 8th band by default) to the SAR data bands
        and saves the modified stack to the specified output path.

        Args:
            combined_stack_path (str): Path to the combined Sentinel-2 and SAR data stack.
            mask_band_index (int): Index of the band containing the mask within the stack.
            output_file_path (str): Path where the modified stack will be saved.
        """
        if output_file_path is None:
            output_file_path = combined_stack_path.replace('_sar.tif', '_sar_masked.tif')

        # Open the combined stack
        with rasterio.open(combined_stack_path) as src:
            meta = src.meta.copy()
            data = src.read()

            # Identify where the first band's value is -9999
            mask = data[0, :, :] == -9999  # Band indices are 0-based

            # Set values in bands 6 and 7 to -9999 where the mask is True
            data[5, mask] = -9999  # Applying mask to band 6
            data[6, mask] = -9999  # Applying mask to band 7

            # Save the masked stack
            with rasterio.open(output_file_path, 'w', **meta) as dst:
                dst.write(data)
                print(f"Masked stack saved to {output_file_path}")
                dst.close()
            src.close()

    # def rename_processed_files(self):
    #     # Define the pattern to match the filenames
    #     pattern = re.compile(rf'{self.data_type}_(\d{{8}})_sen2_(\d{{8}})_(T\d{{2}}\w{{3}})_agb_radd_fmask_stack_(\d{{4}})_(\d{{4}})_sentinel_agb_normalized_sar_masked_normalized\.tif$')
    #
    #     for filename in os.listdir(self.output_path):
    #         match = pattern.match(filename)
    #         if match:
    #             date1, date2, tile, width, height = match.groups()
    #
    #             # Construct the new filename
    #             new_filename = f"{date2}_{tile}_agb_radd_fmask_stack_{width}_{height}_sentinel_agb_normalized_{self.data_type}_masked_normalized.tif"
    #             new_filepath = os.path.join(self.output_path, new_filename)
    #
    #             # Rename the file
    #             old_filepath = os.path.join(self.output_path, filename)
    #             os.rename(old_filepath, new_filepath)
    #             print(f"Renamed {filename} to {new_filename}")
    #
    #             # # Extract the components from the filename
    #             # data_type, date1, date2, date3, tile, width, height = match.groups()
    #             #
    #             # # Construct the new filename
    #             # new_filename = f"{date3}_{tile}_agb_radd_fmask_stack_{width}_{height}_sentinel_agb_normalized_{self.data_type}_masked_normalized.tif"
    #             # new_filepath = os.path.join(self.output_path, new_filename)
    #             #
    #             # # Rename the file
    #             # old_filepath = os.path.join(self.output_path, filename)
    #             # os.rename(old_filepath, new_filepath)
    #             # print(f"Renamed {filename} to {new_filename}")





    # def compute_global_mean_std(self, input_folder, bands=[6, 7]):
    #     sum_means = np.zeros(len(bands))
    #     sum_variances = np.zeros(len(bands))
    #     total_pixels = 0
    #
    #     for filename in os.listdir(input_folder):
    #         if filename.endswith('_normalized.tif') and self.data_type in filename:
    #             filepath = os.path.join(input_folder, filename)
    #             with rasterio.open(filepath) as src:
    #                 for i, band_idx in enumerate(bands):
    #                     band = src.read(band_idx).astype(np.float32)
    #                     valid_mask = band > self.nodata_value
    #                     valid_pixels = band[valid_mask]
    #                     sum_means[i - bands[0]] += valid_pixels.mean()
    #                     sum_variances[i - bands[0]] += valid_pixels.var()
    #                     total_pixels += valid_pixels.size
    #
    #     global_means = sum_means / total_pixels
    #     global_stds = np.sqrt(sum_variances / total_pixels)
    #
    #     return global_means, global_stds



# class SARLoader:
#     def __init__(self, sen2_stack_dir, output_dir):
#         self.sen2_stack_dir = sen2_stack_dir
#         self.output_dir = output_dir

#     def calculate_global_statistics(self, input_folder, data_type, bands=[6, 7] ):
#         global_min = np.full(len(bands), np.inf)
#         global_max = np.full(len(bands), -np.inf)
#         sums = np.zeros(len(bands))
#         sums_squares = np.zeros(len(bands))
#         pixel_counts = np.zeros(len(bands))
#
#         for filename in os.listdir(input_folder):
#             if filename.endswith('_sentinel_agb_normalized_sar.tif') and data_type in filename:
#                 filepath = os.path.join(input_folder, filename)
#                 with rasterio.open(filepath) as src:
#                     for i, band_index in enumerate(bands):
#                         band = src.read(band_index).astype(np.float32)
#                         valid_mask = band > 0  # Assuming -9999 is the no-data value
#                         valid_pixels = band[valid_mask]
#
#                         # Update min and max
#                         global_min[i] = min(np.min(valid_pixels), global_min[i])
#                         global_max[i] = max(np.max(valid_pixels), global_max[i])
#
#                         # Update sum and sum of squares for mean and std calculation
#                         sums[i] += np.sum(valid_pixels)
#                         sums_squares[i] += np.sum(np.square(valid_pixels))
#                         pixel_counts[i] += np.count_nonzero(valid_mask)
#
#         global_means = sums / pixel_counts
#         global_stds = np.sqrt((sums_squares / pixel_counts) - np.square(global_means))
#
#         return global_min, global_max, global_means, global_stds
#
#     def normalize_sar_bands(self, input_file, global_mean, global_std, bands=[6, 7]):
#         with rasterio.open(input_file) as src:
#             meta = src.meta.copy()
#             data = src.read()
#
#             for i, band in enumerate(bands):
#                 # Apply normalization: (band - global_mean) / global_std
#                 data[band - 1] = (data[band - 1] - global_mean[i]) / global_std[i]
#
#             output_file_path = input_file.replace('_sar_masked.tif', '_sar_masked_normalized.tif')
#             with rasterio.open(output_file_path, 'w', **meta) as dst:
#                 dst.write(data)
#
#             print(f"Normalized SAR bands in {input_file} and saved to {output_file_path}")

    #
    # def normalize_sar_bands(self, combined_stack_dir, bands_to_normalize=[6, 7], nodata_value=-9999):
    #     """
    #     Normalizes the SAR bands in the combined Sentinel-2 and SAR stacks.
    #
    #     Args:
    #         combined_stack_dir (str): Directory containing combined stack files.
    #         bands_to_normalize (list): Bands to normalize (default is [6, 7] for SAR).
    #         nodata_value (int): No data value to be ignored in the calculations.
    #     """
    #     for filename in os.listdir(combined_stack_dir):
    #         if filename.endswith('_sentinel_agb_normalized_sar.tif'):
    #             filepath = os.path.join(combined_stack_dir, filename)
    #             with rasterio.open(filepath, 'r+') as src:
    #                 for band_idx in bands_to_normalize:
    #                     band = src.read(band_idx).astype(np.float32)
    #                     valid_mask = band != nodata_value
    #
    #                     # Normalize the band to the 0-1 range based on its own min and max
    #                     min_val, max_val = band[valid_mask].min(), band[valid_mask].max()
    #                     band[valid_mask] = (band[valid_mask] - min_val) / (max_val - min_val)
    #                     src.write_band(band_idx, band)
    #
    # def compute_global_statistics(self, combined_stack_dir, bands=[1, 2, 3, 4, 5, 6], nodata_value=-9999):
    #     """
    #     Computes global min, max, mean, and std for specified bands across all files in the directory.
    #
    #     Args:
    #         combined_stack_dir (str): Directory containing combined stack files.
    #         bands (list): Bands to compute statistics for.
    #         nodata_value (int): No data value to be ignored in the calculations.
    #     """
    #     global_min = np.full(len(bands), np.inf)
    #     global_max = np.full(len(bands), -np.inf)
    #     sum_means = np.zeros(len(bands))
    #     sum_squares = np.zeros(len(bands))
    #     total_pixels = np.zeros(len(bands))
    #
    #     for filename in os.listdir(combined_stack_dir):
    #         if filename.endswith('_sentinel_agb_normalized_sar.tif'):
    #             filepath = os.path.join(combined_stack_dir, filename)
    #             with rasterio.open(filepath) as src:
    #                 for i, band_idx in enumerate(bands):
    #                     band = src.read(band_idx).astype(np.float32)
    #                     valid_mask = band != nodata_value
    #                     valid_pixels = band[valid_mask]
    #                     if valid_pixels.size > 0:
    #                         global_min[i] = min(global_min[i], valid_pixels.min())
    #                         global_max[i] = max(global_max[i], valid_pixels.max())
    #                         sum_means[i] += valid_pixels.sum()
    #                         sum_squares[i] += np.sum(np.square(valid_pixels))
    #                         total_pixels[i] += valid_pixels.size
    #
    #     global_means = sum_means / total_pixels
    #     global_stds = np.sqrt((sum_squares / total_pixels) - np.square(global_means))
    #
    #     return {"min": global_min, "max": global_max, "means": global_means, "stds": global_stds}