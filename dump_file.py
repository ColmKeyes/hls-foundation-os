#####################################################################################
### Final Clearout
###
#####################################################################################




#################
## 2 sar model run input processor
#################



# import os
# from src.sar_model_input_processor import SARLoader
#
# sen2_stack_dir = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\globalnorm\15000_minalerts"
# output_dir = r"E:\Data\Sentinel2_data\30pc_cc\Borneo_June2021_Dec_2023_30pc_cc_stacks_agb_radd_sar"
#
# sar_model_processing = SARLoader(sen2_stack_dir, output_dir)
#
# # Set this to "coh" or "bsc" based on which data you want to process
# data_type = "bsc"
#
# # Calculate Global Statistics for SAR Bands
# global_min, global_max, global_mean, global_std = sar_model_processing.calculate_global_statistics(output_dir, bands=[6, 7], data_type=data_type)
#
# # Normalize SAR bands in each stack using global statistics and reapply masks
# for file in os.listdir(output_dir):
#     if file.endswith(f'_sentinel_agb_normalized_sar_masked.tif') and data_type in file:
#         combined_stack_path = os.path.join(output_dir, file)
#
#         # Normalize SAR bands using the global statistics
#         sar_model_processing.normalize_sar_bands(combined_stack_path, global_mean, global_std, bands=[6, 7])
#
#         # Reapply masks after normalization
#         sar_model_processing.apply_mask_and_save_to_sar_bands(combined_stack_path)
#
#         print(f"Processed {combined_stack_path} for data type: {data_type}")





#
# import os
#
# from src.sar_model_input_processor import SARLoader
# sen2_stack_dir = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\globalnorm\15000_minalerts"  # r"E:\Data\Sentinel2_data\30pc_cc\Borneo_June2021_Dec_2023_30pc_cc_stacks_agb_radd"
# # Define the output directory for processed files
# output_dir = r"E:\Data\Sentinel2_data\30pc_cc\Borneo_June2021_Dec_2023_30pc_cc_stacks_agb_radd_sar"
#
#
#
# sar_model_processing = SARLoader(sen2_stack_dir, output_dir)
#
# # Step 1: Calculate Global Statistics for SAR Bands
# global_min, global_max, global_mean, global_std = sar_model_processing.calculate_global_statistics(output_dir, bands=[6, 7])
#
# # Step 2: Normalize SAR bands in each stack using global statistics and reapply masks
# for file in os.listdir(output_dir):
#     if file.endswith('_sentinel_agb_normalized_sar.tif'):
#         combined_stack_path = os.path.join(output_dir, file)
#
#         # Normalize SAR bands using the global statistics
#         sar_model_processing.normalize_sar_bands(combined_stack_path, global_mean, global_std, bands=[6, 7])
#
#         # Optionally, reapply masks after normalization if necessary
#         sar_model_processing.apply_mask_and_save_to_sar_bands(combined_stack_path)
#


#
# for file in os.listdir(output_dir):
#     if file.endswith('_sentinel_agb_normalized_sar.tif'):
#         ############
#         # Step 4: re-mask SAR stacks
#         ############
#         combined_stack_path = os.path.join(output_dir, file)
#         sar_model_processing.apply_mask_and_save_to_sar_bands(combined_stack_path)
#
#         ############
#         # Step 5: Normalise SAR stacks & calculate statistics
#         ############
#




#################
## test_analysis
#################

# class TestAnalysis:
#     def __init__(self, log_file):
#         self.log_file = log_file
#         self.data = self.load_log_data()
#
#     def load_log_data(self):
#         """Load the log data from the JSON file."""
#         with open(self.log_file, 'r') as f:
#             return json.load(f)
#
#     def get_metrics(self):
#         """Get all metrics contained in the log file."""
#         return self.data['metric']


#
# class TestAnalysis:
#     def __init__(self, log_file):
#         # Expecting log_files to be a list of log file paths for the runs of a single model
#         self.log_file = log_file
#         self.data = [self.load_log_data(log_file)]# for log_file in log_files]
#
#     def load_log_data(self, log_file):
#         """Load the log data from the JSON file."""
#         with open(log_file, 'r') as f:
#             return json.load(f)
#
#     def get_average_metrics(self):
#         """Calculate the average metrics across multiple runs."""
#         metrics_list = [run_data['metric'] for run_data in self.data]
#
#         # Initialize a dictionary to store the average metrics
#         average_metrics = {}
#
#         # Get the list of all metric names
#         metric_names = metrics_list[0].keys()
#
#         # Calculate the average for each metric
#         for name in metric_names:
#             metric_values = [metrics[name] for metrics in metrics_list]
#             average_metrics[name] = np.mean(metric_values)
#
#         return average_metrics


#################
## sar model input processor
#################


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




#################
## model management
#################

#
# def compare_checkpoints(original_checkpoint_path, updated_checkpoint_path):
#     """Compare the weights of two checkpoints."""
#     original_checkpoint = torch.load(original_checkpoint_path, map_location='cpu')
#     updated_checkpoint = torch.load(updated_checkpoint_path, map_location='cpu')
#
#     for key in original_checkpoint['state_dict']:
#         if key in updated_checkpoint['state_dict']:
#             original_value = original_checkpoint['state_dict'][key]
#             updated_value = updated_checkpoint['state_dict'][key]
#             # Ensure we compare tensors of the same shape
#             if original_value.shape != updated_value.shape:
#                 print(f"Shape mismatch for: {key}")
#             elif not torch.equal(original_value, updated_value):
#                 print(f"Value changed for: {key}")



#################
## model input processor
#################
# def find_global_min_max(self, folder_path, nodata_value=-9999):
#     """
#     Finds the global minimum and maximum values across the first 6 bands of all Sentinel-2 data files in a folder,
#     excluding no-data and values below zero.
#     """
#     global_min = np.inf
#     global_max = -np.inf
#
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith('_sentinel_normalized.tif'):
#             file_path = os.path.join(folder_path, file_name)
#             with rasterio.open(file_path) as src:
#                 for i in range(1, 7):  # Only consider the first 6 bands
#                     band = src.read(i).astype(np.float32)
#                     valid_mask = (band != nodata_value) & (band >= 0)
#                     valid_data = band[valid_mask]
#
#                     if valid_data.size > 0:  # Check if there's any valid data
#                         band_min = valid_data.min()
#                         band_max = valid_data.max()
#
#                         if band_min < global_min:
#                             global_min = band_min
#                         if band_max > global_max:
#                             global_max = band_max
#     print(f"Global Max: {global_max}, Global Min: {global_min}")
#     return global_min, global_max
#
# def normalize_dataset(self, folder_path, nodata_value=-9999):
#     """
#     Normalizes all Sentinel-2 data files in a folder using global min and max values.
#     """
#     global_min, global_max = self.find_global_min_max(folder_path, nodata_value)
#
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith('_sentinel_normalized.tif'):
#             file_path = os.path.join(folder_path, file_name)
#             output_path = os.path.splitext(file_path)[0] + '_DS_normalized.tif'
#
#             with rasterio.open(file_path) as src:
#                 meta = src.meta
#                 meta.update(dtype=rasterio.float32)
#
#                 with rasterio.open(output_path, 'w', **meta) as dst:
#                     for i in range(1, src.count + 1):
#                         band = src.read(i).astype(np.float32)
#
#                         if i <= 6:  # Normalize the first 6 bands
#                             valid_mask = (band != nodata_value) & (band >= 0)
#                             normalized_band = np.where(valid_mask, (band - global_min) / (global_max - global_min), nodata_value)
#                             dst.write_band(i, normalized_band)
#
#                         else:  # Directly add the 7th band without normalization
#                             dst.write_band(i, band)
#             #os.remove(file_path)
#             print(f"Normalized file saved as: {output_path}")






#################
## model analysis
#################

# def add_legend(ax, cmap, labels):
#     patches = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(len(labels))]
#     ax.legend(handles=patches, labels=labels, loc='lower right')

# def plot_image_triplet(input_path, output_path, ground_truth_path, image_basename, data_type, save_path):
#     suffix = ''
#     if data_type == 'backscatter':
#         suffix = '_bsc_masked_normalized'
#     elif data_type == 'coherence':
#         suffix = '_coh_masked_normalized'
#
#     run_number = ''
#     for part in save_path.split('_'):
#         if part.startswith('run'):
#             run_number = part
#             break
#
#
#     input_image = os.path.join(input_path, f"{image_basename}_sentinel_agb_normalized{suffix}.tif")
#     output_image = os.path.join(output_path, f"{image_basename}_sentinel_agb_normalized{suffix}_pred_{run_number}_op.tif")
#     ground_truth_image = os.path.join(ground_truth_path, f"{image_basename}_radd_labelled_agb.tif")
#
#     class_cmap = ListedColormap(['lightblue', 'orange', 'grey'])
#     difference_cmap = ListedColormap(['orange', "lightblue", 'grey'])
#
#     inProj = Proj(init='epsg:32650')
#     outProj = Proj(init='epsg:4326')
#
#     # Transform the corners of the image to lat/long
#     left, bottom = transform(inProj, outProj, 230536.5095550618, -107359.60842863785)
#     right, top = transform(inProj, outProj, 245873.9621862365, -92022.15579746317)
#
#     with rasterio.open(input_image) as src:
#         if data_type == "backscatter" or data_type == "coherence":
#             input_rgb = src.read([1, 2, 3])
#
#         input_rgb = src.read([1, 2, 3])
#         input_rgb = np.clip(input_rgb / 0.1, 0, 1)
#
#     with rasterio.open(output_image) as src:
#         output_data = src.read(1).astype(float)
#         output_data[output_data == -1] = np.nan
#
#     with rasterio.open(ground_truth_image) as src:
#         gt_data = src.read(1).astype(float)
#         gt_data[gt_data == -9999] = np.nan
#
#     # Create the difference map, reversing the color for change and no change
#     difference_map = np.where(np.isnan(output_data) | np.isnan(gt_data), 2, 1 - np.abs(gt_data - output_data))
#
#     fig, axs = plt.subplots(1, 4, figsize=(30, 8), sharex=True, sharey=True)
#     axs[0].imshow(input_rgb.transpose(1, 2, 0), extent=[left, right, bottom, top])
#     axs[0].set_title('Input RGB')
#     axs[0].set_xlabel('Longitude')
#     axs[0].set_ylabel('Latitude')
#
#     show_class_data(output_data, axs[1], class_cmap, 'Output', [left, right, bottom, top])
#     show_class_data(gt_data, axs[2], class_cmap, 'Ground Truth', [left, right, bottom, top])
#
#     # Updated to reverse the colors for the difference map
#     axs[3].imshow(difference_map, cmap=difference_cmap, vmin=0, vmax=2, extent=[left, right, bottom, top])
#     axs[3].set_title('Difference Map')
#     axs[3].set_xlabel('Longitude')
#     axs[3].set_ylabel('Latitude')
#
#     add_legend(axs[2], class_cmap, ['Non-Disturbed', 'Disturbed', 'No Data'])
#     add_legend(axs[3], difference_cmap, ['Difference', 'No Difference', 'No Data'])
#
#     output_filename = os.path.join(save_path, f"{image_basename}_comparison.png")
#     plt.tight_layout()
#     plt.savefig(output_filename, bbox_inches='tight')
#     plt.close()
#
#     print(f"File saved at {output_filename}")



# def plot_image_triplet(input_path, output_path, ground_truth_path, image_basename, data_type, save_path):
#     suffix = ''
#     if data_type == 'backscatter':
#         suffix = '_bsc_masked_normalized'
#     elif data_type == 'coherence':
#         suffix = '_coh_masked_normalized'
#
#     input_image = os.path.join(input_path, f"{image_basename}_sentinel_agb_normalized{suffix}.tif")
#     output_image = os.path.join(output_path, f"{image_basename}_sentinel_agb_normalized{suffix}_pred_final_run1.tif")
#     ground_truth_image = os.path.join(ground_truth_path, f"{image_basename}_radd_labelled_agb.tif")
#
#     class_cmap = ListedColormap(['lightblue', 'orange', 'grey'])
#     difference_cmap = ListedColormap(['orange', "lightblue", 'grey'])
#
#     with rasterio.open(input_image) as src:
#         input_rgb = src.read([1, 2, 3])
#         input_rgb = np.clip(input_rgb / 0.1, 0, 1)
#
#     with rasterio.open(output_image) as src:
#         output_data = src.read(1).astype(float)
#         output_data[output_data == -1] = np.nan
#
#     with rasterio.open(ground_truth_image) as src:
#         gt_data = src.read(1).astype(float)
#         gt_data[gt_data == -9999] = np.nan
#
#     # Create the difference map, reversing the color for change and no change
#     difference_map = np.where(np.isnan(output_data) | np.isnan(gt_data), 2, 1 - np.abs(gt_data - output_data))
#
#     fig, axs = plt.subplots(1, 4, figsize=(30, 8), sharex=True, sharey=True)
#     axs[0].imshow(input_rgb.transpose(1, 2, 0))
#     axs[0].set_title('Input RGB')
#     axs[0].axis('on')
#
#     show_class_data(output_data, axs[1], class_cmap, 'Output')
#     show_class_data(gt_data, axs[2], class_cmap, 'Ground Truth')
#
#     # Updated to reverse the colors for the difference map
#     axs[3].imshow(difference_map, cmap=difference_cmap, vmin=0, vmax=2)
#     axs[3].set_title('Difference Map')
#     axs[3].axis('on')
#
#     add_legend(axs[2], class_cmap, ['Non-Disturbed', 'Disturbed', 'No Data'])
#     add_legend(axs[3], difference_cmap, ['Difference', 'No Difference', 'No Data'])
#
#     output_filename = os.path.join(save_path, f"{image_basename}_comparison.png")
#     plt.tight_layout()
#     plt.savefig(output_filename, bbox_inches='tight')
#     plt.close()
#
#     print(f"File saved at {output_filename}")
#
# def show_class_data(data, ax, cmap, title):
#     mapped_data = np.full(data.shape, 2)  # Default to no data
#     mapped_data[data == 0] = 0  # Non-disturbed
#     mapped_data[data == 1] = 1  # Disturbed
#     ax.imshow(mapped_data, cmap=cmap)
#     ax.set_title(title)
#     ax.axis('on')
#
# def add_legend(ax, cmap, labels):
#     patches = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(len(labels))]
#     ax.legend(handles=patches, labels=labels, loc='lower right')
#








#
# def plot_image_triplet(input_path, output_path, ground_truth_path, image_basename, data_type, save_path):
#     if data_type == 'backscatter':
#         suffix = '_bsc_masked_normalized'
#     elif data_type == 'coherence':
#         suffix = '_coh_masked_normalized'
#     else:
#         suffix = ''
#
#     input_image = os.path.join(input_path, f"{image_basename}_sentinel_agb_normalized{suffix}.tif")
#     output_image = os.path.join(output_path, f"{image_basename}_sentinel_agb_normalized{suffix}_pred_final_run1.tif")
#     ground_truth_image = os.path.join(ground_truth_path, f"{image_basename}_radd_labelled_agb.tif")
#
#     with rasterio.open(input_image) as src:
#         input_rgb = src.read([1, 2, 3])
#         input_rgb = np.clip(input_rgb / 0.1, 0, 1)  # Normalize for display
#
#     with rasterio.open(output_image) as src:
#         output_data = src.read(1).astype(float)
#         output_data[output_data == -1] = np.nan  # Set no-data values to NaN
#
#     with rasterio.open(ground_truth_image) as src:
#         gt_data = src.read(1).astype(float)
#         gt_data[gt_data == -9999] = np.nan  # Set no-data values to NaN
#
#     # Define custom colors
#     no_difference_color = 'lightblue'  # Light blue for no change
#     difference_color = 'orange'  # Orange for change
#     nodata_color = 'grey'  # Grey for no data
#
#     difference_map = np.where(np.isnan(output_data) | np.isnan(gt_data), 2, np.abs(gt_data - output_data))
#
#     # Custom colormap for the difference map
#     difference_cmap = ListedColormap([no_difference_color, difference_color, nodata_color])
#
#     fig, axs = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
#     axs[0].imshow(input_rgb.transpose(1, 2, 0))
#     axs[0].set_title('Input RGB')
#     axs[1].imshow(output_data, cmap='gray', vmin=0, vmax=1)
#     axs[1].set_title('Output')
#     axs[2].imshow(gt_data, cmap='gray', vmin=0, vmax=1)
#     axs[2].set_title('Ground Truth')
#     axs[3].imshow(difference_map, cmap=difference_cmap, vmin=0, vmax=2)
#     axs[3].set_title('Difference Map')
#
#     # Add north arrow to the first plot
#     x, y, arrow_length = 0.1, 0.9, 0.1
#     axs[0].annotate('N', xy=(x, y - arrow_length), xytext=(x, y),
#                     arrowprops=dict(facecolor='black', width=5, headwidth=15),
#                     ha='center', va='center', fontsize=20, xycoords=axs[0].transAxes)
#
#     # Legend for the difference map
#     labels = ['No Difference', 'Difference', 'No Data']
#     patches = [plt.Line2D([0], [0], marker='o', color='w', label=labels[i],
#                           markerfacecolor=difference_cmap.colors[i], markersize=15) for i in range(len(labels))]
#     axs[3].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
#
#     plt.tight_layout()
#     output_filename = os.path.join(save_path, f"{image_basename}_comparison.png")
#     plt.savefig(output_filename, bbox_inches='tight')
#     plt.close()
#
#     print(f"File saved at {output_filename}")





#
# def plot_image_triplet(input_path, output_path, ground_truth_path, image_basename, data_type, save_path):
#     suffix = ''
#     if data_type == 'backscatter':
#         suffix = '_bsc_masked_normalized'
#     elif data_type == 'coherence':
#         suffix = '_coh_masked_normalized'
#
#     input_image = os.path.join(input_path, f"{image_basename}_sentinel_agb_normalized{suffix}.tif")
#     output_image = os.path.join(output_path, f"{image_basename}_sentinel_agb_normalized{suffix}_pred_final_run1.tif")
#     ground_truth_image = os.path.join(ground_truth_path, f"{image_basename}_radd_labelled_agb.tif")
#
#     class_cmap = ListedColormap(['lightblue', 'orange', 'grey'])
#     difference_cmap = ListedColormap(['lightblue', 'orange', 'grey'])
#
#     with rasterio.open(input_image) as src:
#         input_rgb = src.read([1, 2, 3])
#         input_rgb = np.clip(input_rgb / 0.1, 0, 1)
#
#     with rasterio.open(output_image) as src:
#         output_data = src.read(1).astype(float)
#         output_data[output_data == -1] = np.nan
#
#     with rasterio.open(ground_truth_image) as src:
#         gt_data = src.read(1).astype(float)
#         gt_data[gt_data == -9999] = np.nan
#
#     difference_map = np.where(np.isnan(output_data) | np.isnan(gt_data), 2, np.abs(gt_data - output_data))
#
#     fig, axs = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
#     axs[0].imshow(input_rgb.transpose(1, 2, 0))
#     axs[0].set_title('Input RGB')
#     axs[0].axis('on')
#
#     show_class_data(output_data, axs[1], class_cmap, 'Output')
#     show_class_data(gt_data, axs[2], class_cmap, 'Ground Truth')
#
#     axs[3].imshow(difference_map, cmap=difference_cmap)
#     axs[3].set_title('Difference Map')
#     axs[3].axis('on')
#
#     add_legend(axs[2], class_cmap, ['Non-Disturbed', 'Disturbed', 'No Data'])
#     add_legend(axs[3], difference_cmap, ['No Change', 'Change', 'No Data'])
#
#     output_filename = os.path.join(save_path, f"{image_basename}_comparison.png")
#     plt.tight_layout()
#     plt.savefig(output_filename, bbox_inches='tight')
#     plt.close()
#
#     print(f"File saved at {output_filename}")
#
# def show_class_data(data, ax, cmap, title):
#     mapped_data = np.full(data.shape, 2)
#     mapped_data[data == 0] = 0
#     mapped_data[data == 1] = 1
#     ax.imshow(mapped_data, cmap=cmap)
#     ax.set_title(title)
#     ax.axis('on')
#
# def add_legend(ax, cmap, labels):
#     patches = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(len(labels))]
#     ax.legend(handles=patches, labels=labels, loc='lower right')
#
#
# def plot_image_triplet(input_path, output_path, ground_truth_path, image_basename, data_type, save_path):
#     # Adjust the suffix based on the data type
#     suffix = ''
#     if data_type == 'backscatter':
#         suffix = '_bsc_masked_normalized'
#     elif data_type == 'coherence':
#         suffix = '_coh_masked_normalized'
#
#     input_image = os.path.join(input_path, f"{image_basename}_sentinel_agb_normalized{suffix}.tif")
#     output_image = os.path.join(output_path, f"{image_basename}_sentinel_agb_normalized{suffix}_pred_final_run1.tif")
#     ground_truth_image = os.path.join(ground_truth_path, f"{image_basename}_radd_labelled_agb.tif")
#
#     # Create a colormap for the output and ground truth data
#     nodata_color = np.array([0.5, 0.5, 0.5])  # Grey color for no data
#     cmap = ListedColormap(['black', 'white', nodata_color])
#
#     with rasterio.open(input_image) as src:
#         input_rgb = src.read([1, 2, 3])
#         input_rgb = np.clip(input_rgb / 0.1, 0, 1)
#         # Set no-data values to grey in the input RGB image
#         nodata_mask = np.all(input_rgb == 0, axis=0)
#         input_rgb[:, nodata_mask] = nodata_color[:, None]
#
#         # Add geographic coordinates
#         extent = rasterio.plot.plotting_extent(src)
#
#     with rasterio.open(output_image) as src:
#         # Ensure the data read is float, to accommodate NaN values
#         output_data = src.read(1).astype(float)  # Convert to float
#         output_data[output_data == -1] = np.nan  # Set no-data values to NaN
#
#
#     with rasterio.open(ground_truth_image) as src:
#         gt_data = src.read(1)
#         gt_data[gt_data == -9999] = np.nan  # Handle no-data values
#
#     # Calculate the difference map
#     difference_map = output_data - gt_data
#
#     fig, axs = plt.subplots(1, 4, figsize=(24, 6))
#     show(input_rgb, ax=axs[0], extent=extent, title='Input RGB')
#     show(output_data, ax=axs[1], cmap=cmap, extent=extent, title='Output')
#     show(gt_data, ax=axs[2], cmap=cmap, extent=extent, title='Ground Truth')
#     im = axs[3].imshow(difference_map, cmap='coolwarm', vmin=-1, vmax=1, extent=extent)
#     axs[3].set_title('Difference Map')
#
#     # Add north arrow to the first plot
#     x, y, arrow_length = 0.1, 0.9, 0.1
#     axs[0].annotate('N', xy=(x, y-arrow_length), xytext=(x, y),
#                     arrowprops=dict(facecolor='black', width=5, headwidth=15),
#                     ha='center', va='center', fontsize=20, xycoords=axs[0].transAxes)
#
#     # Add legend
#     labels = ['Non-Disturbed', 'Disturbed', 'No Data']
#     colors = [cmap.colors[i] for i in range(3)]
#     patches = [plt.Line2D([0], [0], marker='o', color='w', label=labels[i],
#                            markerfacecolor=colors[i], markersize=15) for i in range(len(labels))]
#     axs[3].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
#
#     # Add colorbar for the difference map
#     cbar = fig.colorbar(im, ax=axs[3])
#     cbar.set_label('Difference')
#
#     # Save the output
#     output_filename = os.path.join(save_path, f"{image_basename}_comparison.png")
#     plt.tight_layout()
#     plt.savefig(output_filename, bbox_inches='tight')
#     plt.close()
#
#     print(f"File saved at {output_filename}")

#
#
# def plot_image_triplet( input_path, output_path, ground_truth_path, image_basename, data_type, save_path):
#         if data_type == 'backscatter':
#             suffix = '_bsc_masked_normalized'
#         elif data_type == 'coherence':
#             suffix = '_coh_masked_normalized'
#         else:
#             suffix = ''
#
#         input_image = os.path.join(input_path, f"{image_basename}_sentinel_agb_normalized{suffix}.tif")
#         output_image = os.path.join(output_path, f"{image_basename}_sentinel_agb_normalized{suffix}_pred_final_run1.tif")
#         ground_truth_image = os.path.join(ground_truth_path, f"{image_basename}_radd_labelled_agb.tif")
#
#         # Create a colormap for the output and ground truth data
#         cmap = ListedColormap(['grey', 'black', 'white'])
#
#         with rasterio.open(input_image) as src:
#             input_rgb = src.read([1, 2, 3])
#             input_rgb = np.clip(input_rgb / 0.1, 0, 1)
#             # Set no-data values to grey in the input RGB image
#             nodata_mask = np.all(input_rgb == 0, axis=0)
#             for i in range(3):  # Set no-data regions to grey
#                 input_rgb[i][nodata_mask] = 0.5
#
#             # Add geographic coordinates
#             extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
#
#         with rasterio.open(output_image) as src:
#             output_data = src.read(1).astype(float)  # Convert to float
#             output_data[output_data == -9999] = np.nan  # Assign NaN to no-data values
#
#         with rasterio.open(ground_truth_image) as src:
#             gt_data = src.read(1).astype(float)  # Convert to float
#             gt_data[gt_data == -9999] = np.nan  # Assign NaN to no-data values
#
#         difference_map = np.where(np.isnan(gt_data), np.nan, output_data - gt_data)
#
#         fig, axs = plt.subplots(1, 4, figsize=(24, 6))
#         axs[0].imshow(input_rgb.transpose(1, 2, 0), extent=extent)
#         axs[0].set_title('Input RGB')
#         axs[0].axis('on')  # Keep axis to show the geographic coordinates
#
#         axs[1].imshow(output_data, cmap=cmap, vmin=0, vmax=1, extent=extent)
#         axs[1].set_title('Output')
#         axs[1].axis('on')
#
#         axs[2].imshow(gt_data, cmap=cmap, vmin=0, vmax=1, extent=extent)
#         axs[2].set_title('Ground Truth')
#         axs[2].axis('on')
#
#         im = axs[3].imshow(difference_map, cmap='coolwarm', vmin=-1, vmax=1, extent=extent)
#         axs[3].set_title('Difference Map')
#         axs[3].axis('on')
#
#         # Add north arrow
#         x, y, arrow_length = 0.95, 0.95, 0.1
#         axs[0].annotate('', xy=(x, y), xytext=(x, y - arrow_length),
#                         arrowprops=dict(facecolor='black', width=5, headwidth=15),
#                         ha='center', va='center', fontsize=20, xycoords=axs[0].transAxes)
#
#         # Add colorbar for the difference map
#         fig.colorbar(im, ax=axs[3])
#
#         # Save the output
#         output_filename = os.path.join(save_path, f"{image_basename}_comparison.png")
#         plt.tight_layout()
#         plt.savefig(output_filename, bbox_inches='tight')
#         plt.close()
#
#         print(f"File saved at {output_filename}")

        #
        # def plot_image_triplet(input_path, output_path, ground_truth_path, image_basename, data_type, save_path):
        #     # Adjust the suffix based on the data type
        #     if data_type == 'backscatter':
        #         suffix = '_bsc_masked_normalized'
        #     elif data_type == 'coherence':
        #         suffix = '_coh_masked_normalized'
        #     else:
        #         suffix = ''
        #
        #     input_image = os.path.join(input_path, f"{image_basename}_sentinel_agb_normalized{suffix}.tif")
        #     output_image = os.path.join(output_path, f"{image_basename}_sentinel_agb_normalized{suffix}_pred_final_run1.tif")
        #     ground_truth_image = os.path.join(ground_truth_path, f"{image_basename}_radd_labelled_agb.tif")
        #
        #     # Read the images and prepare the data
        #     with rasterio.open(input_image) as src:
        #         input_rgb = src.read([1, 2, 3])  # Assuming RGB are the first three bands
        #         input_rgb = np.clip(input_rgb / 0.1, 0, 1)  # Normalize values for display
        #
        #     with rasterio.open(output_image) as src:
        #         output_data = src.read(1)
        #
        #     with rasterio.open(ground_truth_image) as src:
        #         gt_data = src.read(1)
        #         gt_data = np.where(gt_data == -9999, np.nan, gt_data)  # Handle no-data values
        #
        #     difference_map = np.where(np.isnan(gt_data), np.nan, output_data - gt_data)
        #
        #     # Plotting the data
        #     fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        #     axs[0].imshow(input_rgb.transpose(1, 2, 0))
        #     axs[0].set_title('Input RGB')
        #     axs[0].axis('off')
        #
        #     axs[1].imshow(output_data, cmap='gray', vmin=0, vmax=1)
        #     axs[1].set_title('Output')
        #     axs[1].axis('off')
        #
        #     axs[2].imshow(gt_data, cmap='gray', vmin=0, vmax=1)
        #     axs[2].set_title('Ground Truth')
        #     axs[2].axis('off')
        #
        #     axs[3].imshow(difference_map, cmap='coolwarm', vmin=-1, vmax=1)
        #     axs[3].set_title('Difference Map')
        #     axs[3].axis('off')
        #
        #     os.makedirs(save_path, exist_ok=True)
        #     output = os.path.join(save_path, f"{image_basename}_comparison.png")
        #     plt.tight_layout()
        #     plt.savefig(output)
        #     plt.close()
        #     print(f"File created: {output} ")


#################
## hls stacks prep
#################

#
# def apply_fmask(self, sentinel_stack_path, fmask_path, output_file):
#     CLOUD_BIT = 1 << 1     # Bit 1 for clouds
#     CLOUD_SHADOW_BIT = 1 << 3  # Bit 3 for cloud shadow
#
#     with rasterio.open(sentinel_stack_path) as sentinel_stack, rasterio.open(fmask_path) as fmask:
#         fmask_data = fmask.read(1)
#
#         # Cloud and cloud shadow masks
#         cloud_mask = (fmask_data & CLOUD_BIT) != 0
#         cloud_shadow_mask = (fmask_data & CLOUD_SHADOW_BIT) != 0
#         combined_mask = cloud_mask | cloud_shadow_mask
#
#         # masked_data = np.empty_like(sentinel_stack.read(), dtype=rasterio.float32)
#         for band in range(sentinel_stack.count):
#             band_data = sentinel_stack.read(band + 1)
#             # masked_band = np.where(combined_mask, -9999, band_data) -9999 is no good.
#             masked_data[band] = np.ma.masked_where(combined_mask, band_data)
#
#             ## now redundant
#             # masked_data[band] = masked_band
#
#         output_profile = sentinel_stack.profile.copy()
#         output_profile.update(dtype=rasterio.float32)#, nodata=-9999)
#         with rasterio.open(output_file, 'w', **output_profile) as dest:
#             dest.write(masked_data)

#
#
#
#
#
# def forest_loss_mask(self, sentinel_stacks, forest_loss_path, output_path, reference_year=2020):
#     for sentinel_file in os.listdir(sentinel_stacks):
#         if sentinel_file.endswith('_agb_radd_stack.tif'):  # Adjust as per your naming convention
#             sentinel_stack_path = os.path.join(sentinel_stacks, sentinel_file)
#             with rasterio.open(sentinel_stack_path) as sentinel_stack:
#                 sentinel_bounds = sentinel_stack.bounds
#                 sentinel_crs = sentinel_stack.crs
#
#                 for forest_loss_file in os.listdir(forest_loss_path):
#                     if forest_loss_file.endswith(".tif"):
#                         forest_loss_file_path = os.path.join(forest_loss_path, forest_loss_file)
#                         with rasterio.open(forest_loss_file_path) as forest_loss:
#                             # if not box(*sentinel_bounds).intersects(box(*forest_loss.bounds)):
#                             #     continue
#
#                             forest_loss_data = np.zeros((1, sentinel_stack.height, sentinel_stack.width), dtype=rasterio.float32)
#                             transform, width, height = calculate_default_transform(
#                                 forest_loss.crs, sentinel_crs, forest_loss.width, forest_loss.height, *forest_loss.bounds,
#                                 dst_width=sentinel_stack.width, dst_height=sentinel_stack.height)
#                             reproject(
#                                 source=rasterio.band(forest_loss, 1),
#                                 destination=forest_loss_data[0],
#                                 src_transform=forest_loss.transform,
#                                 src_crs=forest_loss.crs,
#                                 dst_transform=sentinel_stack.transform,
#                                 dst_crs=sentinel_crs,
#                                 resampling=Resampling.nearest)
#
#                             sentinel_data = sentinel_stack.read()
#
#                             # Check file type and apply mask
#                             if 'treecover' in forest_loss_file:
#                                 mask_condition = forest_loss_data[0] < 80
#                             elif 'lossyear' in forest_loss_file:
#                                 current_year = reference_year - 2000  # Adjust based on your reference year format
#                                 mask_condition = (forest_loss_data[0] > 0) & (forest_loss_data[0] < current_year)
#
#                             sentinel_data[:, mask_condition] = sentinel_stack.nodata
#
#                             output_profile = sentinel_stack.profile.copy()
#                             output_file_name = sentinel_file.replace('_agb_radd_stack.tif', '_agb_radd_forest_stack.tif')
#                             output_file_path = os.path.join(output_path, output_file_name)
#                             with rasterio.open(output_file_path, 'w', **output_profile) as dst:
#                                 dst.write(sentinel_data)
#                             print(f"Masked with {forest_loss_file} and saved to {output_file_name}")
#

#
# def forest_loss_mask(self, sentinel_stack_path, forest_loss_path, output_path, date_threshold):
#     """
#     Mask Sentinel-2 stack with Hansen forest loss data. Mask conditions vary by file type.
#
#     Args:
#         sentinel_stack_path (str): Path to the Sentinel-2 stack file.
#         forest_path (str): Directory containing Hansen forest loss data.
#         output_path (str): Output directory for the masked files.
#         date_threshold (int): Threshold year for the loss_year data.
#     """
#     # Define the CRS for WGS84
#     wgs84_crs = 'EPSG:4326'
#
#     for sentinel_file in os.listdir(sentinel_stack_path):
#         if sentinel_file.endswith('_stack.tif'):
#             sentinel_stack_full_path = os.path.join(sentinel_stack_path, sentinel_file)
#             tile, date = sentinel_file.split('_')[1], sentinel_file.split('_')[2]
#
#             # Identify the corresponding Hansen file
#             for forest_file in os.listdir(forest_loss_path):
#                 if tile in forest_file and ('treecover2000' in forest_file or 'loss_year' in forest_file):
#                     forest_full_path = os.path.join(forest_loss_path, forest_file)
#
#                     with rasterio.open(sentinel_stack_full_path) as sentinel, rasterio.open(forest_full_path) as forest:
#                         # Check if CRS and resolution match, otherwise reproject
#                         if sentinel.crs != forest.crs or sentinel.res != forest.res:
#                             # Perform re-projection as needed (example given in the previous function)
#                             pass
#
#                         # Determine mask condition based on file type
#                         if 'treecover2000' in forest_file:
#                             # Mask treecover2000 data: values < 80 are masked
#                             mask_condition = forest.read(1) < 80
#                         elif 'loss_year' in forest_file:
#                             # Mask loss_year data: mask all years before June 2021
#                             # Assuming loss_year data is encoded as years since 2000
#                             mask_year = forest.read(1) + 2000
#                             mask_condition = mask_year < date_threshold
#                         else:
#                             # Skip if file does not match expected patterns
#                             continue
#
#                         # Apply mask
#                         sentinel_data = sentinel.read()
#                         sentinel_data[:, mask_condition] = 0  # Assuming 0 is the nodata value
#
#                         # Write output
#                         output_file = os.path.join(output_path, sentinel_file.replace('_stack.tif', '_masked.tif'))
#                         with rasterio.open(output_file, 'w', **sentinel.profile) as dst:
#                             dst.write(sentinel_data)
#
#
#
# def forest_loss_mask(self, sentinel_stacks, forest_loss_path, output_path):
#     for sentinel_file in os.listdir(sentinel_stacks):
#         if sentinel_file.endswith('_agb_radd_stack.tif'):  # Assuming you're applying this after AGB merging
#             sentinel_stack_path = os.path.join(sentinel_stacks, sentinel_file)
#             with rasterio.open(sentinel_stack_path) as sentinel_stack:
#                 sentinel_bounds = sentinel_stack.bounds
#                 sentinel_crs = sentinel_stack.crs
#
#                 # Attempt to apply mask with each forest loss file
#                 for forest_loss_file_path in os.listdir(forest_loss_path):
#                     with rasterio.open(os.path.join(forest_loss_path,forest_loss_file_path)) as forest_loss:
#                         # Skip if there's no overlap
#                         if not box(*sentinel_bounds).intersects(box(*forest_loss.bounds)):
#                             continue
#
#                         # Reproject forest loss data to match Sentinel stack CRS and resolution
#                         forest_loss_data = np.zeros((1, sentinel_stack.height, sentinel_stack.width), dtype=rasterio.float32)
#                         transform, width, height = calculate_default_transform(
#                             forest_loss.crs, sentinel_stack.crs, forest_loss.width, forest_loss.height, *forest_loss.bounds,
#                             dst_width=sentinel_stack.width, dst_height=sentinel_stack.height)
#                         reproject(
#                             source=rasterio.band(forest_loss, 1),
#                             destination=forest_loss_data[0],
#                             src_transform=forest_loss.transform,
#                             src_crs=forest_loss.crs,
#                             dst_transform=transform,
#                             dst_crs=sentinel_stack.crs,
#                             resampling=Resampling.nearest)
#
#                         # Apply the mask
#                         sentinel_data = sentinel_stack.read()
#                         mask = forest_loss_data[0] < 80
#                         sentinel_data[:, mask] = sentinel_stack.nodata
#
#                         # Write the masked data to a new file
#                         output_profile = sentinel_stack.profile.copy()
#                         output_file_name = sentinel_file.replace('_with_agb.tif', '_masked.tif')
#                         output_file_path = os.path.join(output_path, output_file_name)
#                         with rasterio.open(output_file_path, 'w', **output_profile) as dst:
#                             dst.write(sentinel_data)
#                         print(f"Masked with {os.path.basename(forest_loss_file_path)} and saved to {output_file_name}")
#                         break  # Stop after successful masking




#################
## dataset management
#################

# def split_dataset(self, ):
#     # Create directories if they don't exist
#     os.makedirs(self.train_dir, exist_ok=True)
#     os.makedirs(self.val_dir, exist_ok=True)
#     os.makedirs(self.test_dir, exist_ok=True)
#
#     # Step 1: Identify and move unique tiles to the test set
#     unique_files = self.find_unique_tiles()
#     self.move_files(unique_files, self.test_dir)
#
#     # Step 2: Split remaining files into train and validation sets
#     remaining_files = [file for file in os.listdir(self.source_dir) if file.endswith('.tif')]
#     train_files, val_files = train_test_split(remaining_files, test_size=self.val_split, random_state=42)
#
#     # Move files to their respective directories
#     self.move_files(train_files, self.train_dir)
#     self.move_files(val_files, self.val_dir)
#


# def split_dataset(self, folder_path, val_split=0.2):
#     train_dir = os.path.join(folder_path, 'train')
#     val_dir = os.path.join(folder_path, 'val')
#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(val_dir, exist_ok=True)
#
#     # Filter files
#     files = [f for f in os.listdir(folder_path) if f.endswith('.tif') and ('_radd' in f or '_sentinel' in f)]
#
#     # Group files by common identifier
#     grouped_files = {}
#     for file in files:
#         identifier = "_".join(file.split('_')[:-1])  # Adjust based on your naming convention
#         grouped_files.setdefault(identifier, []).append(file)
#
#     file_groups = list(grouped_files.values())  # List of file groups
#
#     # Split file groups into train and val
#     total_size = len(file_groups)
#     indices = torch.randperm(total_size).tolist()
#     split_idx = int(total_size * val_split)
#
#     train_groups = [file_groups[i] for i in indices[split_idx:]]
#     val_groups = [file_groups[i] for i in indices[:split_idx]]
#
#     # Move files in each group to their respective directories
#     for group in train_groups:
#         for f in group:
#             shutil.move(os.path.join(folder_path, f), train_dir)
#     for group in val_groups:
#         for f in group:
#             shutil.move(os.path.join(folder_path, f), val_dir)
#

#
# def split_dataset(self, folder_path, val_split=0.2):
#     train_dir = os.path.join(folder_path, 'train')
#     val_dir = os.path.join(folder_path, 'val')
#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(val_dir, exist_ok=True)
#
#     # Filter files
#     files = [f for f in os.listdir(folder_path) if f.endswith('.tif') and ('_radd' in f or '_sentinel' in f)]
#
#     # Split into train and val
#     total_size = len(files)
#     indices = torch.randperm(total_size).tolist()
#     split_idx = int(total_size * val_split)
#
#     train_files = [files[i] for i in indices[split_idx:]]
#     val_files = [files[i] for i in indices[:split_idx]]
#
#     # Move files
#     for f in train_files:
#         shutil.move(os.path.join(folder_path, f), train_dir)
#     for f in val_files:
#         shutil.move(os.path.join(folder_path, f), val_dir)


#
# def split_dataset(self, source_dir, pairs):
#     os.makedirs(self.train_dir, exist_ok=True)
#     os.makedirs(self.val_dir, exist_ok=True)
#
#     dataset = CustomDataset(source_dir, self.pairs)
#     total_size = len(dataset)
#     val_size = int(total_size * self.val_split)
#     train_size = total_size - val_size
#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#
#     # Move files to their respective directories
#     for i in train_dataset.indices:
#         sentinel_file, radd_file = dataset[i]
#         if not os.path.exists(os.path.join(self.train_dir, os.path.basename(sentinel_file))):
#             shutil.move(sentinel_file, self.train_dir)
#         if not os.path.exists(os.path.join(self.train_dir, os.path.basename(radd_file))):
#             shutil.move(radd_file, self.train_dir)
#
#     for i in val_dataset.indices:
#         sentinel_file, radd_file = dataset[i]
#         if not os.path.exists(os.path.join(self.val_dir, os.path.basename(sentinel_file))):
#             shutil.move(sentinel_file, self.val_dir)
#         if not os.path.exists(os.path.join(self.val_dir, os.path.basename(radd_file))):
#             shutil.move(radd_file, self.val_dir)


#################
## test run analysis
#################


# The test_run_analysis script
# import os
# import glob
# from src.test_analysis import TestAnalysis
# import pandas as pd
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
#
# # Directory containing all log directories
# base_log_dir = r"E:\PycharmProjects\hls-foundation-os\Prithvi-100m\test_logs"
#
# # Initialize a list to hold all the metrics data
# all_metrics = []
#
# # Iterate over each model's log directory
# for model_dir in os.listdir(base_log_dir):
#     model_log_dir = os.path.join(base_log_dir, model_dir)
#     if os.path.isdir(model_log_dir):
#         # Find the JSON file in the model's log directory
#         json_files = glob.glob(os.path.join(model_log_dir, '*.json'))
#         for json_file in json_files:
#             print(f"Analyzing {json_file}")
#             analysis = TestAnalysis(json_file)
#             metrics = analysis.get_average_metrics()
#             metrics['model'] = model_dir  # Add the model name to the metrics
#             all_metrics.append(metrics)
#
# # Create a DataFrame from the collected metrics data
# df = pd.DataFrame(all_metrics)
#
# # Set the model as the index of the DataFrame
# df.set_index('model', inplace=True)
#
# # Display the table
# print(df)
#
# # Save the DataFrame to CSV
# csv_file = os.path.join(base_log_dir, 'model_metrics_test.csv')
# df.to_csv(csv_file)
# print(f"Metrics saved to {csv_file}")







# # The test_run_analysis script
# import os
# import glob
# from src.test_analysis import TestAnalysis
# import pandas as pd
# import re
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
#
# def get_base_model_name(dir_name):
#     # Ensure we only process directories that start with 'best'
#     if not dir_name.startswith('best'):
#         return None
#
#     # Split the directory name and keep only relevant parts
#     # We want to keep the model type and minalerts value but ignore the specific run number
#     parts = dir_name.split('_')
#     # Exclude parts that are purely numeric or start with "run"
#     # Keep the minalerts value by checking for its presence
#     base_parts = [part for part in parts if not part.isnumeric() and not part.startswith('run') or 'minalerts' in part]
#     base_model_name = '_'.join(base_parts)
#     return base_model_name
#
# # Directory containing all log directories
# base_log_dir = r"E:\PycharmProjects\hls-foundation-os\Prithvi-100m\test_logs"
#
# # Initialize a dictionary to group log files by their base model names
# model_logs = {}
#
# # Gather all JSON files and group them by their base model name
# for model_dir in os.listdir(base_log_dir):
#     model_log_dir = os.path.join(base_log_dir, model_dir)
#     if os.path.isdir(model_log_dir) and model_dir.startswith('best'):
#         base_model_name = get_base_model_name(model_dir)
#         if base_model_name:
#             json_files = glob.glob(os.path.join(model_log_dir, '*.json'))
#             if base_model_name in model_logs:
#                 model_logs[base_model_name].extend(json_files)
#             else:
#                 model_logs[base_model_name] = json_files
# # Initialize a list to hold all the metrics data
# all_metrics = []
#
# # Analyze the metrics for each base model name
# for base_model, log_files in model_logs.items():
#     print(f"Analyzing {base_model}")
#     analysis = TestAnalysis(log_files)
#     metrics = analysis.get_average_metrics()
#     if metrics:  # Ensure there is data to add
#         metrics['model'] = base_model  # Add the base model name to the metrics
#         all_metrics.append(metrics)
#
# # Create a DataFrame from the collected metrics data
# df = pd.DataFrame(all_metrics)
#
# # Set the model as the index of the DataFrame
# df.set_index('model', inplace=True)
#
# # Display the table
# print(df)
#
# # Save the DataFrame to CSV
# csv_file = os.path.join(base_log_dir, 'model_metrics_test.csv')
# df.to_csv(csv_file)
# print(f"Metrics saved to {csv_file}")



#################
## model run analysis
#################


# # run_model_analysis.py
#
# import matplotlib.pyplot as plt
# from src.model_analysis import ModelAnalysis
#
# # Paths to your log files
# log_files = {
#     # "Prithvi_no_finetune": "path/to/prithvi_no_finetune.log.json",
#     "Prithvi_finetune": "E:\PycharmProjects\hls-foundation-os\Prithvi-100m/best_mIoU_iter_1000_minalerts_10000_prithvi.log.json",
#     "UNet": "E:\PycharmProjects\hls-foundation-os\Prithvi-100m/best_mIoU_iter_1000_minalerts_10000_unet.log.json",
#
# }
#
# # Initialize ModelAnalysis objects
# analyses = {name: ModelAnalysis(log_file) for name, log_file in log_files.items()}
#
# # Plot metrics
#
# metrics = ["aAcc", "mIoU", "mAcc", "IoU.Forest", "IoU.Disturbed_Forest", "Acc.Forest", "Acc.Disturbed_Forest"]
# for metric in metrics:
#     fig, ax = plt.subplots()
#     for name, analysis in analyses.items():
#         analysis.plot_metric(metric, ax, label=name)
#
#     ax.set_ylim([0, 1])
#     plt.title(metric)
#     plt.savefig(f'E:\Data\Results\Prithvi_model_analysis_images/{metric}_comparison.png')
#     plt.close()
#




#################
## inference run analysis
#################

# import matplotlib.pyplot as plt
# import os
# from src.model_analysis import plot_image_triplet
#
# model_config_pairs = [
#     ("Prithvi-100m_backscatter/best_mIoU_iter_1000_minalerts_15000_prithvi_backscatter_final_run1.pth", "forest_disturbances_config_backscatter.py"),
#     ("Prithvi-100m_coherence/best_mIoU_iter_1000_minalerts_15000_prithvi_coherence_final_run1.pth", "forest_disturbances_config_coherence.py"),
#     ("Prithvi-100m_burnscars/best_mIoU_iter_1000_minalerts_15000_prithvi_burnscars_final_run1.pth", "forest_disturbances_config_burnscars.py"),
#     ("Prithvi-100m/best_mIoU_iter_400_minalerts_15000_prithvi_final_run1.pth","forest_disturbances_config.py"),
#
# ]
#
# image_basenames = [
#     "2023290_T50MKE_agb_radd_fmask_stack_1024_3072",
#     "2023276_T49MDU_agb_radd_fmask_stack_2048_1024",
#     "2023271_T49MDU_agb_radd_fmask_stack_2048_1024",
#     "2023245_T50MKE_agb_radd_fmask_stack_2048_512",
#     "2023241_T49MDU_agb_radd_fmask_stack_2048_1024",
#     "2023111_T49MET_agb_radd_fmask_stack_512_512",
#     "2023076_T49MET_agb_radd_fmask_stack_512_512"
# ]
#
# # Directory where you want to save the combined images
# save_path = r"E:\PycharmProjects\hls-foundation-os\Prithvi-100m\test_logs\test_image_results\comparisons"
# os.makedirs(save_path, exist_ok=True)
#
#
# for image_basename in image_basenames:
#     plot_image_triplet(
#     input_path=r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\globalnorm\10000_minalerts\test",
#     output_path=r"E:\PycharmProjects\hls-foundation-os\Prithvi-100m\test_logs\test_image_results\best_mIoU_iter_1000_minalerts_15000_prithvi",
#     ground_truth_path=r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\globalnorm\10000_minalerts\test",
#     image_basename=image_basename,
#     save_path=save_path
#
#     )


#################
## run test command unet
#################
# from mim.commands.test import test
# import os
#
# # Ensure CUDA operations are executed synchronously if necessary
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#
# # Define the parameters for testing
# package_name = 'mmsegmentation'
# config_path = 'E:\hls-foundation-os\configs/forest_distrubances_config.py'
# ckpt_path = r"E:\PycharmProjects\hls-foundation-os\Prithvi-100m\best_mIoU_iter_1000.pth"#"E:/burn_scars_Prithvi_100M.pth"
#
# # Specify the metrics to evaluate
# metrics = ['mIoU']
#
# # Call the test function
# success, message = test(package=package_name,
#                         config=config_path,
#                         checkpoint=ckpt_path)
#                         # ,eval=metrics)
#
# # Check the result
# if success:
#     print("Testing completed successfully.")
# else:
#     print(f"Testing failed: {message}")
#



# import subprocess
# import json
#
# # Define the paths to your config and checkpoint files
# config_path = r"E:\hls-foundation-os\configs\unet_forest_disturbance_config.py"
# checkpoint_path = r"E:\PycharmProjects\hls-foundation-os\Prithvi-100m\best_mIoU_iter_1000_minalerts_10000_unet.pth"
#
# # Define the command to run the mmseg testing script directly
# command = [
#     'python',
#     r'c:\anaconda3\envs\prithvi-100m\lib\site-packages\mmseg\.mim\tools\test.py',
#     config_path,
#     checkpoint_path,
#     '--eval', 'mIoU',
#     '--launcher', 'none'  # Assuming you are running this in a non-distributed manner
# ]
#
# # Execute the command
# subprocess.run(command)
#
#
# # Execute the command and capture the output
# result = subprocess.run(command, capture_output=True, text=True)
#
# # Process the result if needed and save to JSON
# test_results = result.stdout  # or result.stderr based on where the output is
#
# # Assuming test_results is already in JSON format or can be converted into a dictionary
# try:
#     test_results_dict = json.loads(test_results)
# except json.JSONDecodeError:
#     # Handle possible JSON decode error, maybe the output is not in JSON format
#     print("Error decoding JSON from the test output.")
#     test_results_dict = {}
#
# # Define the path to the JSON file where you want to save the results
# results_path = "test_results.json"
#
# # Append the results to the JSON file
# with open(results_path, 'a') as json_file:
#     json.dump(test_results_dict, json_file)
#     json_file.write('\n')  # Add a newline to separate entries



#################
## run test command prithvi
#################
# import subprocess
# import os
#
# models = [
#     "best_mIoU_iter_400_minalerts_15000_prithvi_final_run1.pth",
#     "best_mIoU_iter_300_minalerts_15000_prithvi_final_run2.pth",
#     "best_mIoU_iter_500_minalerts_15000_prithvi_final_run3.pth",
#     "best_mIoU_iter_400_minalerts_15000_prithvi_burnscars_final_run1.pth",
# "best_mIoU_iter_1000_minalerts_15000_prithvi_backscatter_final_run1.pth",
# "best_mIoU_iter_1000_minalerts_15000_prithvi_coherence_final_run1.pth"
# ]
#
# base_ckpt_path = r"E:\PycharmProjects\hls-foundation-os\Prithvi-100m"
# base_log_dir = r"E:\PycharmProjects\hls-foundation-os\Prithvi-100m\test_logs"
# config_path = r"E:\hls-foundation-os\configs\forest_disturbances_config.py"
# output_images_path = r"E:\PycharmProjects\hls-foundation-os\Prithvi-100m\test_logs\test_image_results"  # Specify your output image directory
#
#
# for model in models:
#     # Construct checkpoint path
#     checkpoint_path = os.path.join(base_ckpt_path, model)
#
#     # Create a specific log directory for each model
#     model_log_dir = os.path.join(base_log_dir, os.path.splitext(model)[0])
#     if not os.path.exists(model_log_dir):
#         os.makedirs(model_log_dir)
#
#     # Define the command to run the mmseg testing script for the current model
#     command = [
#         'python',
#         r'c:\anaconda3\envs\prithvi-100m\lib\site-packages\mmseg\.mim\tools\test.py',
#         config_path,
#         checkpoint_path,
#         '--out', os.path.join(model_log_dir, 'results.pkl'),
#         '--eval', 'mIoU',
#         # '--show-dir', output_images_path,  # No need, infere3nce does this for us. Directory to save visualization results
#
#         '--work-dir', model_log_dir,
#         '--launcher', 'none'
#     ]
#
#     # Execute the command
#     print(f"Testing with model: {model}")
#     subprocess.run(command)
#     print(f"Finished testing with model: {model}\n")
#


#################
## run inference command
#################
# import subprocess
#
# # Define command parameters
# config_path = "E:\hls-foundation-os\configs/unet_forest_disturbance_config.py" #forest_distrubances_config.py" #burn_scars_config.py"
# ckpt_path = r"E:\PycharmProjects\hls-foundation-os\Prithvi-100m\best_mIoU_iter_1000_minalerts_12500_unet.pth" #"E:/burn_scars_Prithvi_100M.pth"
# input_path = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\globalnorm\10000_minalerts\test" #r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\inference_test\best_sample/"
#
# output_path = r"E:\PycharmProjects\hls-foundation-os\Prithvi-100m\test_logs\test_image_results" #r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\inference_test"
# input_type = "tif"
# bands = "[0,1,2,3,4,5]"
# out_channels = 1
#
# # Construct the command
# command = [
#     'python', 'model_inference.py',
#     '-config', config_path,
#     '-ckpt', ckpt_path,
#     '-input', input_path,
#     '-output', output_path,
#     '-input_type', input_type,
#     '-bands', bands,
#     # '-out_channels', str(out_channels)
# ]
#
# # Run the command
# subprocess.run(command)


# import subprocess
# import os
#
# # List of model checkpoint filenames
# models = [
# "/Prithvi-100m_backscatter/best_mIoU_iter_1000_minalerts_15000_prithvi_backscatter_final_run1.pth",
# "/Prithvi-100m_coherence/best_mIoU_iter_1000_minalerts_15000_prithvi_coherence_final_run1.pth",
# "/Prithvi-100m_burnscars/best_mIoU_iter_1000_minalerts_15000_prithvi_coherence_final_run1.pth"
# ]
#
# base_ckpt_path = r"E:\PycharmProjects\hls-foundation-os"
# base_output_path = r"E:\PycharmProjects\hls-foundation-os\Prithvi-100m\test_logs\test_image_results"
#
#
#
# ## CANT RUN UNET AND PRITHVI CONFIGS AT THE SAME TIME
# config_path = r"E:\hls-foundation-os\configs/forest_disturbances_config.py"
# config_path_burnscars = r"E:\hls-foundation-os\configs/forest_disturbances_config_burnscars.py"
# config_path_coherence = r"E:\hls-foundation-os\configs/forest_disturbances_config_coherence.py"
# config_path_backscatter = r"E:\hls-foundation-os\configs/forest_disturbances_config_backscatter.py"
#
# input_path = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\globalnorm\15000_minalerts\test/"
# input_type = "tif"
# bands = "[0,1,2,3,4,5]"
# out_channels = 1
#
# for model in models:
#     # Construct checkpoint path
#     ckpt_path = os.path.join(base_ckpt_path, model)
#
#     # Create a specific output directory for each model
#     model_output_path = os.path.join(base_output_path, os.path.splitext(model)[0])
#     if not os.path.exists(model_output_path):
#         os.makedirs(model_output_path)
#
#     # Construct the command
#     command = [
#         'python', 'model_inference.py',
#         '-config', config_path,
#         '-ckpt', ckpt_path,
#         '-input', input_path,
#         '-output', model_output_path,
#         '-input_type', input_type,
#         '-bands', bands,
#         # '-out_channels', str(out_channels)
#     ]
#
#     # Run the command
#     subprocess.run(command)



# import subprocess
# import os
#
# # Model checkpoints and their corresponding configuration paths
# model_config_pairs = [
#     ("Prithvi-100m/best_mIoU_iter_400_minalerts_15000_prithvi_final_run1.pth","forest_disturbances_config.py"),
#     # ("Prithvi-100m_backscatter/best_mIoU_iter_1000_minalerts_15000_prithvi_backscatter_final_run1.pth", "forest_disturbances_config_backscatter.py"),
#     # ("Prithvi-100m_coherence/best_mIoU_iter_1000_minalerts_15000_prithvi_coherence_final_run1.pth", "forest_disturbances_config_coherence.py"),
#     # ("Prithvi-100m_burnscars/best_mIoU_iter_400_minalerts_15000_prithvi_birnscars_final_run1.pth", "forest_disturbances_config_burnscars.py"),
# ]
#
# base_ckpt_path = r"E:\PycharmProjects\hls-foundation-os"
# base_config_path = r"E:\hls-foundation-os\configs"
# base_output_path = r"E:\PycharmProjects\hls-foundation-os\Prithvi-100m\test_logs\test_image_results"
#
# input_path = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\globalnorm\15000_minalerts\test/"
# input_type = "tif"
# bands = "[0,1,2,3,4,5]"
#
# for model, config in model_config_pairs:
#     # Construct checkpoint and config paths
#     ckpt_path = os.path.join(base_ckpt_path, model)
#     config_path = os.path.join(base_config_path, config)
#
#     # Create a specific output directory for each model
#     model_output_path = os.path.join(base_output_path, os.path.splitext(model)[0])
#     if not os.path.exists(model_output_path):
#         os.makedirs(model_output_path)
#
#     # Construct the command
#     command = [
#         'python', 'model_inference.py',
#         '-config', config_path,
#         '-ckpt', ckpt_path,
#         '-input', input_path,
#         '-output', model_output_path,
#         '-input_type', input_type,
#         '-bands', bands,
#     ]
#
#     # Run the command
#     subprocess.run(command)




#################
## run condat command
#################
# import subprocess
# import os
# import mim
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# import torch
# # Command and arguments
# # command = "mim"
# # args = ["train", "mmsegmentation", "--launcher", "none", "--gpus", "1", "E:\hls-foundation-os\configs/forest_distrubances_config.py"]
#
# # Running the command
# # subprocess.run([command] + args)
#
#
#
# from mim.commands.train import train
#
# # Define the parameters for training
# package_name = 'mmsegmentation'
# config_path = "E:\hls-foundation-os\configs/forest_disturbances_config.py"
# num_gpus = 1
#
# # Call the train function
# success, message = train(package=package_name, config=config_path, gpus=num_gpus)
#
# # Check the result
# if success:
#     print("Training completed successfully.")
# else:
#     print(f"Training failed: {message}")
#
# #from mim.commands.test import t














#
# def find_corresponding_files(self, tile_id):
#     sen2_files = [os.path.join(self.sen2_stack_path, filename) for filename in os.listdir(self.sen2_stack_path)
#                   if tile_id in filename]#and filename.endswith('_agb_radd_stack.tif')]
#
#     # Find all combined SAR files in the output directory corresponding to the tile
#     sar_files = [os.path.join(self.output_path, filename) for filename in os.listdir(self.output_path)if tile_id in self.sar_data_path and filename.endswith('.tif')]
#
#     # Match Sentinel-2 files with the closest previous combined SAR file
#     matched_files = []
#     for sen2_file in sen2_files:
#         closest_sar_files = self.find_closest_sar_file(os.path.basename(sen2_file), sar_files)
#         for sar_file in closest_sar_files:
#             with rasterio.open(sen2_file) as sen2, rasterio.open(sar_file) as sar:
#                 transformer_to_wgs84_sen2 = Transformer.from_crs(sen2.crs, 'EPSG:4326', always_xy=True)
#                 transformer_to_wgs84_sar = Transformer.from_crs(sar.crs, 'EPSG:4326', always_xy=True)
#                 sen2_bounds_wgs84 = shapely_transform(transformer_to_wgs84_sen2.transform, box(*sen2.bounds))
#                 sar_bounds_wgs84 = shapely_transform(transformer_to_wgs84_sar.transform, box(*sar.bounds))
#                 if sen2_bounds_wgs84.intersects(sar_bounds_wgs84):
#                     matched_files.append((sen2_file, sar_file))
#                     break  # Stop looking for matches once an overlap is found
#                 else:
#                     print(f"No geographic overlap for {os.path.basename(sen2_file)} in provided SAR files.")
#
#     return matched_files


# def find_closest_sar_file(self, sen2_file, sar_files):
#     # Extract the date from the Sentinel-2 filename
#     sen2_date_str = sen2_file.split('_')[0]  # Adjust this index based on your filename format
#     year = int(sen2_date_str[:4])
#     day_of_year = int(sen2_date_str[4:])
#     sen2_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
#
#     closest_sar_file = None
#     min_date_diff = None
#
#     if self.data_type == 'coherence':
#         pattern = r'coherence_window_28_IW\d_burst_\d_\d\.tif$'
#     elif self.data_type == 'backscatter':
#         pattern = r'backscatter_multilook_window_28_IW\d_burst_\d_\d\.tif$'
#
#     burst_pattern = re.compile(pattern)
#
#     for sar_file in sar_files:
#         if burst_pattern.search(sar_file):
#             # Extract and parse the date from the combined SAR filename
#             sar_date_str = sar_file.split('_')[-12]
#             sar_date = datetime.strptime(sar_date_str, '%Y%m%d')
#
#             if sar_date <= sen2_date:
#                 date_diff = (sen2_date - sar_date).days
#                 if min_date_diff is None or date_diff < min_date_diff:
#                     min_date_diff = date_diff
#                     closest_sar_file = sar_file
#         else:
#             continue
#
#     return closest_sar_file
#
# def find_corresponding_files(self, tile_id):
#     # Find the Sentinel-2 files for the given tile
#     sen2_files = [os.path.join(self.sen2_stack_path, filename) for filename in os.listdir(self.sen2_stack_path)
#                   if tile_id in filename]#and filename.endswith('_agb_radd_stack.tif')]
#
#     # Find all combined SAR files in the output directory corresponding to the tile
#     sar_files = [os.path.join(self.output_path, filename) for filename in os.listdir(self.output_path)if tile_id in self.sar_data_path and filename.endswith('.tif')]
#
#     # Match Sentinel-2 files with the closest previous combined SAR file
#     matched_files = []
#     for sen2_file in sen2_files:
#         closest_sar_file = self.find_closest_sar_file(os.path.basename(sen2_file), sar_files)
#         for sar_file in closest_sar_files:
#             # Check for geographic overlap
#             with rasterio.open(sen2_file) as sen2, rasterio.open(closest_sar_file) as sar:
#                 # Create transformers to transform coordinates to WGS84
#                 transformer_to_wgs84_sen2 = Transformer.from_crs(sen2.crs, 'EPSG:4326', always_xy=True)
#                 transformer_to_wgs84_sar = Transformer.from_crs(sar.crs, 'EPSG:4326', always_xy=True)
#
#                 # Transform bounds to WGS84
#                 sen2_bounds_wgs84 = shapely_transform(transformer_to_wgs84_sen2.transform, box(*sen2.bounds))
#                 sar_bounds_wgs84 = shapely_transform(transformer_to_wgs84_sar.transform, box(*sar.bounds))
#
#                 # Check for intersection in WGS84
#                 if sen2_bounds_wgs84.intersects(sar_bounds_wgs84):
#                     matched_files.append((sen2_file, closest_sar_file))
#                 else:
#                     print(f"No geographic overlap between {os.path.basename(sen2_file)} and {os.path.basename(closest_sar_file)}.")
#
#     return matched_files

#
# def find_corresponding_files(self, tile_id):
#     # Find the Sentinel-2 files for the given tile
#     sen2_files = [os.path.join(self.sen2_stack_path, filename) for filename in os.listdir(self.sen2_stack_path)
#                   if tile_id in filename and filename.endswith('_agb_radd_stack.tif')]
#
#     # Find all combined SAR files in the output directory corresponding to the tile
#     sar_files = [os.path.join(self.output_path, filename) for filename in os.listdir(self.output_path)if tile_id in self.sar_data_path and filename.endswith('.tif')]
#
#     # Match Sentinel-2 files with the closest previous combined SAR file
#     matched_files = []
#     for sen2_file in sen2_files:
#         closest_sar_file = self.find_closest_sar_file(os.path.basename(sen2_file), sar_files)
#         if closest_sar_file:
#             if self.check_overlap(sen2_file, closest_sar_file):
#                 matched_files.append((sen2_file, closest_sar_file))
#
#     return matched_files

# def check_overlap(self, sen2_file, sar_file):
#     with rasterio.open(sen2_file) as sen2, rasterio.open(sar_file) as sar:
#         # Get bounding boxes
#         sen2_bounds = sen2.bounds
#         sar_bounds = sar.bounds
#
#         # Check for overlap
#         overlap = not (sar_bounds.right < sen2_bounds.left or
#                        sar_bounds.left > sen2_bounds.right or
#                        sar_bounds.bottom > sen2_bounds.top or
#                        sar_bounds.top < sen2_bounds.bottom)
#
#         return overlap
#
#

# def overlay_sar_on_sen2(self, sen2_file_path, sar_file_path):
#     output_path = os.path.join(self.output_path, os.path.basename(sen2_file_path).replace('.tif', '_sar_overlay.tif'))
#
#     with rasterio.open(sen2_file_path) as sen2_dataset:
#         sen2_profile = sen2_dataset.profile
#         sen2_profile.update(count=sen2_dataset.count)  # Keep the original number of bands
#
#     # Read the SAR data
#     with rasterio.open(sar_file_path) as sar_dataset:
#         sar_data_vv = sar_dataset.read(1)  # First band (VV)
#         sar_data_vh = sar_dataset.read(2)  # Second band (VH)
#         sar_transform = sar_dataset.transform
#         sar_crs = sar_dataset.crs
#
#     # Write the data into the Sentinel-2 raster
#     with rasterio.open(output_path, 'w', **sen2_profile) as dst:
#         # Write the original bands from the Sentinel-2 dataset
#         for band_index in range(1, sen2_profile['count'] + 1):
#             if band_index == 6:
#                 # Replace with VV band data
#                 dst.write(sar_data_vv, 6)
#             elif band_index == 7:
#                 # Replace with VH band data
#                 dst.write(sar_data_vh, 7)
#             else:
#                 # Copy the original Sentinel-2 band data
#                 with rasterio.open(sen2_file_path) as sen2_dataset:
#                     dst.write(sen2_dataset.read(band_index), band_index)
#
#     print(f"SAR data overlaid on Sentinel-2 bands 6 and 7 at {output_path}")
#     return output_path
#
#
# def create_blank_stack_with_sar_bands(self, sen2_file_path, sar_file_path, output_path):
#     with rasterio.open(sen2_file_path) as sen2_dataset:
#         sen2_profile = sen2_dataset.profile
#         sen2_profile.update(dtype=rasterio.float32, nodata=-9999)  # Define no-data value and data type
#
#         # Create a blank array for each band
#         blank_stack = np.full((sen2_profile['count'], sen2_profile['height'], sen2_profile['width']), sen2_profile['nodata'], dtype=sen2_profile['dtype'])
#
#     # Read SAR data
#     with rasterio.open(sar_file_path) as sar_dataset:
#         sar_data_vv = sar_dataset.read(1)  # First band (VV)
#         sar_data_vh = sar_dataset.read(2)  # Second band (VH)
#
#     # Assign SAR data to the 6th and 7th bands
#     blank_stack[5, :, :] = sar_data_vv
#     blank_stack[6, :, :] = sar_data_vh
#
#     # Write the blank stack to the new file
#     with rasterio.open(output_path, 'w', **sen2_profile) as dst:
#         dst.write(blank_stack)
#
#     print(f"Blank stack with SAR data in bands 6 and 7 created at {output_path}")
#     return output_path
#

# def create_buffered_sar_image(self, sen2_dataset_path, sar_dataset_path, output_path):
#     with rasterio.open(sen2_dataset_path) as sen2_dataset:
#         # Get Sentinel-2 metadata and dimensions
#         sen2_meta = sen2_dataset.meta
#         sen2_height = sen2_dataset.height
#         sen2_width = sen2_dataset.width
#
#     with rasterio.open(sar_dataset_path) as sar_dataset:
#         # Get SAR data
#         sar_data_vv = sar_dataset.read(1)  # Assuming VV is the first band
#         sar_data_vh = sar_dataset.read(2)  # Assuming VH is the second band
#         sar_transform = sar_dataset.transform
#
#     # Create an empty array with the dimensions of the Sentinel-2 data
#     no_data_value = -9999
#     buffered_sar_vv = np.full((sen2_height, sen2_width), no_data_value, dtype=np.float32)
#     buffered_sar_vh = np.full((sen2_height, sen2_width), no_data_value, dtype=np.float32)
#
#     # Calculate the position of the SAR data within the Sentinel-2 array
#     sar_bounds = rasterio.transform.array_bounds(sar_dataset.height, sar_dataset.width, sar_transform)
#     sar_window = sen2_dataset.window(*sar_bounds)
#     row_off = int(sar_window.row_off)
#     col_off = int(sar_window.col_off)
#
#     # Place SAR data into the empty array at the correct position
#     buffered_sar_vv[row_off:row_off + sar_data_vv.shape[0], col_off:col_off + sar_data_vv.shape[1]] = sar_data_vv
#     buffered_sar_vh[row_off:row_off + sar_data_vh.shape[0], col_off:col_off + sar_data_vh.shape[1]] = sar_data_vh
#
#     # Update metadata for the new dataset
#     out_meta = sen2_meta.copy()
#     out_meta.update({'count': 2, 'dtype': 'float32', 'nodata': no_data_value})
#
#     # Write the new dataset with the buffered SAR data
#     with rasterio.open(output_path, 'w', **out_meta) as dst:
#         dst.write(buffered_sar_vv, 1)
#         dst.write(buffered_sar_vh, 2)
#
#     print(f"Buffered SAR image saved to {output_path}")


#

# def filter_stacks_and_radd_by_AGB(self, input_folder, output_folder, check_intersection_file=None, valid_classes=[2, 3, 4, 5], nodata_value=-9999):
#
#     """
#     Filters Sentinel stacks and corresponding RADD alert files by AGB land classification.
#     # 1 = Intact Lowland Forest
#     # 2 = Intact Montane Forest
#     # 3 = Secondary and Degraded Forest
#     # 4 = Peat Swamp Forest
#     # 5 = Mangrove Forest
#     # 6 = Swamp Scrublands
#     # 7 = Crops/Agriculture
#     # 8 = Tree Plantations
#     # 9 = Urban/Settlement
#     # 10 = Scrublands
#     # 11 = Inland Water
#     Args:
#         input_folder (str): Path to the folder containing Sentinel and RADD files.
#         output_folder (str): Path to the folder where filtered files will be saved.
#         valid_classes (list): AGB class values to retain.
#         nodata_value (int): No-data value for filtered pixels.
#     """
#
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#
#     # intersection_file = set(os.listdir(check_intersection_file))
#
#     for filename in os.listdir(input_folder):
#         if filename.endswith('modified_sentinel.tif'):
#             # Construct both sentinel and radd filenames to check if they exist in training_narrow_folder
#             sentinel_check_name = filename.replace('_sentinel_normalized.tif',"_sentinel_normalized_agb.tif")
#             radd_check_name = filename.replace('_sentinel_normalized.tif', '_radd_agb.tif')
#
#             # Skip processing if either file exists in the training_narrow folder
#             # if sentinel_check_name in intersection_file or radd_check_name in intersection_file:
#             #     print(f"Skipping {filename} as its Sentinel or RADD version exists in training_narrow folder.")
#             #     continue
#
#             sentinel_file_path = os.path.join(input_folder, filename)
#             radd_filename = filename.replace('_modified_sentinel.tif', '_modified_radd.tif')
#             radd_file_path = os.path.join(input_folder, radd_filename)
#
#             # Process the Sentinel file
#             with rasterio.open(sentinel_file_path) as sentinel_src:
#                 agb_band = sentinel_src.read(7)  # Assuming AGB band is the 7th band
#                 agb_mask = np.isin(agb_band, valid_classes)
#
#                 output_sentinel_filename = f"{filename.replace('.tif', '')}_agb.tif"
#                 output_sentinel_path = os.path.join(output_folder, output_sentinel_filename)
#                 self.apply_mask_and_save(sentinel_src, agb_mask, output_sentinel_path, nodata_value)
#
#             # Process the RADD alert file if it exists
#             with rasterio.open(radd_file_path) as radd_src:
#                 output_radd_filename = f"{radd_filename.replace('.tif', '')}_agb.tif"
#                 output_radd_path = os.path.join(output_folder, output_radd_filename)
#                 self.apply_mask_and_save(radd_src, agb_mask, output_radd_path, nodata_value)
#
# def apply_mask_and_save(self, src, mask, output_path, nodata_value):
#     """
#     Applies a mask to all bands of the source dataset and saves to a new file.
#
#     Args:
#         src (rasterio.DatasetReader): Source dataset.
#         mask (numpy.ndarray): Mask to apply.
#         output_path (str): Path to save the filtered file.
#         nodata_value (int): No-data value for filtered pixels.
#     """
#     masked_data = np.empty_like(src.read(), dtype=src.dtypes[0])
#     for i in range(src.count):
#         band_data = src.read(i + 1)
#         masked_data[i] = np.where(mask, band_data, nodata_value)
#
#     # Prepare output file with updated profile
#     output_profile = src.profile.copy()
#     output_profile.update(nodata=nodata_value)
#
#     # Write the masked data to a new file
#     with rasterio.open(output_path, 'w', **output_profile) as dst:
#         dst.write(masked_data)
#     print(f"Processed and saved to {output_path}")

#
# # Ensure output folder exists
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # Iterate over Sentinel stack files in the input folder
# for filename in os.listdir(input_folder):
#     if filename.endswith('_sentinel_normalized.tif'):  # Adjust based on your file naming convention
#         sentinel_file_path = os.path.join(input_folder, filename)
#         # Construct RADD alert file name based on Sentinel file name
#         radd_filename = filename.replace('_sentinel_normalized.tif', '_radd.tif')
#         radd_file_path = os.path.join(input_folder, radd_filename)
#
#
#
#         # Open AGB band from Sentinel stack
#         with rasterio.open(sentinel_file_path) as sentinel_src:
#             agb_band = sentinel_src.read(7)  # Assuming AGB band is the 7th band
#
#             # Define AGB mask for the specified categories
#             agb_mask = np.isin(agb_band, [2, 3, 4, 5])
#
#             # Apply AGB mask to Sentinel data
#             base_filename, file_extension = os.path.splitext(filename)
#             output_sentinel_filename = f"{base_filename}_agb{file_extension}"
#             output_sentinel_path = os.path.join(output_folder, output_sentinel_filename)
#             self.apply_mask_and_save(sentinel_src, agb_mask, output_sentinel_path, nodata_value)
#
#         # Process corresponding RADD alert file if it exists
#         if os.path.exists(radd_file_path):
#             with rasterio.open(radd_file_path) as radd_src:
#                 radd_base_filename, radd_file_extension = os.path.splitext(radd_filename)
#                 output_radd_filename = f"{radd_base_filename}_agb{radd_file_extension}"
#                 output_radd_path = os.path.join(output_folder, output_radd_filename)
#                 self.apply_mask_and_save(radd_src, agb_mask, output_radd_path, nodata_value)
#


# # Process Sentinel stack
# with rasterio.open(sentinel_file_path) as sentinel_src:
#     agb_classification = sentinel_src.read(7)  # Assuming 7th band is AGB classification
#     agb_mask = np.isin(agb_classification, valid_classes)
#
#     # Apply AGB mask to Sentinel data and RADD alerts
#     output_sentinel_path = os.path.join(output_folder, f"filtered_{filename}")
#     self.apply_mask_and_save(sentinel_src, agb_mask, output_sentinel_path, nodata_value)
#
# # Process corresponding RADD alert file if it exists
# if os.path.exists(radd_file_path):
#     with rasterio.open(radd_file_path) as radd_src:
#         output_radd_path = os.path.join(output_folder, f"filtered_{radd_filename}")
#         self.apply_mask_and_save(radd_src, agb_mask, output_radd_path, nodata_value)
#
# # Apply AGB mask to Sentinel data
# base_filename, file_extension = os.path.splitext(filename)
# output_sentinel_filename = f"{base_filename}_agb{file_extension}"
# output_sentinel_path = os.path.join(output_folder, output_sentinel_filename)
# self.apply_mask_and_save(sentinel_src, agb_mask, output_sentinel_path, nodata_value)
#
# # Process corresponding RADD alert file if it exists
# if os.path.exists(radd_file_path):
#     radd_base_filename, radd_file_extension = os.path.splitext(radd_filename)
#     output_radd_filename = f"{radd_base_filename}_agb{radd_file_extension}"
#     output_radd_path = os.path.join(output_folder, output_radd_filename)
#     self.apply_mask_and_save(radd_src, agb_mask, output_radd_path, nodata_value)
#




#
#
# def normalize_single_file_rasterio(self, file_path ,nodata_value=-9999):
#     """
#     Normalizes each band of a single Sentinel-2 data file to a range of 0-1,
#     and writes the normalized data as float32 to a new file with '_normalized' suffix.
#
#     Args:
#         file_path (str): Path to the Sentinel-2 data file.
#         nodata_value (int): Value to be treated as 'no data' and excluded from normalization.
#     """
#     if file_path.endswith('_sentinel.tif'):
#         # Create output file path by appending '_normalized' before the file extension
#         output_path = os.path.splitext(file_path)[0] + '_normalized.tif'
#
#         with rasterio.open(file_path) as src:
#             meta = src.meta
#
#             # Update the metadata to float32
#             meta.update(dtype=rasterio.float32)
#
#             # Create a new file for the normalized data
#             with rasterio.open(output_path, 'w', **meta) as dst:
#                 for i in range(1, src.count): #only first 6 bands, 7th is classification#+ 1):
#                     band = src.read(i).astype(np.float32)
#
#                     # Mask for valid data: not no-data and >= 0
#                     valid_mask = (band != nodata_value) & (band >= 0)
#
#                     # Prepare a masked array, keeping spatial structure
#                     band_masked = np.ma.masked_array(band, ~valid_mask)
#
#                     # Normalize only valid data
#                     if np.ma.is_masked(band_masked):
#                         min_val = band_masked.min()
#                         max_val = band_masked.max()
#
#                         # Normalize valid data
#                         band_normalized = (band_masked - min_val) / (max_val - min_val)
#                         band_normalized.fill_value = nodata_value
#                         dst.write_band(i, band_normalized.filled())
#                         #os.remove(file_path)
#
#         print(f"Normalized file saved as: {output_path}")


# alert_counts = []
# for file in os.listdir(stack_directory):
#     if file.endswith("fmask_stack.tif") or (stack_directory.endswith("Tiles_512_30pc_cc") and file.endswith(".tif")):
#         file_path = os.path.join(stack_directory, file)
#
#         with rasterio.open(file_path) as src:
#             radd_alerts = src.read(1)  # Assuming the RADD alert band is the first band
#             alert_count = np.count_nonzero(radd_alerts > 0)    ### alert_label_value)
#             print(f"file:{file}, alert count: {alert_count}")
#             alert_counts.append(alert_count)
#
#
#             src.close()
#             if alert_count < min_alert_pixels:
#                 base_name = file.replace('_radd.tif', '')
#                 for suffix in [".tif",'_radd_.tif', '_radd.tif', '_sentinel.tif']:
#                     file_to_delete = os.path.join(stack_directory, f"{base_name}")
#                     if os.path.exists(file_to_delete):
#                         os.remove(file_to_delete)
#                         print(f"Removed file {file_to_delete}")
#
#
#                 print(f"Removed stack {file} due to insufficient RADD alerts:{alert_count}.")
#
#
# def alter_radd_data_to_label(self, radd_tiles_path):
#
#     # Iterate over RADD files in the directory
#     for filename in os.listdir(radd_tiles_path):
#         if filename.endswith('.tif') and not filename.endswith('_.tif') and not filename.endswith('raddaltered.tif'):
#             # Extract and parse the date from the RADD alert filename
#             sentinel_stack_date = datetime.datetime.strptime(filename.split('_')[0], "%Y%j").date()
#
#             radd_alert_path = os.path.join(radd_tiles_path, filename)
#             output_filename = os.path.splitext(filename)[0] + '_raddaltered.tif'
#             output_path = os.path.join(radd_tiles_path, output_filename)
#
#             # Open the RADD alert raster file
#             with rasterio.open(radd_alert_path) as src:
#                 data = src.read(1)  # Assuming RADD data is in the first band
#
#                 # Initialize a mask for future events
#                 future_events_mask = np.zeros(data.shape, dtype=bool)
#
#                 # If the RADD alert date is in the future relative to the sentinel stack, mask it out
#                 if data > sentinel_stack_date:
#                     future_events_mask[data > 0] = True  # Mask where RADD alerts exist
#
#                 # Apply the future events mask
#                 data[future_events_mask] = src.nodata  # Use the nodata value defined in the source file
#
#                 # Convert remaining values greater than 0 to 1
#                 data[data > 0] = 1
#
#                 # Update profile and save the altered data
#                 profile = src.profile
#                 with rasterio.open(output_path, 'w', **profile) as dst:
#                     dst.write(data, 1)  # Write modified data back to band 1
#
#                 print(f"Processed and saved {output_filename}")

#
# def alter_radd_data_to_label(self, radd_tiles_path):
#     # Iterate over files in the training directory
#     for filename in os.listdir(radd_tiles_path):
#         if filename.endswith('.tif') and not filename.endswith('_.tif') and not filename.endswith('raddaltered.tif'):
#             radd_alert_path = os.path.join(radd_tiles_path, filename)
#             output_filename = os.path.splitext(filename)[0] + '_raddaltered.tif'
#             output_path = os.path.join(radd_tiles_path, output_filename)
#
#             # Open the RADD alert raster file
#             with rasterio.open(radd_alert_path) as src:
#                 # Read all the data (assuming the RADD data is in the first band)
#                 data = src.read()
#
#                 # Convert values greater than 0 to 1 in the first band
#                 radd_data = data[0, :, :]
#                 radd_data[radd_data > 0] = 1
#                 data[0, :, :] = radd_data  # Replace the first band with the altered data
#
#                 # Save the altered data to the new file, including all bands
#                 profile = src.profile
#                 src.close()
#                 profile.update(count=data.shape[0])  # Update the count with the number of bands
#                 with rasterio.open(output_path, 'w', **profile) as dst:
#                     dst.write(data)
#
#             # Now remove the original file, after confirming the new file is written
#             file_to_delete = radd_alert_path
#             if os.path.exists(file_to_delete):
#                 os.remove(file_to_delete)
#                 print(f"Removed file {file_to_delete}")

## crop fmask to sen2 stacks
# for sentinel2_file in os.listdir(forest_stacks_folder):
#     if sentinel2_file.endswith('.tif'):
#         sentinel2_file_path = os.path.join(stack_path_list, sentinel2_file)
#
#         # Call crop_images_to_stacks for each Sentinel-2 file
#         hls_data.crop_single_stack(sentinel2_file_path,
#                                    os.path.join(land_cover_path, "Kalimantan_land_cover.tif"), cropped_land_cover_path)
#

# # Process each Sentinel-2 stack file with the corresponding FMask file
# for stack_file in os.listdir(fmask_applied_folder):
#     if stack_file.endswith('.v2.0.Fmask.tif'):
#         parts = stack_file.split('.')
#         date = parts[3]  # Extract the date
#         tile = parts[2]  # Extract the tile
#         fmask_file = f"HLS.S30.{tile.upper()}.{date}.v2.0.Fmask.tif"
#         sentinel_stack_path = os.path.join(fmask_applied_folder, stack_file)
#         fmask_path = os.path.join(fmask_folder, fmask_file)
#
#         if os.path.exists(sentinel_stack_path) and os.path.exists(fmask_path):
#             output_file = os.path.join(fmask_applied_folder, f"{tile}_{date}_radd_Fmaskapplied.tif")
#             hls_data.apply_fmask(sentinel_stack_path, fmask_path, output_file)





##########
## creates the stack_radds files that dont have extended date name
##########

# for sentinel2_file in os.listdir(stack_path_list):
#     if sentinel2_file.endswith('.tif'):
#         sentinel2_file_path = os.path.join(stack_path_list, sentinel2_file)
#         date = sentinel2_file.split('_')[0]
#         tile = sentinel2_file.split('_')[1]
#
#         # Corresponding RADD alerts file path
#         radd_file = f"{date}_{tile}_resampled_merged_radd_alerts_qgis_int16_compressed_30m.tif"
#         radd_file_path = os.path.join(cropped_radd_alert_path, radd_file)
#
#         # Output file path for the merged stack
#         output_file = os.path.join(combined_radd_sen2_stack_path, f"{tile}_{date}_stacked_radd.tif")
#
#         # Check if the output file already exists, skip if it does
#         if os.path.exists(output_file):
#             print(f"Output file {output_file} already exists. Skipping...")
#             continue
#
#         if os.path.exists(radd_file_path):
#             warped_band_files = []
#
#             # Warp each band in Sentinel-2 stack
#             for i in range(1, 7):  # Adjust based on the number of bands in your Sentinel-2 stack
#                 warped_band_file = f"{output_file.replace('.tif', '')}_warped_band_{i}.tif"
#                 hls_data.warp_band(sentinel2_file_path, i, warped_band_file)
#                 warped_band_files.append(warped_band_file)
#
#             # Warp the RADD alerts band
#             warped_radd_file = f"{output_file.replace('.tif', '')}_warped_radd_band.tif"
#             hls_data.warp_band(radd_file_path, 1, warped_radd_file)  # Assuming RADD file has only one band
#             warped_band_files.append(warped_radd_file)
#
#             # Merge the warped bands
#             hls_data.merge_bands(warped_band_files, output_file)
#
#             # Clean up individual band files
#             for file in warped_band_files:
#                 os.remove(file)
#
#             print(f"Warped and merged raster saved to {output_file}")


#
# def merge_with_agb(self, agb_path, output_path):
#     for sentinel_file in os.listdir(self.stack_path_list):
#         if sentinel_file.endswith('_stack.tif'):
#             sentinel_stack_path = os.path.join(self.stack_path_list, sentinel_file)
#
#             # Extract the tile and date from the Sentinel-2 file name
#             tile, date = sentinel_file.split('_')[0].split(".")
#
#             with rasterio.open(sentinel_stack_path) as sentinel_stack, rasterio.open(agb_path) as agb:
#                 # Check if AGB data needs to be reprojected to match Sentinel-2 stack CRS and resolution
#                 if agb.crs != sentinel_stack.crs or agb.res != sentinel_stack.res:
#                     agb_data = np.zeros((1, sentinel_stack.height, sentinel_stack.width), dtype=rasterio.float32)
#                     # Define the target transform and shape
#                     transform, width, height = calculate_default_transform(
#                         agb.crs, sentinel_stack.crs, agb.width, agb.height, *agb.bounds,
#                         dst_width=sentinel_stack.width, dst_height=sentinel_stack.height)
#                     # Reproject AGB data to match Sentinel-2 stack CRS and resolution
#                     reproject(
#                         source=rasterio.band(agb, 1),
#                         destination=agb_data[0],
#                         src_transform=agb.transform,
#                         src_crs=agb.crs,
#                         dst_transform=transform,
#                         dst_crs=sentinel_stack.crs,
#                         resampling=Resampling.nearest)
#                 else:
#                     # Read AGB data directly if no reprojection is needed
#                     agb_data = agb.read(1)
#
#                 # Stack AGB data as an additional band to Sentinel-2 data
#                 stacked_data = np.concatenate((sentinel_stack.read(), agb_data), axis=0)
#
#                 # Update the profile for the output file
#                 output_profile = sentinel_stack.profile.copy()
#                 output_profile.update(count=stacked_data.shape[0])
#
#                 # Write the stacked data to a new file
#                 output_file_name = sentinel_file.replace('.tif', '_agb.tif')
#                 output_file_path = os.path.join(output_path, output_file_name)
#                 with rasterio.open(output_file_path, 'w', **output_profile) as dst:
#                     dst.write(stacked_data)
#
# def merge_with_agb(self, sentinel_stack_path, agb_path, output_path):
#     with rasterio.open(sentinel_stack_path) as sentinel_stack, rasterio.open(agb_path) as agb:
#         # Check and reproject AGB data to match Sentinel-2 stack CRS and resolution
#         if agb.crs != sentinel_stack.crs or agb.res != sentinel_stack.res:
#             # Reproject and resample AGB data
#             # (Add reprojection and resampling code here)
#             pass
#
#         # Read and stack data
#         sentinel_data = sentinel_stack.read()
#         agb_data = agb.read(1)
#
#         # Stack AGB data as an additional band to Sentinel-2 data
#         stacked_data = np.concatenate((sentinel_data, agb_data[None, :, :]), axis=0)
#
#         # Update the profile for the output file
#         output_profile = sentinel_stack.profile.copy()
#         output_profile.update(count=sentinel_data.shape[0] + 1)
#
#         # Write the stacked data to a new file
#         output_file = os.path.join(output_path, os.path.basename(sentinel_stack_path).replace('.tif', '_with_AGB.tif'))
#         with rasterio.open(output_file, 'w', **output_profile) as dest:
#             dest.write(stacked_data)





#
# def write_hls_rasterio_stack(self):
#     """
#     Write folder of Sentinel-2 GeoTIFFs, corresponding Fmask, to a GeoTIFF stack file.
#     """
#     # Create a dictionary to hold file paths for each tile-date combination
#     tile_date_files = {}
#
#     # Collect all band files into the dictionary
#     for file in os.listdir(self.sentinel2_path):
#         if any(band in file for band in self.bands) and file.endswith('.tif'):
#             # Construct the key as tile-date (e.g., T49MCU.2021186)
#             tile_date_key = '.'.join(file.split('.')[3][:7])
#             if tile_date_key not in tile_date_files:
#                 tile_date_files[tile_date_key] = []
#             tile_date_files[tile_date_key].append(os.path.join(self.sentinel2_path, file))
#
#     # Process each tile-date set of files
#     for tile_date, files in tile_date_files.items():
#         # Sort files to ensure they are in the correct order
#         sorted_files = sorted(files, key=lambda x: self.bands.index(x.split('.')[-2]))
#
#         # Corresponding Fmask file
#         fmask_file_name = f"{tile_date}.Fmask.tif"
#         fmask_file = os.path.join(self.sentinel2_path, fmask_file_name)
#
#         if not os.path.exists(fmask_file):
#             print(f"Fmask file missing for {tile_date}, skipping.")
#             continue  # Skip if corresponding Fmask file does not exist
#
#         # Files to be stacked (only Sentinel-2 bands and Fmask)
#         files_to_stack = sorted_files + [fmask_file]
#
#         # Read the first image to setup profile
#         with rasterio.open(files_to_stack[0]) as src_image:
#             dst_profile = src_image.profile.copy()
#             # Update count to include all bands plus Fmask
#             dst_profile.update({"count": len(files_to_stack)})
#
#             # Create stack directory if it doesn't exist
#             if not os.path.exists(self.stack_path_list):
#                 os.makedirs(self.stack_path_list)
#
#             stack_file = os.path.join(self.stack_path_list, f"{tile_date}_stack.tif")
#             with rasterio.open(stack_file, 'w', **dst_profile) as dst:
#                 for i, file_path in enumerate(files_to_stack, start=1):
#                     with rasterio.open(file_path) as src:
#                         data = src.read(1)  # Read the first band
#                         dst.write(data, i)
#             print(f"Stack file created: {stack_file}")


#
#
# def write_hls_rasterio_stack(self):
#     """
#     Write folder of Sentinel-2 GeoTIFFs, corresponding Fmask, to a GeoTIFF stack file.
#     """
#
#     for file in os.listdir(self.sentinel2_path):
#         if any(band in file for band in self.bands) and file.endswith('.tif'):
#             sentinel_file = os.path.join(self.sentinel2_path, file)
#
#             # Corresponding Fmask file
#             fmask_file_name = '.'.join(file.split('.')[:-2]) + '.Fmask.tif'
#             fmask_file = os.path.join(self.sentinel2_path, fmask_file_name)
#
#             if not os.path.exists(fmask_file):
#                 continue  # Skip if corresponding Fmask file does not exist
#
#             # Files to be stacked (only Sentinel-2 bands and Fmask)
#             files = [sentinel_file, fmask_file]
#
#             # Read the first image to setup profile
#             with rasterio.open(sentinel_file) as src_image:
#                 dst_crs = src_image.crs
#                 dst_transform, dst_width, dst_height = calculate_default_transform(
#                     src_image.crs, dst_crs, src_image.width, src_image.height, *src_image.bounds)
#
#                 # Create a profile for the stack
#                 dst_profile = src_image.profile.copy()
#                 dst_profile.update({
#                     "driver": "GTiff",
#                     "count": len(files),
#                     "crs": dst_crs,
#                     "transform": dst_transform,
#                     "width": dst_width,
#                     "height": dst_height
#                 })
#
#                 # Create stack directory if it doesn't exist
#                 if not os.path.exists(self.stack_path_list):
#                     os.makedirs(self.stack_path_list)
#
#                 stack_file = os.path.join(self.stack_path_list, f'{os.path.splitext(file)[0]}_stack.tif')
#                 with rasterio.open(stack_file, 'w', **dst_profile) as dst:
#                     for i, file_path in enumerate(files, start=1):
#                         with rasterio.open(file_path) as src:
#                             data = src.read(1)  # Read the first band
#                             dst.write(data, i)
#
#


#
# def alter_radd_data_to_label(self, radd_tiles_path):
#     # Iterate over files in the training directory
#     for filename in os.listdir(radd_tiles_path):
#         if filename.endswith('.tif'):
#             radd_alert_path = os.path.join(radd_tiles_path, filename)
#             output_filename = os.path.splitext(filename)[0] + '_.tif'
#             output_path = os.path.join(radd_tiles_path, output_filename)
#
#             # Open the RADD alert raster file
#             with rasterio.open(radd_alert_path) as src:
#                 # Read the data
#                 radd_data = src.read(1)  # Assuming RADD data is in the first band
#
#                 # Convert values greater than 0 to 1
#                 radd_data[radd_data > 0] = 1
#
#                 # Save the altered data to the new file
#                 with rasterio.open(
#                     output_path,
#                     'w',
#                     driver='GTiff',
#                     height=radd_data.shape[0],
#                     width=radd_data.shape[1],
#                     count=1,
#                     dtype=radd_data.dtype,
#                     crs=src.crs,
#                     transform=src.transform
#                 ) as dst:
#                     dst.write(radd_data, 1)
#
#                 src.close()
#                 file_to_delete = radd_alert_path
#                 if os.path.exists(file_to_delete):
#                     os.remove(file_to_delete)
#                     print(f"Removed file {file_to_delete}")


#
# def resample_radd_alerts(self, merged_radd_alerts):
#     """
#     Resamples 'merged_radd_alerts.tif' to match the resolution of a Sentinel-2 image.
#     """
#
#     # Use the first Sentinel-2 image to determine the target resolution and transform
#     sentinel_files = [f for f in os.listdir(self.sentinel2_path) if f.endswith('.tif')]
#     if not sentinel_files:
#         raise ValueError("No Sentinel-2 images found in the specified path.")
#
#     sentinel_path = os.path.join(self.sentinel2_path, sentinel_files[0])
#     with rasterio.open(sentinel_path) as sentinel_dataset:
#         sentinel_transform = sentinel_dataset.transform
#         sentinel_crs = sentinel_dataset.crs
#         # You could also use sentinel_dataset.res to get the (x, y) resolution if needed
#
#     # Open the merged RADD alerts image
#     #merged_radd_path = os.path.join(self.radd_alert_path, 'merged_radd_alerts_qgis_int16_compressed.tif')
#     with rasterio.open(merged_radd_alerts) as merged_radd_dataset:
#         # Calculate the transform and dimensions for the new resolution to match the Sentinel-2 image
#         transform, width, height = calculate_default_transform(
#             merged_radd_dataset.crs, sentinel_crs,
#             merged_radd_dataset.width, merged_radd_dataset.height,
#             *merged_radd_dataset.bounds,
#             dst_transform=sentinel_transform
#         )
#
#         # Define the metadata for the resampled dataset
#         out_meta = merged_radd_dataset.meta.copy()
#
#         out_meta.update({
#             "driver": "GTiff",
#             "height": height,
#             "width": width,
#             "transform": transform,
#             "crs": sentinel_crs,
#             "count": merged_radd_dataset.count,  # Keep the number of bands
#             "compress": "LZW",  # Use LZW compression
#             "dtype": 'int16'  # Ensure data type matches the input file
#             # "tiled": True,  # Enable tiling
#             # "blockxsize": 256,  # Tile size (can be adjusted)
#             # "blockysize": 256
#         })
#         #
#         # out_meta.update({
#         #     "driver": "GTiff",
#         #     "height": height,
#         #     "width": width,
#         #     "transform": transform,
#         #     "crs": sentinel_crs,
#         #     "count": merged_radd_dataset.count  # Keep the number of bands
#         # })
#
#         # Perform the resampling
#         resampled_radd_path = os.path.join(self.radd_alert_path, f'resampled_merged_radd_alerts_qgis_int16_compressed.tif')
#         with rasterio.open(resampled_radd_path, 'w', **out_meta) as dest:
#             for i in range(1, merged_radd_dataset.count + 1):
#                 # Reproject and resample each band
#                 reproject(
#                     source=rasterio.band(merged_radd_dataset, i),
#                     destination=rasterio.band(dest, i),
#                     src_transform=merged_radd_dataset.transform,
#                     src_crs=merged_radd_dataset.crs,
#                     dst_transform=transform,
#                     dst_crs=sentinel_crs,
#                     resampling=Resampling.nearest
#                 )
#
#     print(f"Resampled raster saved to {resampled_radd_path}")
#     return resampled_radd_path




# def crop_single_stack(self, sentinel_stack_path, single_image_path, output_path):
#
#
#     ##############################
#     ## RADD ALERTS VERSION
#     ##############################
#     with rasterio.open(sentinel_stack_path) as sentinel_stack:
#         sentinel_bounds = sentinel_stack.bounds
#         sentinel_crs = sentinel_stack.crs
#
#         with rasterio.open(single_image_path) as image_raster:
#             image_bounds = image_raster.bounds
#             image_crs = image_raster.crs
#
#             # Reproject the image raster to the CRS of the Sentinel-2 stack
#             transformer = partial(
#                 pyproj.transform,
#                 image_crs,  # source coordinate system (Image CRS)
#                 sentinel_crs  # destination coordinate system (Sentinel-2 stack CRS)
#             )
#
#             sentinel_box = box(*sentinel_bounds)
#             sentinel_box_4326 = sentinel_box.__geo_interface__
#             sentinel_box_image_crs = shapely_transform(transformer, sentinel_box)
#
#             ## select only image raster where definite events occur.
#
#             # Get the transform from the original raster dataset
#             image_transform = image_raster.transform
#
#             # Mask or clip the image raster to the area of the Sentinel-2 stack, specifying the nodata value and transform
#             image_cropped, transform = mask(image_raster, [sentinel_box_image_crs], crop=True, filled=False, pad=False, nodata=0)
#
#             image_cropped[0][image_cropped[0] != 3] = 0
#
#             # Update the profile for the cropped image
#             output_profile = image_raster.profile.copy()
#             output_profile.update({
#                 'height': image_cropped.shape[1],
#                 'width': image_cropped.shape[2],
#                 'transform': transform
#             })
#
#             # Extract the relevant parts of the file name from the sentinel_stack_path
#             parts = os.path.basename(sentinel_stack_path).split('.')[0].split('_')
#             identifier = f"{parts[0]}_{parts[1]}"
#
#             # Assuming the base name of the single_image_path is 'resampled_radd_alerts_int16_compressed.tif'
#             suffix = os.path.basename(single_image_path)
#
#             # Combine the identifier and suffix to form the output file name
#             output_file_name = f"{identifier}_{suffix}"
#             output_file_path = os.path.join(output_path, output_file_name)
#
#
#             with rasterio.open(output_file_path, 'w', **output_profile) as dest:
#                 dest.write(image_cropped)





# File paths
sentinel_file = "E:/Data/Sentinel2_data/30pc_cc/Borneo_June2020_Jan2023_30pc_cc_stacks/2021205_T50MKC_stack.tif"
radd_file = "E:/Data/Sentinel2_data/30pc_cc/Radd_Alerts_Borneo_Cropped_30pc_cc/2021205_T50MKC_reprojected_radd_alerts_band2.tif"
output_file = "E:/Data/Sentinel2_data/30pc_cc/Borneo_June2020_Jan2023_30pc_cc_stacks_radd/2021205_T50MKC_stacked_radd_GDAL.tif"

# Warp each band in Sentinel-2 stack
warped_band_files = []
for i in range(1, 7):  # Assuming 6 bands in Sentinel-2 stack
    warped_band_file = f"E:/Data/Sentinel2_data/30pc_cc/Borneo_June2020_Jan2023_30pc_cc_stacks_radd/2021205_T50MKC_warped_band_{i}.tif"
    hls_data.warp_band(sentinel_file, i, warped_band_file)
    warped_band_files.append(warped_band_file)

# Warp the RADD alerts band
warped_radd_file = "E:/Data/Sentinel2_data/30pc_cc/Borneo_June2020_Jan2023_30pc_cc_stacks_radd/2021205_T50MKC_warped_radd_band.tif"
hls_data.warp_band(radd_file, 1, warped_radd_file)  # Assuming the RADD file has only one band
warped_band_files.append(warped_radd_file)

# Merge the warped bands
hls_data.merge_bands(warped_band_files, output_file)

# Clean up individual band files
for file in warped_band_files:
    os.remove(file)

print("Process completed.")
#








#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

import rasterio
from rasterio.enums import Resampling

def reorder_and_add_blank_band(input_file, output_file):
    with rasterio.open(input_file) as src:
        # Create metadata for the new raster with an additional band
        meta = src.meta.copy()
        meta.update(count=7)

        # Create the new raster file
        with rasterio.open(output_file, 'w', **meta) as dst:
            # Write a blank band as the first band
            blank_band = src.read(1) * 0  # Create a blank band
            dst.write(blank_band, 1)

            # Write the original bands to the new positions (2 to 7)
            for band in range(1, src.count + 1):
                data = src.read(band)
                dst.write(data, band + 1)  # Shift original bands up by one position

# File paths
sentinel_file = r'E:\Data\Sentinel2_data\30pc_cc\Borneo_June2020_Jan2023_30pc_cc_stacks\2021205_T50MKC_stack.tif'
reordered_file = r'E:\Data\Sentinel2_data\30pc_cc\Borneo_June2020_Jan2023_30pc_cc_stacks\2021205_T50MKC_stack_reordered.tif'

# Reorder bands and add a blank band
reorder_and_add_blank_band(sentinel_file, reordered_file)




def warp_rasters(input_files, output_file, src_nodata=None, dst_nodata=None):
    """
    Merges and/or reprojects raster files using gdal.Warp.

    :param input_files: List of file paths of the rasters to merge.
    :param output_file: File path for the output merged raster.
    :param src_nodata: NoData value in source files (optional).
    :param dst_nodata: NoData value for the output file (optional).
    """
    # Warp options
    warp_options = gdal.WarpOptions(format='GTiff',
                                    srcNodata=src_nodata,
                                    dstNodata=dst_nodata,
                                    multithread=True)

    # Perform the warp
    gdal.Warp(destNameOrDestDS=output_file,
              srcDSOrSrcDSTab=input_files,
              options=warp_options)

    print(f"Merged raster saved to {output_file}")

# Define the input files and output file
input_files = [
    "E:/Data/Sentinel2_data/30pc_cc/Borneo_June2020_Jan2023_30pc_cc_stacks/2021205_T50MKC_stack_reordered.tif",
    "E:/Data/Sentinel2_data/30pc_cc/Radd_Alerts_Borneo_Cropped_30pc_cc/2021205_T50MKC_reprojected_radd_alerts_band2.tif"
]
output_file = "E:/Data/Sentinel2_data/30pc_cc/Borneo_June2020_Jan2023_30pc_cc_stacks_radd/2021205_T50MKC_stacked_radd_GDAL.tif"

# Run the warp
warp_rasters(input_files, output_file)



#
#
#
#
# from osgeo import gdal
#
# def warp_raster(input_file, output_file, reference_file):
#     """
#     Warps a raster to match the projection and alignment of a reference raster.
#
#     :param input_file: Path to the input raster to be warped.
#     :param output_file: Path for the output warped raster.
#     :param reference_file: Path to the reference raster to match.
#     """
#     # Open the reference dataset to get the projection and geotransform
#     ref_ds = gdal.Open(reference_file)
#     ref_proj = ref_ds.GetProjection()
#
#     # Perform the warp
#     warp_options = gdal.WarpOptions(dstSRS=ref_proj)
#     gdal.Warp(destNameOrDestDS=output_file, srcDSOrSrcDSTab=input_file, options=warp_options)
#
#     print(f"Warped raster saved to {output_file}")
#
# # Define file paths
# radd_file = "E:/Data/Sentinel2_data/30pc_cc/Radd_Alerts_Borneo_Cropped_30pc_cc/2021205_T50MKC_reprojected_radd_alerts_band2.tif"
# output_radd_file = "E:/Data/Sentinel2_data/30pc_cc/Borneo_June2020_Jan2023_30pc_cc_stacks_radd/2021205_T50MKC_warped.tif"
# sentinel_file =  "E:/Data/Sentinel2_data/30pc_cc/Borneo_June2020_Jan2023_30pc_cc_stacks/2021205_T50MKC_stack.tif"
#
# # Warp the RADD alerts band
# warp_raster(radd_file, output_radd_file, sentinel_file)
#
#





import rasterio
from rasterio.merge import merge

def stack_bands(sentinel_file, warped_radd_file, output_file):
    """
    Stacks bands from a Sentinel-2 file and a warped RADD alerts file into a single raster.

    :param sentinel_file: Path to the Sentinel-2 file.
    :param warped_radd_file: Path to the warped RADD alerts file.
    :param output_file: Path for the output stacked raster.
    """
    with rasterio.open(sentinel_file) as sen_ds, rasterio.open(warped_radd_file) as radd_ds:
        # Read the Sentinel-2 bands
        sen_bands = [sen_ds.read(i) for i in range(1, sen_ds.count + 1)]

        # Read the warped RADD alert band (assuming it's the second band)
        radd_band = radd_ds.read(1)

        # Prepare the metadata for the output dataset
        out_meta = sen_ds.meta.copy()
        out_meta.update(count=sen_ds.count + 1)

        # Write the stacked bands to the output dataset
        with rasterio.open(output_file, 'w', **out_meta) as out_ds:
            for i, band in enumerate(sen_bands, start=1):
                out_ds.write(band, i)
            out_ds.write(radd_band, sen_ds.count + 1)

    print(f"Stacked raster saved to {output_file}")
#
# Define file paths
output_file = "E:/Data/Sentinel2_data/30pc_cc/Borneo_June2020_Jan2023_30pc_cc_stacks_radd/2021205_T50MKC_stacked_radd_GDAL.tif"

# Stack the bands
stack_bands(sentinel_file, warped_radd_file, output_file)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# def stack_bands(sentinel_file, radd_file, output_file):
#     """
#     Stacks bands from a Sentinel-2 file and a RADD alerts file into a single raster.
#
#     :param sentinel_file: Path to the Sentinel-2 file.
#     :param radd_file: Path to the RADD alerts file.
#     :param output_file: Path for the output stacked raster.
#     """
#     # Open the Sentinel-2 and RADD alert datasets
#     sentinel_ds = gdal.Open(sentinel_file)
#     radd_ds = gdal.Open(radd_file)
#
#     # Create an output dataset with 7 bands (6 from Sentinel and 1 from RADD)
#     driver = gdal.GetDriverByName('GTiff')
#     out_ds = driver.Create(output_file, sentinel_ds.RasterXSize, sentinel_ds.RasterYSize, 7, sentinel_ds.GetRasterBand(1).DataType)
#     out_ds.SetGeoTransform(sentinel_ds.GetGeoTransform())
#     out_ds.SetProjection(sentinel_ds.GetProjection())
#
#     # Copy Sentinel-2 bands to the output dataset
#     for i in range(1, 7):
#         out_band = out_ds.GetRasterBand(i)
#         in_band = sentinel_ds.GetRasterBand(i)
#         out_band.WriteArray(in_band.ReadAsArray())
#         out_band.FlushCache()
#
#     # Add the RADD alert band as the 7th band in the output dataset
#     radd_band = radd_ds.GetRasterBand(1)  # Assuming we want the 2nd band from the RADD alerts
#     out_band = out_ds.GetRasterBand(7)
#     out_band.WriteArray(radd_band.ReadAsArray())
#     out_band.FlushCache()
#
#     # Clean up
#     sentinel_ds = None
#     radd_ds = None
#     out_ds = None
#
#     print(f"Stacked raster saved to {output_file}")
#
# # Define file paths
# sentinel_file = "E:/Data/Sentinel2_data/30pc_cc/Borneo_June2020_Jan2023_30pc_cc_stacks/2021205_T50MKC_stack.tif"
# radd_file = "E:/Data/Sentinel2_data/30pc_cc/Radd_Alerts_Borneo_Cropped_30pc_cc/2021205_T50MKC_reprojected_radd_alerts_band2.tif"
# output_file = "E:/Data/Sentinel2_data/30pc_cc/Borneo_June2020_Jan2023_30pc_cc_stacks_radd/2021205_T50MKC_stacked_radd_GDAL.tif"
#
# # Run the stacking
# stack_bands(sentinel_file, radd_file, output_file)


# input_files = [
#     "E:/Data/Sentinel2_data/30pc_cc/Borneo_June2020_Jan2023_30pc_cc_stacks/2021205_T50MKC_stack.tif",
#     "E:/Data/Sentinel2_data/30pc_cc/Radd_Alerts_Borneo_Cropped_30pc_cc/2021205_T50MKC_reprojected_radd_alerts_band2.tif"
# ]
# output_file = "E:/Data/Sentinel2_data/30pc_cc/Borneo_June2020_Jan2023_30pc_cc_stacks_radd/2021205_T50MKC_stacked_radd_GDAL.tif"
#




    #
    # def stack_images(self, input_folder, aux_folder, output_folder, aux_type):
    #     """
    #     Stacks each Sentinel-2 image with the second band of its corresponding auxiliary image.
    #     First clips the auxiliary image to match the Sentinel-2 extent.
    #     """
    #
    #     aux_suffix = {'radd': '_resampled_merged_radd_alerts_qgis_int16_compressed_30m.tif', 'land_cover': '_land_cover.tif'}.get(aux_type, '')
    #
    #     for file in os.listdir(input_folder):
    #         if file.endswith('_stack.tif'):
    #             date_tile = file.split('_')[0]
    #
    #             aux_file = f"{date_tile}{aux_suffix}"
    #             aux_file_path = os.path.join(aux_folder, aux_file)
    #
    #             if os.path.exists(aux_file_path):
    #                 sentinel_file_path = os.path.join(input_folder, file)
    #                 clipped_aux_file_path = os.path.join(aux_folder, f"clipped_{aux_file}")
    #
    #                 # Clip the auxiliary file to match the Sentinel-2 extent
    #                 self.clip_to_extent(aux_file_path, sentinel_file_path, clipped_aux_file_path)
    #
    #                 # Open the clipped auxiliary image and the Sentinel-2 image
    #                 with rasterio.open(sentinel_file_path) as src1, \
    #                         rasterio.open(clipped_aux_file_path) as src2:
    #
    #                     # Update metadata for the output dataset
    #                     out_meta = src1.meta.copy()
    #                     out_meta.update({"count": src1.count + 1})
    #
    #                     output_path = os.path.join(output_folder, f"{date_tile}_stacked_{aux_type}.tif")
    #                     with rasterio.open(output_path, 'w', **out_meta) as dest:
    #                         for i in range(1, src1.count + 1):
    #                             data = src1.read(i)
    #                             dest.write(data, i)
    #
    #                         # Write the second band from the clipped auxiliary image
    #                         aux_data = src2.read(2)  # Reading only the second band
    #                         dest.write(aux_data, src1.count + 1)
    #
    #     print(f"Image stacking for {aux_type} complete.")
    #
    #
    #
    # def stack_images(self, input_folder, aux_folder, output_folder, aux_type):
    #     """
    #     Stacks each Sentinel-2 image with the second band of its corresponding auxiliary image.
    #     """
    #
    #     # Define the file suffix based on the auxiliary image type
    #     aux_suffix = {'radd': '_resampled_merged_radd_alerts_qgis_int16_compressed_30m.tif', 'land_cover': '_land_cover.tif'}.get(aux_type, '')
    #
    #     # Loop through all files in the Sentinel-2 input folder
    #     for file in os.listdir(input_folder):
    #         if file.endswith('_stack.tif'):
    #             date = file.split('_')[0]
    #             tile = file.split('_')[1]
    #
    #             # Construct the filename for the auxiliary image
    #             aux_file = f"{date}_{tile}{aux_suffix}"
    #             aux_file_path = os.path.join(aux_folder, aux_file)
    #
    #             if os.path.exists(aux_file_path):
    #                 with rasterio.open(os.path.join(input_folder, file)) as src1, \
    #                         rasterio.open(aux_file_path) as src2:
    #
    #                     # Ensure the spatial dimensions match
    #                     # Add resampling or cropping logic here if needed
    #
    #                     # Update metadata for the output dataset
    #                     out_meta = src1.meta.copy()
    #                     out_meta.update({"count": src1.count + 1})  # Adding one band from the auxiliary image
    #
    #                     output_path = os.path.join(output_folder, f"{date}_{tile}_stacked_{aux_type}.tif")
    #                     with rasterio.open(output_path, 'w', **out_meta) as dest:
    #                         # Write the existing bands from Sentinel-2
    #                         for i in range(1, src1.count + 1):
    #                             data = src1.read(i)
    #                             dest.write(data, i)
    #
    #                         # Write the second band from the auxiliary image
    #                         aux_data = src2.read(2)  # Reading only the second band
    #                         dest.write(aux_data, src1.count + 1)
    #
    #     print(f"Image stacking for {aux_type} complete.")



    #
    # def stack_images(self, input_folder, aux_folder, output_folder, aux_type):
    #     """
    #     Stacks each Sentinel-2 image with its corresponding auxiliary image (e.g., RADD alerts, land cover)
    #     based on the date_tile string.
    #
    #     :param input_folder: Folder containing Sentinel-2 stack images.
    #     :param aux_folder: Folder containing auxiliary images (RADD alerts, land cover, etc.).
    #     :param output_folder: Folder where the new stacked images will be saved.
    #     :param aux_type: Type of auxiliary image (e.g., 'radd', 'land_cover').
    #     """
    #
    #     # Define the file suffix based on the auxiliary image type
    #     aux_suffix = {
    #         'radd': '_resampled_merged_radd_alerts_qgis_int16_compressed.tif',
    #         'land_cover': '_land_cover.tif'
    #     }.get(aux_type, '')
    #
    #     # Loop through all files in the Sentinel-2 input folder
    #     for file in os.listdir(input_folder):
    #         if file.endswith('_stack.tif'):
    #             # Extract the date_tile string from the Sentinel-2 filename
    #             date = file.split('_')[0]
    #             tile = file.split('_')[1]
    #
    #             # Construct the filename for the auxiliary image
    #             aux_file = f"{date}_{tile}{aux_suffix}"
    #             aux_file_path = os.path.join(aux_folder, aux_file)
    #
    #             if os.path.exists(aux_file_path):
    #                 # Open the Sentinel-2 stack image and the auxiliary image
    #                 with rasterio.open(os.path.join(input_folder, file)) as src1, \
    #                      rasterio.open(aux_file_path) as src2:
    #
    #                     # Check if the dimensions and CRS match; if not, resample or reproject as necessary
    #
    #                     # Stack the images together
    #                     stacked_data = merge([src1, src2])
    #
    #                     # Define the metadata for the output dataset
    #                     out_meta = src1.meta.copy()
    #                     out_meta.update({
    #                         "count": src1.count + src2.count  # Update the band count
    #                     })
    #
    #                     # Save the new stacked image
    #                     output_path = os.path.join(output_folder, f"{date}_{tile}_stacked_{aux_type}.tif")
    #                     with rasterio.open(output_path, 'w', **out_meta) as dest:
    #                         dest.write(stacked_data)
    #
    #     print(f"Image stacking for {aux_type} complete.")
    #
    #














# def crop_single_stack(self, sentinel_stack_path, single_image_path, output_path):
#
#     ##############################
#     ## AGB LAND CLASSIFICATION VERSION
#     ##############################
#     with rasterio.open(sentinel_stack_path) as sentinel_stack:
#         sentinel_bounds = sentinel_stack.bounds
#         sentinel_crs = sentinel_stack.crs
#
#         with rasterio.open(single_image_path) as image_raster:
#             image_bounds = image_raster.bounds
#             image_crs = image_raster.crs
#
#             # Reproject the image raster to the CRS of the Sentinel-2 stack
#             transformer = partial(
#                 pyproj.transform,
#                 image_crs,  # source coordinate system (Image CRS)
#                 sentinel_crs  # destination coordinate system (Sentinel-2 stack CRS)
#             )
#
#             sentinel_box = box(*sentinel_bounds)
#
#             sentinel_box_4326 = sentinel_box.__geo_interface__
#             sentinel_box_image_crs = shapely_transform(transformer, sentinel_box)
#
#             ## select only image raster where definite events occur.
#
#             # Get the transform from the original raster dataset
#             image_transform = image_raster.transform
#             # Convert image_bounds to a Shapely box
#             image_box = box(image_bounds.left, image_bounds.bottom, image_bounds.right, image_bounds.top)
#
#             if not sentinel_box_image_crs.intersects(image_box):
#                 print("doesn't intersect")
#                 return
#
#             # Mask or clip the image raster to the area of the Sentinel-2 stack, specifying the nodata value and transform
#             image_cropped, transform = mask(image_raster, [sentinel_box_image_crs], crop=True, filled=False, pad=False, nodata=0)
#    #             image_cropped[0][image_cropped[0] != 3] = 0
#
#             # Update the profile for the cropped image
#             output_profile = image_raster.profile.copy()
#             output_profile.update({
#                 'height': image_cropped.shape[1],
#                 'width': image_cropped.shape[2],
#                 'transform': transform
#             })
#
#             # Extract the relevant parts of the file name from the sentinel_stack_path
#             parts = os.path.basename(sentinel_stack_path).split('.')[0].split('_')
#             identifier = f"{parts[0]}_{parts[1]}"
#
#             # Assuming the base name of the single_image_path is 'resampled_radd_alerts_int16_compressed.tif'
#             suffix = os.path.basename(single_image_path)
#
#             # Combine the identifier and suffix to form the output file name
#             output_file_name = f"{identifier}_{suffix}"
#             output_file_path = os.path.join(output_path, output_file_name)
#
#             if os.path.exists(output_file_path):
#                 print(f"File {output_file_path} already exists. Skipping cropping.")
#                 return  # Skip the rest of the function
#
#
#             with rasterio.open(output_file_path, 'w', **output_profile) as dest:
#                 dest.write(image_cropped)


#
# def crop_images_to_stacks(self, sentinel_stacks_folder, images_path, output_folder):
#     """
#     Crops raster images (either a single image or multiple images in a folder) to match the extents of Sentinel-2 image stacks.
#     """
#     # Check if images_path is a directory or a single file
#     is_directory = os.path.isdir(images_path)
#
#     # Define the projection transformer
#     transformer = partial(
#         pyproj.transform,
#         pyproj.Proj(init='epsg:32649'),  # source coordinate system (Sentinel-2 stacks)
#         pyproj.Proj(init='epsg:4326')  # destination coordinate system (Images)
#     )
#
#     for sentinel_file in os.listdir(sentinel_stacks_folder):
#         if sentinel_file.endswith('_stack.tif'):
#             sentinel_path = os.path.join(sentinel_stacks_folder, sentinel_file)
#             with rasterio.open(sentinel_path) as sentinel_stack:
#                 sentinel_bounds = sentinel_stack.bounds
#                 sentinel_crs = sentinel_stack.crs
#                 sentinel_box = box(*sentinel_bounds)
#
#                 # Reproject Sentinel-2 bounding box to EPSG:4326
#                 sentinel_box_4326 = shapely_transform(transformer, sentinel_box)
#
#                 # Iterate over the images (if directory) or just use the single image
#                 images_to_process = os.listdir(images_path) if is_directory else [images_path]
#                 for image_file in images_to_process:
#                     image_path = os.path.join(images_path, image_file) if is_directory else image_file
#                     with rasterio.open(image_path) as image_raster:
#                         # Reproject the image_box to EPSG:4326
#                         image_box = box(*image_raster.bounds)
#                         image_box_4326 = shapely_transform(
#                             partial(
#                                 pyproj.transform,
#                                 image_raster.crs,
#                                 pyproj.Proj(init='epsg:4326')
#                             ),
#                             image_box
#                         )
#
#                         # Check if the bounding boxes intersect (overlap)
#                         if not sentinel_box_4326.intersects(image_box_4326):
#                             continue  # Skip if no overlap
#
#                         # Calculate the transformation and new dimensions for reprojection
#                         transform, width, height = calculate_default_transform(
#                             image_raster.crs, sentinel_crs, image_raster.width, image_raster.height, *image_raster.bounds
#                         )
#
#                         # Reproject image raster in memory
#                         with MemoryFile() as memfile:
#                             with memfile.open(driver='GTiff', height=height, width=width, count=1, dtype=image_raster.dtypes[0], crs=sentinel_crs,
#                                               transform=transform) as image_reprojected:
#                                 reproject(
#                                     source=image_raster.read(1),  # Assuming a single band
#                                     destination=image_reprojected,
#                                     src_transform=image_raster.transform,
#                                     src_crs=image_raster.crs,
#                                     dst_transform=transform,
#                                     dst_crs=sentinel_crs,
#                                     resampling=Resampling.nearest
#                                 )
#
#                                 # Get the window for cropping
#                                 window = from_bounds(*sentinel_bounds, transform=transform, width=width, height=height)
#                                 image_cropped = image_reprojected.read(1, window=window)
#
#                                 # Copy the profile to use it outside the 'with' block
#                                 output_profile = image_reprojected.profile.copy()
#                                 output_profile.update({
#                                     'height': window.height,
#                                     'width': window.width,
#                                     'transform': image_reprojected.window_transform(window)
#                                 })
#
#                             output_file_name = f"{sentinel_file.split('.')[0]}_{os.path.basename(image_file)}"
#                             output_file_path = os.path.join(output_folder, output_file_name)
#
#                             # Save the cropped image raster
#                             with rasterio.open(output_file_path, "w", **output_profile) as dest:
#                                 dest.write(image_cropped)

#
# def crop_images_to_stacks(self, sentinel_stacks_folder, images_path, output_folder):
#     """
#     Crops raster images (either a single image or multiple images in a folder) to match the extents of Sentinel-2 image stacks.
#     """
#     # Check if images_path is a directory or a single file
#     is_directory = os.path.isdir(images_path)
#
#     # Define the projection transformer
#     transformer = partial(
#         pyproj.transform,
#         pyproj.Proj(init='epsg:32649'),  # source coordinate system (Sentinel-2 stacks)
#         pyproj.Proj(init='epsg:4326')  # destination coordinate system (Images)
#     )
#
#     for sentinel_file in os.listdir(sentinel_stacks_folder):
#         if sentinel_file.endswith('_stack.tif'):
#             sentinel_path = os.path.join(sentinel_stacks_folder, sentinel_file)
#             with rasterio.open(sentinel_path) as sentinel_stack:
#                 sentinel_bounds = sentinel_stack.bounds
#                 sentinel_crs = sentinel_stack.crs
#                 sentinel_box = box(*sentinel_bounds)
#
#                 # Reproject Sentinel-2 bounding box to EPSG:4326
#                 sentinel_box_4326 = shapely_transform(transformer, sentinel_box)
#
#                 # Iterate over the images (if directory) or just use the single image
#                 images_to_process = os.listdir(images_path) if is_directory else [images_path]
#                 for image_file in images_to_process:
#                     image_path = os.path.join(images_path, image_file) if is_directory else image_file
#                     with rasterio.open(image_path) as image_raster:
#                         # Reproject the image_box to EPSG:4326
#                         image_box = box(*image_raster.bounds)
#                         image_box_4326 = shapely_transform(
#                             partial(
#                                 pyproj.transform,
#                                 image_raster.crs,
#                                 pyproj.Proj(init='epsg:4326')
#                             ),
#                             image_box
#                         )
#
#                         # Check if the bounding boxes intersect (overlap)
#                         if not sentinel_box_4326.intersects(image_box_4326):
#                             continue  # Skip if no overlap
#
#                         # Calculate the transformation and new dimensions for reprojection
#                         transform, width, height = calculate_default_transform(
#                             image_raster.crs, sentinel_crs, image_raster.width, image_raster.height, *image_raster.bounds
#                         )
#
#                         # Reproject image raster in memory
#                         with MemoryFile() as memfile:
#                             with memfile.open(driver='GTiff', height=height, width=width, count=image_raster.count, dtype=image_raster.dtypes[0], crs=sentinel_crs,
#                                               transform=transform) as image_reprojected:
#                                 for i in range(1, image_raster.count + 1):
#                                     reproject(
#                                         source=rasterio.band(image_raster, i),
#                                         destination=rasterio.band(image_reprojected, i),
#                                         src_transform=image_raster.transform,
#                                         src_crs=image_raster.crs,
#                                         dst_transform=transform,
#                                         dst_crs=sentinel_crs,
#                                         resampling=Resampling.nearest
#                                     )
#
#                                 # Get the window for cropping
#                                 window = from_bounds(*sentinel_bounds, transform=transform, width=width, height=height)
#                                 image_cropped = image_reprojected.read(window=window)
#
#                                 # Copy the profile to use it outside the 'with' block
#                                 output_profile = image_reprojected.profile.copy()
#                                 output_profile.update({
#                                     'height': window.height,
#                                     'width': window.width,
#                                     'transform': image_reprojected.window_transform(window)
#                                 })
#
#                             output_file_name = f"{sentinel_file.split('.')[0]}_{os.path.basename(image_file)}"
#                             output_file_path = os.path.join(output_folder, output_file_name)
#
#                             # Save the cropped image raster
#                             with rasterio.open(output_file_path, "w", **output_profile) as dest:
#                                 dest.write(image_cropped)
# #
#
# def crop_images_to_stacks(self, sentinel_stacks_folder, images_path, output_folder):
#     """
#     Crops raster images (either a single image or multiple images in a folder) to match the extents of Sentinel-2 image stacks.
#     """
#     # Check if images_path is a directory or a single file
#     is_directory = os.path.isdir(images_path)
#
#     # Define the projection transformer
#     transformer = partial(
#         pyproj.transform,
#         pyproj.Proj(init='epsg:32649'),  # source coordinate system (Sentinel-2 stacks)
#         pyproj.Proj(init='epsg:4326')  # destination coordinate system (Images)
#     )
#
#     for sentinel_file in os.listdir(sentinel_stacks_folder):
#         if sentinel_file.endswith('_stack.tif'):
#             sentinel_path = os.path.join(sentinel_stacks_folder, sentinel_file)
#             with rasterio.open(sentinel_path) as sentinel_stack:
#                 sentinel_bounds = sentinel_stack.bounds
#                 sentinel_crs = sentinel_stack.crs
#                 sentinel_box = box(*sentinel_bounds)
#
#                 # Reproject Sentinel-2 bounding box to EPSG:4326
#                 sentinel_box_4326 = shapely_transform(transformer, sentinel_box)
#
#                 # Iterate over the images (if directory) or just use the single image
#                 images_to_process = os.listdir(images_path) if is_directory else [images_path]
#                 for image_file in images_to_process:
#                     image_path = os.path.join(images_path, image_file) if is_directory else image_file
#                     with rasterio.open(image_path) as image_raster:
#                         image_box = box(*image_raster.bounds)
#
#                         # Check if the bounding boxes intersect (overlap)
#                         if not sentinel_box_4326.intersects(image_box):
#                             continue  # Skip if no overlap
#
#                         # Calculate the transformation and new dimensions for reprojection
#                         transform, width, height = calculate_default_transform(
#                             image_raster.crs, sentinel_crs, image_raster.width, image_raster.height, *image_raster.bounds
#                         )
#
#                         # Reproject image raster in memory
#                         with MemoryFile() as memfile:
#                             with memfile.open(driver='GTiff', height=height, width=width, count=image_raster.count, dtype=image_raster.dtypes[0], crs=sentinel_crs,
#                                               transform=transform) as image_reprojected:
#                                 for i in range(1, image_raster.count + 1):
#                                     reproject(
#                                         source=rasterio.band(image_raster, i),
#                                         destination=rasterio.band(image_reprojected, i),
#                                         src_transform=image_raster.transform,
#                                         src_crs=image_raster.crs,
#                                         dst_transform=transform,
#                                         dst_crs=sentinel_crs,
#                                         resampling=Resampling.nearest
#                                     )
#
#                                 # Get the window for cropping
#                                 window = from_bounds(*sentinel_bounds, transform=transform, width=width, height=height)
#                                 image_cropped = image_reprojected.read(window=window)
#
#                                 # Copy the profile to use it outside the 'with' block
#                                 output_profile = image_reprojected.profile.copy()
#                                 output_profile.update({
#                                     'height': window.height,
#                                     'width': window.width,
#                                     'transform': image_reprojected.window_transform(window)
#                                 })
#
#                             output_file_name = f"{sentinel_file.split('.')[0]}_{os.path.basename(image_file)}"
#                             output_file_path = os.path.join(output_folder, output_file_name)
#
#                             # Save the cropped image raster
#                             with rasterio.open(output_file_path, "w", **output_profile) as dest:
#                                 dest.write(image_cropped)




# def combine_radd_alerts(self, sentinel_base_name):
#     """
#     Combines multiple RADD alert images for a given Sentinel-2 tile into a single image.
#
#     Args:
#         sentinel_base_name (str): The base name of the Sentinel-2 tile.
#
#     Returns:
#         str: File path of the combined RADD alert image.
#     """
#
#     radd_files = [f for f in os.listdir(self.radd_alert_path) if sentinel_base_name in f]
#     if not radd_files:
#         return None  # Return None if no corresponding RADD alert files found
#
#     # Initialize an array to hold combined data
#     combined_radd = None
#
#     for radd_file in radd_files:
#         radd_path = os.path.join(self.radd_alert_path, radd_file)
#         with rasterio.open(radd_path) as src:
#             radd_data = src.read(1)  # Assuming RADD alert data is in the first band
#
#             if combined_radd is None:
#                 # Initialize combined_radd with the shape and type of the first RADD file
#                 combined_radd = np.zeros_like(radd_data)
#
#             # Combine by taking the maximum value (useful for binary/categorical data)
#             combined_radd = np.maximum(combined_radd, radd_data)
#
#     # Save the combined RADD alert image
#     combined_radd_path = os.path.join(self.radd_alert_path, f'{sentinel_base_name}_combined_radd.tif')
#     with rasterio.open(radd_files[0]) as src:
#         profile = src.profile
#         with rasterio.open(combined_radd_path, 'w', **profile) as dst:
#             dst.write(combined_radd, 1)
#
#     return combined_radd_path






# """
# Write selected bands from Sentinel-2 imagery for each date to separate GeoTIFF stack files.
# """
# # Group files by Julian date
# date_grouped_files = defaultdict(list)
# for file in os.listdir(self.path):
#     if file.endswith('.tif'):
#         # Extract Julian date from filename
#         julian_date = file.split('.')[3][:7]  # Adjust the slicing if necessary
#         date_grouped_files[julian_date].append(file)
#
# for julian_date, files in date_grouped_files.items():
#     # Filter files for the specified bands
#     selected_files = [f for f in files if any(band in f for band in self.bands)]
#
#     if not selected_files:
#         continue  # Skip if no files match the criteria
#
#     # Read the first file to setup profile
#     src_image = rasterio.open(os.path.join(self.path, selected_files[0]), "r")
#
#     dst_profile = src_image.profile.copy()
#     dst_profile.update({
#         "driver": "GTiff",
#         "count": len(selected_files),
#         "height": src_image.height,
#         "width": src_image.width,
#         "transform": src_image.transform,
#         "crs": src_image.crs
#     })
#
#     output_file_path = os.path.join(self.stack_path_list, f'{julian_date}_stack.tif')
#     if not os.path.exists(os.path.dirname(output_file_path)):
#         os.makedirs(os.path.dirname(output_file_path))
#
#     with rasterio.open(output_file_path, 'w', **dst_profile) as dst:
#         for i, file in enumerate(selected_files, start=1):
#             with rasterio.open(os.path.join(self.path, file)) as src:
#                 data = src.read(1)  # Read the first band
#                 if write:
#                     dst.write(data, i)


#
# def resample_radd_alerts(self):
#     """
#     Resamples 'merged_radd_alerts.tif' to a 30m resolution.
#     """
#
#     # Open the merged RADD alerts image
#     merged_radd_path = os.path.join(self.radd_alert_path, 'merged_radd_alerts_qgis_int16_compressed.tif')
#     with rasterio.open(merged_radd_path) as merged_radd_dataset:
#         # Calculate the transform and dimensions for the new resolution
#         # Assuming the original resolution is 10m, so dividing by 3 to get roughly 30m
#         transform, width, height = calculate_default_transform(
#             merged_radd_dataset.crs, merged_radd_dataset.crs,
#             merged_radd_dataset.width, merged_radd_dataset.height,
#             *merged_radd_dataset.bounds,
#             dst_width=merged_radd_dataset.width // 3,
#             dst_height=merged_radd_dataset.height // 3
#         )
#
#         # Define the metadata for the resampled dataset
#         out_meta = merged_radd_dataset.meta.copy()
#         out_meta.update({
#             "driver": "GTiff",
#             "height": height,
#             "width": width,
#             "transform": transform,
#             "crs": merged_radd_dataset.crs
#         })
#
#         # Perform the resampling
#         resampled_radd_path = os.path.join(self.radd_alert_path, 'resampled_merged_radd_alerts_qgis_int16_compressed_30m.tif')
#         with rasterio.open(resampled_radd_path, 'w', **out_meta) as dest:
#             reproject(
#                 source=rasterio.band(merged_radd_dataset, 1),
#                 destination=rasterio.band(dest, 1),
#                 src_transform=merged_radd_dataset.transform,
#                 src_crs=merged_radd_dataset.crs,
#                 dst_transform=transform,
#                 dst_crs=merged_radd_dataset.crs,
#                 resampling=Resampling.nearest
#             )
#
#     return resampled_radd_path


#
# def resample_radd_alerts(self):
#     """
#     Resamples 'merged_radd_alerts.tif' to match the resolution of a Sentinel-2 image.
#     """
#
#     # Use the first Sentinel-2 image to determine the target resolution
#     sentinel_files = [f for f in os.listdir(self.sentinel2_path) if f.endswith('.tif')]
#     if not sentinel_files:
#         raise ValueError("No Sentinel-2 images found in the specified path.")
#
#     with rasterio.open(os.path.join(self.sentinel2_path, sentinel_files[0])) as sentinel_dataset:
#         sentinel_transform = sentinel_dataset.transform
#         sentinel_crs = sentinel_dataset.crs
#
#     # Open the merged RADD alerts image
#     merged_radd_path = os.path.join(self.radd_alert_path, 'merged_radd_alerts.tif')
#     with rasterio.open(merged_radd_path) as merged_radd_dataset:
#         # Calculate the transform and dimensions for the new resolution
#         transform, width, height = calculate_default_transform(
#             merged_radd_dataset.crs, sentinel_crs,
#             merged_radd_dataset.width, merged_radd_dataset.height,
#             *merged_radd_dataset.bounds,
#             dst_transform=sentinel_transform
#         )
#
#         # Define the metadata for the resampled dataset
#         out_meta = merged_radd_dataset.meta.copy()
#         out_meta.update({
#             "driver": "GTiff",
#             "height": height,
#             "width": width,
#             "transform": transform,
#             "crs": sentinel_crs
#         })
#
#         # Perform the resampling
#         resampled_radd_path = os.path.join(self.radd_alert_path, 'resampled_radd_alerts.tif')
#         with rasterio.open(resampled_radd_path, 'w', **out_meta) as dest:
#             for i in range(1, merged_radd_dataset.count + 1):
#                 reproject(
#                     source=rasterio.band(merged_radd_dataset, i),
#                     destination=rasterio.band(dest, i),
#                     src_transform=merged_radd_dataset.transform,
#                     src_crs=merged_radd_dataset.crs,
#                     dst_transform=transform,
#                     dst_crs=sentinel_crs,
#                     resampling=Resampling.nearest
#                 )
#
#     return resampled_radd_path


#
# def stitch_radd_alerts(self):
#     """
#     Stitches all RADD alert images into a single large image.
#     """
#
#     # List to store each RADD alert dataset
#     radd_datasets = []
#
#     for radd_file in os.listdir(self.radd_alert_path):
#         if radd_file.endswith('.tif'):
#             radd_path = os.path.join(self.radd_alert_path, radd_file)
#             with rasterio.open(radd_path) as src:
#                 radd_datasets.append(src.read())
#
#     # Assuming all arrays have the same shape and two bands
#     merged_radd_band1 = np.zeros_like(radd_datasets[0][0])
#     merged_radd_band2 = np.zeros_like(radd_datasets[0][1])
#
#     for radd_data in radd_datasets:
#         # Merge band 1
#         merged_radd_band1 = np.maximum(merged_radd_band1, radd_data[0])
#         # Merge band 2
#         merged_radd_band2 = np.maximum(merged_radd_band2, radd_data[1])
#
#     # Create a new dataset for the merged data
#     merged_radd_path = os.path.join(self.radd_alert_path, 'merged_radd_alerts.tif')
#     with rasterio.open(radd_datasets[0].name, 'r') as src:
#         out_meta = src.meta.copy()
#         out_meta.update({"count": 2})
#         with rasterio.open(merged_radd_path, 'w', **out_meta) as dest:
#             dest.write(merged_radd_band1, 1)
#             dest.write(merged_radd_band2, 2)
#
#     return merged_radd_path

# def stitch_radd_alerts(self):
#     """
#     Stitches all RADD alert images into a single large image, handling multiple bands.
#     """
#
#     radd_datasets = [rasterio.open(os.path.join(self.radd_alert_path, f)) for f in os.listdir(self.radd_alert_path) if f.endswith('.tif')]
#
#     if not radd_datasets:
#         raise ValueError("No RADD alert images found.")
#
#     num_bands = radd_datasets[0].count
#     merged_transform, out_meta = None, None
#
#     # Prepare an empty list to store merged data for each band
#     merged_data = [None] * num_bands
#
#     for band in range(1, num_bands + 1):
#         sources = [d.read(band, masked=True) for d in radd_datasets]
#         merged_band_data, merged_transform = merge(sources)
#         merged_data[band - 1] = merged_band_data
#
#     out_meta = radd_datasets[0].meta.copy()
#     out_meta.update({
#         "driver": "GTiff",
#         "height": merged_data[0].shape[1],
#         "width": merged_data[0].shape[2],
#         "transform": merged_transform,
#         "count": num_bands
#     })
#
#     merged_radd_path = os.path.join(self.radd_alert_path, 'merged_radd_alerts.tif')
#     with rasterio.open(merged_radd_path, 'w', **out_meta) as dest:
#         for i, data in enumerate(merged_data, start=1):
#             dest.write(data.squeeze(), i)
#
#     # Close all open datasets
#     for ds in radd_datasets:
#         ds.close()
#
#     return merged_radd_path


# def stitch_radd_alerts(self):
#     """
#     Stitches all RADD alert images into a single large image.
#     """
#
#     # List to store each RADD alert dataset
#     radd_datasets = []
#
#     for radd_file in os.listdir(self.radd_alert_path):
#         if radd_file.endswith('.tif'):
#             radd_path = os.path.join(self.radd_alert_path, radd_file)
#             radd_datasets.append(rasterio.open(radd_path).read())
#     # Merge all the RADD alert datasets
#     merged_radd, merged_transform = merge(radd_datasets)
#
#     # Define the metadata for the new, merged dataset
#     out_meta = radd_datasets[0].meta.copy()
#     out_meta.update({
#         "driver": "GTiff",
#         "height": merged_radd.shape[1],
#         "width": merged_radd.shape[2],
#         "transform": merged_transform
#     })
#
#     # Write the merged RADD alert image to a new file
#     merged_radd_path = os.path.join(self.radd_alert_path, 'merged_radd_alerts.tif')
#     with rasterio.open(merged_radd_path, 'w', **out_meta) as dest:
#         dest.write(merged_radd)
#
#     # Close all open datasets
#     for ds in radd_datasets:
#         ds.close()
#
#     return merged_radd_path

def stitch_radd_alerts(self):
    """
    Stitches all RADD alert images into a single large image.
    """

    # List to store each RADD alert dataset
    radd_datasets = []
    first_meta = None  # Variable to store the metadata of the first dataset

    for radd_file in os.listdir(self.radd_alert_path):
        if radd_file.endswith('.tif'):
            radd_path = os.path.join(self.radd_alert_path, radd_file)
            with rasterio.open(radd_path) as src:
                if first_meta is None:
                    first_meta = src.meta  # Save the metadata of the first dataset
                radd_datasets.append(src.read())

    # Initialize merged bands
    merged_radd_band1 = np.zeros_like(radd_datasets[0][0])
    merged_radd_band2 = np.zeros_like(radd_datasets[0][1])

    for radd_data in radd_datasets:
        # Merge band 1
        merged_radd_band1 = np.maximum(merged_radd_band1, radd_data[0])
        # Merge band 2
        merged_radd_band2 = np.maximum(merged_radd_band2, radd_data[1])

    # Create a new dataset for the merged data
    merged_radd_path = os.path.join(self.radd_alert_path, 'merged_radd_alerts.tif')
    out_meta = first_meta.copy()
    out_meta.update({"count": 2})  # Update metadata for 2 bands
    with rasterio.open(merged_radd_path, 'w', **out_meta) as dest:
        dest.write(merged_radd_band1, 1)
        dest.write(merged_radd_band2, 2)

    return merged_radd_path


