# model_analysis.py

# model_analysis.py

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Arrow
import matplotlib.patheffects as PathEffects
from rasterio.plot import show
from pyproj import Proj, transform


color_maps = {
    '10000': ['#E69F00', '#F0E442', '#56B4E9'],  # Shades for 10000 minalerts
    '12500': ['#009E73', '#8DD3C7', '#0072B2'],  # Shades for 12500 minalerts
    '15000': ['#D55E00', '#CC79A7', '#999999'],  # Shades for 15000 minalerts
}



class ModelAnalysis:
    def __init__(self, log_file):
        self.log_file = log_file
        self.data = self.load_log_data()

    def load_log_data(self):
        """Load the log data from the JSON file."""
        data = []
        with open(self.log_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                # Ensure the entry contains 'epoch' and 'mode' keys and mode is 'val'
                if 'epoch' in entry and 'mode' in entry and entry['mode'] == 'val':
                    data.append(entry)
        return data

    def extract_metric(self, metric_name):
        """Extract a specific metric across epochs."""
        return [entry[metric_name] for entry in self.data if metric_name in entry]

    def get_epochs(self):
        """Extract epoch numbers."""
        return [entry['epoch'] for entry in self.data]


    def plot_metric(self, metric_name, ax, label):
        """Plot a specific metric."""
        values = self.extract_metric(metric_name)
        epochs = self.get_epochs()
        ax.plot(epochs, values, label=label)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.legend()
#



def plot_image_triplet(input_path, output_path, ground_truth_path, image_basename, data_type, save_path):

    unnormalised_image_path = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc" # for printing non-normalised rgb.
    suffix = ''
    if data_type == 'backscatter':
        suffix = '_bsc_masked_normalized'
    elif data_type == 'coherence':
        suffix = '_coh_masked_normalized'


    run_number = ''
    for part in save_path.split('_'):
        if part.startswith('run'):
            run_number = part
            break

    inProj = Proj(init='epsg:32650')
    outProj = Proj(init='epsg:4326')
    # Transform the corners of the image to lat/long
    left, bottom = transform(inProj, outProj, 230536.5095550618, -107359.60842863785)
    right, top = transform(inProj, outProj, 245873.9621862365, -92022.15579746317)


    input_image = os.path.join(input_path, f"{image_basename}_sentinel_agb_normalized{suffix}.tif")
    unnormalised_image = os.path.join(unnormalised_image_path, f"{image_basename}_sentinel.tif")
    output_image = os.path.join(output_path, f"{image_basename}_sentinel_agb_normalized{suffix}_pred_{run_number}_op.tif")
    ground_truth_image = os.path.join(ground_truth_path, f"{image_basename}_radd_labelled_agb.tif")

    class_cmap = ListedColormap(['lightblue', 'orange', 'grey'])
    class_cmap_gt = ListedColormap(['lightblue', 'red', 'grey'])

    difference_cmap = ListedColormap(['orange', "lightblue", 'grey'])
    sar_map = plt.cm.gray

    colorlist = ["grey","lightblue",  "red", "blue", "orange"]  # Colors for NoData, TN, FN, FP, TP
    cmap = ListedColormap(colorlist)
    leg_handles = [Line2D([0], [0], color=color, lw=4) for color in colorlist]
    leg_labels = [ "No Data","TN", "FN", "FP", "TP"]




    fig, axs = plt.subplots(1, 4, figsize=(30, 8), sharex=True, sharey=True)

    with rasterio.open(input_image) as src:
        if data_type in ["backscatter", "coherence"]:
            # For backscatter or coherence, create a composite image using band 5 and band 6
            band1 = src.read(5)
            band2 = src.read(6)

            # Create a composite image
            composite = np.stack((band1, band2, band1), axis=0)  # Stacking along the first axis (color channel)
            composite = composite.astype(float)
            # composite /= composite.max()  # Normalize to [0, 1] range
            p99 = np.percentile(composite, 99)  # Get the 99th percentile
            composite = np.clip(composite, 0, p99) / p99
            axs[0].imshow(composite.transpose(1, 2, 0), cmap="gray", extent=[left, right, bottom, top])  # Transpose for imshow
            axs[0].set_title(f'{data_type.capitalize()} Bands Composite', fontsize=28)
            # Display the composite image

        else:
            # Handle non-backscatter and non-coherence data types
            with rasterio.open(unnormalised_image) as src:

                input_rgb = src.read([1, 2, 3])
                # input_rgb = np.clip(input_rgb / 0.1, 0, 1)
                # max_value = np.nanmax(input_rgb)
                # input_rgb = input_rgb / max_value
                p99 = np.percentile(input_rgb, 99)
                input_rgb = np.clip(input_rgb, 0, p99) / p99

                axs[0].imshow(input_rgb.transpose(1, 2, 0),vmin=0, vmax=1, extent=[left, right, bottom, top])  #vmin=0, vmax=2, Display RGB image
                axs[0].set_title('Input RGB', fontsize=28)
    # with rasterio.open(input_image) as src:
    #     if data_type in ["backscatter", "coherence"]:
    #         # For backscatter or coherence, display band 5 and band 6 separately
    #         band1 = src.read(5)
    #         band2 = src.read(6)
    #         axs[0].imshow(band1, cmap='gray', extent=[left, right, bottom, top])
    #         axs[0].set_title(f'{data_type.capitalize()} Band 1')
    #     else:
    #         input_rgb = src.read([1, 2, 3])
    #         input_rgb = np.clip(input_rgb / 0.1, 0, 1)
    #         axs[0].imshow(input_rgb.transpose(1, 2, 0), extent=[left, right, bottom, top])
    #         axs[0].set_title('Input RGB')


    with rasterio.open(output_image) as src:
        output_data = src.read(1).astype(float)
        output_data[output_data == -1] = np.nan

    with rasterio.open(ground_truth_image) as src:
        gt_data = src.read(1).astype(float)
        gt_data[gt_data == -9999] = np.nan

    # Calculate TN,FN,TP,FP.
    cont = np.zeros_like(output_data, dtype=np.int16)  # Assuming output_data is already defined
    cont[output_data == -9999] = 0  # NoData
    cont[(gt_data == 0) & (output_data == 0)] = 1  # TN: True Negative
    cont[(gt_data == 1) & (output_data == 0)] = 2  # FN: False Negative
    cont[(gt_data == 0) & (output_data == 1)] = 3  # FP: False Positive
    cont[(gt_data == 1) & (output_data == 1)] = 4  # TP: True Positive



    difference_map = np.where(np.isnan(output_data) | np.isnan(gt_data), 2, 1 - np.abs(gt_data - output_data))

    # fig.suptitle('Prithvi 15k', fontsize=20)

    show_class_data(output_data, axs[1], class_cmap, 'Output U-Net', [left, right, bottom, top])
    show_class_data(gt_data, axs[2], class_cmap_gt, 'Labels', [left, right, bottom, top])


    axs[0].set_xlabel('Longitude', fontsize=28)
    axs[0].set_ylabel('Latitude', fontsize=28)
    axs[0].tick_params(axis='both', labelsize=16)  # Adjusting the size of the tick labels
    # axs[3].imshow(difference_map, cmap=difference_cmap, vmin=0, vmax=2, extent=[left, right, bottom, top])
    # axs[3].set_title('Difference Map', fontsize=28)
    # axs[3].set_xlabel('Longitude', fontsize=28)
    # axs[3].set_ylabel('Latitude', fontsize=28)
    # axs[3].tick_params(axis='both', labelsize=16)  # Adjusting the size of the tick labels
    axs[3].imshow(cont, cmap=cmap, vmin=0, vmax=4, interpolation="nearest", extent=[left, right, bottom, top])
    axs[3].set_title('Confusion Matrix Map', fontsize=24)
    axs[3].set_xlabel('Longitude', fontsize=28)
    axs[3].set_ylabel('Latitude', fontsize=28)
    axs[3].tick_params(axis='both', labelsize=16)
    # axs[3].legend(handles=leg_handles, labels=leg_labels, loc='upper right', fontsize=12)

    add_legend(axs[1], class_cmap, ['Non-Disturbed', 'Disturbed', 'No Data'], fontsize=22)
    add_legend(axs[2], class_cmap_gt, ['Non-Disturbed', 'Disturbed', 'No Data'],fontsize=22)
    # add_legend(axs[3], difference_cmap, ['Difference', 'No Difference', 'No Data'],fontsize=22)
    add_legend(axs[3], cmap, leg_labels, fontsize=22)

    output_filename = os.path.join(save_path, f"{image_basename}_comparison.png")
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()

    print(f"File saved at {output_filename}")

def show_class_data(data, ax, cmap, title, extent):
    mapped_data = np.full(data.shape, 2)  # Default to no data
    mapped_data[data == 0] = 0  # Non-disturbed
    mapped_data[data == 1] = 1  # Disturbed
    ax.imshow(mapped_data, cmap=cmap, extent=extent)
    ax.set_title(title, fontsize=28)
    ax.set_xlabel('Longitude', fontsize=28)
    ax.set_ylabel('Latitude', fontsize=28)
    ax.tick_params(axis='both', labelsize=16)
    ax.axis('on')

def add_legend(ax, cmap, labels, fontsize=14):  # Added fontsize parameter with a default value
    patches = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(len(labels))]
    ax.legend(handles=patches, labels=labels, loc='upper right', fontsize=fontsize)


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