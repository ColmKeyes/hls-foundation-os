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

import subprocess
import os

base_ckpt_path = r"E:\PycharmProjects\hls-foundation-os"
base_output_path = r"E:\PycharmProjects\hls-foundation-os\test_image_results"
base_config_path = r"E:\hls-foundation-os\configs"
base_input_path = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\globalnorm"

model_config_pairs = [
    # ("Prithvi-100m/best_mIoU_iter_400_minalerts_15000_prithvi_final_run1.pth","forest_disturbances_config.py"),
    # ("Prithvi-100m_unet/best_mIoU_iter_900_minalerts_15000_unet_final_run1.pth", "forest_disturbances_config_unet.py"),
    # ("Prithvi-100m/best_mIoU_iter_500_minalerts_10000_prithvi_final_run1.pth", "forest_disturbances_config.py"),
    # ("Prithvi-100m_unet/best_mIoU_iter_1000_minalerts_15000_unet_final_run2.pth","forest_disturbances_config_unet.py"),
     # ("Prithvi-100m_burnscars/best_mIoU_iter_500_minalerts_10000_prithvi_burnscars_final_run1.pth", "forest_disturbances_config_burnscars.py"),
    # ("Prithvi-100m/best_mIoU_iter_500_minalerts_10000_prithvi_final_run1.pth", "forest_disturbances_config.py"),
    # ("Prithvi-100m/best_mIoU_iter_500_minalerts_15000_prithvi_final_run3.pth", "forest_disturbances_config.py")
    # ("Prithvi-100m/best_mIoU_iter_400_minalerts_15000_prithvi_final_run1.pth", "forest_disturbances_config.py"),
    # ("Prithvi-100m/best_mIoU_iter_500_minalerts_15000_prithvi_final_run2.pth", "forest_disturbances_config.py")
    # ("Prithvi-100m_burnscars/best_mIoU_iter_500_minalerts_15000_prithvi_burnscars_final_run2.pth", "forest_disturbances_config_burnscars.py"),
    # ("Prithvi-100m_burnscars/best_mIoU_iter_900_minalerts_12500_prithvi_burnscars_final_run1.pth", "forest_disturbances_config_burnscars.py"),
    # ("Prithvi-100m/best_mIoU_iter_60_minalerts_15000_prithvi_final_run1.pth", "forest_disturbances_config.py")




    # ("Prithvi-100m/best_mIoU_iter_500_minalerts_10000_prithvi_final_run1_op.pth","forest_disturbances_config.py"),
    # ("Prithvi-100m/best_mIoU_iter_400_minalerts_10000_prithvi_final_run2_op.pth","forest_disturbances_config.py"),
    # ("Prithvi-100m/best_mIoU_iter_500_minalerts_10000_prithvi_final_run3_op.pth","forest_disturbances_config.py"),
    # ("Prithvi-100m/best_mIoU_iter_500_minalerts_15000_prithvi_final_run1_op.pth","forest_disturbances_config.py"),
    # ("Prithvi-100m/best_mIoU_iter_500_minalerts_15000_prithvi_final_run2_op.pth","forest_disturbances_config.py"),
    # ("Prithvi-100m/best_mIoU_iter_400_minalerts_15000_prithvi_final_run3_op.pth","forest_disturbances_config.py"),
    # ("Prithvi-100m_burnscars/best_mIoU_iter_400_minalerts_15000_prithvi_burnscars_final_run1_op.pth","forest_disturbances_config_burnscars.py"),
    # ("Prithvi-100m_burnscars/best_mIoU_iter_300_minalerts_15000_prithvi_burnscars_final_run2_op.pth","forest_disturbances_config_burnscars.py"),
    # ("Prithvi-100m_burnscars/best_mIoU_iter_500_minalerts_15000_prithvi_burnscars_final_run3_op.pth","forest_disturbances_config_burnscars.py"),
    # ("Prithvi-100m_unet/best_mIoU_iter_500_minalerts_15000_unet_final_run1_op.pth",  "forest_disturbances_config_unet.py"),
    #  ("Prithvi-100m_unet/best_mIoU_iter_500_minalerts_15000_unet_final_run2_op.pth","forest_disturbances_config_unet.py"),
    #   ("Prithvi-100m_unet/best_mIoU_iter_500_minalerts_15000_unet_final_run3_op.pth", "forest_disturbances_config_unet.py"),
    #    ("Prithvi-100m_unet/best_mIoU_iter_500_minalerts_10000_unet_final_run1_op.pth", "forest_disturbances_config_unet.py"),
    #     ("Prithvi-100m_unet/best_mIoU_iter_500_minalerts_10000_unet_final_run2_op.pth", "forest_disturbances_config_unet.py"),
    # ("Prithvi-100m_unet/best_mIoU_iter_500_minalerts_10000_unet_final_run3_op.pth", "forest_disturbances_config_unet.py"),

    ("Prithvi-100m_coherence/best_mIoU_iter_400_minalerts_15000_prithvi_coherence_final_run1_op.pth", "forest_disturbances_config_coherence.py"),
    ("Prithvi-100m_coherence/best_mIoU_iter_400_minalerts_15000_prithvi_coherence_final_run2_op.pth", "forest_disturbances_config_coherence.py"),
    ("Prithvi-100m_coherence/best_mIoU_iter_500_minalerts_15000_prithvi_coherence_final_run3_op.pth", "forest_disturbances_config_coherence.py"),
    ("Prithvi-100m_backscatter/best_mIoU_iter_500_minalerts_15000_prithvi_backscatter_final_run1_op.pth", "forest_disturbances_config_backscatter.py"),
    ("Prithvi-100m_backscatter/best_mIoU_iter_500_minalerts_15000_prithvi_backscatter_final_run2_op.pth", "forest_disturbances_config_backscatter.py"),
    ("Prithvi-100m_backscatter/best_mIoU_iter_500_minalerts_15000_prithvi_backscatter_final_run3_op.pth", "forest_disturbances_config_backscatter.py"),

    ("Prithvi-100m_unet_coherence/best_mIoU_iter_500_minalerts_15000_unet_coherence_final_run1_op.pth", "forest_disturbances_config_unet_coherence.py"),
    ("Prithvi-100m_unet_coherence/best_mIoU_iter_500_minalerts_15000_unet_coherence_final_run1_op.pth", "forest_disturbances_config_unet_coherence.py"),
    ("Prithvi-100m_unet_coherence/best_mIoU_iter_500_minalerts_15000_unet_coherence_final_run1_op.pth", "forest_disturbances_config_unet_coherence.py"),
    ("Prithvi-100m_unet_backscatter/best_mIoU_iter_500_minalerts_15000_unet_backscatter_final_run1_op.pth", "forest_disturbances_config_unet_backscatter.py"),
    ("Prithvi-100m_unet_backscatter/best_mIoU_iter_400_minalerts_15000_unet_backscatter_final_run1_op.pth", "forest_disturbances_config_unet_backscatter.py"),
    ("Prithvi-100m_unet_backscatter/best_mIoU_iter_500_minalerts_15000_unet_backscatter_final_run1_op.pth", "forest_disturbances_config_unet_backscatter.py"),

]

input_type = "tif"
bands = "[0,1,2,3,4,5]"
out_channels = 1

for model, config in model_config_pairs:
    # Construct checkpoint path
    ckpt_path = os.path.join(base_ckpt_path, model)

    # Select input path based on model type
    if "backscatter" in model:
        input_path = os.path.join(base_input_path, "15000_minalerts_backscatter/test/")
    elif "coherence" in model:
        input_path = os.path.join(base_input_path, "15000_minalerts_coherence/test/")
    else:
        input_path = os.path.join(base_input_path, "15000_minalerts/test/")

    # Define config path
    config_path = os.path.join(base_config_path, config)

    # Create a specific output directory for each model
    model_output_path = os.path.join(base_output_path, os.path.splitext(model)[0])
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    # Construct the command
    command = [
        'python', 'model_inference.py',
        '-config', config_path,
        '-ckpt', ckpt_path,
        '-input', input_path,
        '-output', model_output_path,
        '-input_type', input_type,
        '-bands', bands,
    ]

    # Run the command
    subprocess.run(command)