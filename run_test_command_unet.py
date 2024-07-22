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

import subprocess
import os

models = [
# "best_mIoU_iter_500_minalerts_15000_unet_final_run1_op.pth",
# "best_mIoU_iter_500_minalerts_15000_unet_final_run2_op.pth",
# "best_mIoU_iter_500_minalerts_15000_unet_final_run3_op.pth",
# "best_mIoU_iter_500_minalerts_10000_unet_final_run1_op.pth",
# "best_mIoU_iter_500_minalerts_10000_unet_final_run2_op.pth",
# "best_mIoU_iter_500_minalerts_10000_unet_final_run3_op.pth",
# "best_mIoU_iter_500_minalerts_15000_unet_coherence_final_run1_op.pth",
# "best_mIoU_iter_500_minalerts_15000_unet_coherence_final_run2_op.pth",
# "best_mIoU_iter_500_minalerts_15000_unet_coherence_final_run3_op.pth",
#     "best_mIoU_iter_500_minalerts_15000_unet_backscatter_final_run1_op.pth",
#     "best_mIoU_iter_400_minalerts_15000_unet_backscatter_final_run2_op.pth"
#     "best_mIoU_iter_500_minalerts_15000_unet_backscatter_final_run3_op.pth"
    "best_mIoU_iter_500_minalerts_12500_unet_final_run1_op.pth",
    "best_mIoU_iter_500_minalerts_12500_unet_final_run2_op.pth",
    "best_mIoU_iter_500_minalerts_12500_unet_final_run3_op.pth",
]

base_ckpt_path = r"E:\PycharmProjects\hls-foundation-os\Prithvi-100m_unet"
base_log_dir = r"E:\PycharmProjects\hls-foundation-os\test_logs"
config_path = r"E:\hls-foundation-os\configs\forest_disturbances_config_unet.py"

for model in models:
    # Construct checkpoint path
    checkpoint_path = os.path.join(base_ckpt_path, model)

    # Create a specific log directory for each model
    model_log_dir = os.path.join(base_log_dir, os.path.splitext(model)[0])
    if not os.path.exists(model_log_dir):
        os.makedirs(model_log_dir)

    # Define the command to run the mmseg testing script for the current model
    command = [
        'python',
        r'c:\anaconda3\envs\prithvi-100m\lib\site-packages\mmseg\.mim\tools\test.py',
        config_path,
        checkpoint_path,
        '--out', os.path.join(model_log_dir, 'results.pkl'),
        '--eval', 'mIoU',
        '--work-dir', model_log_dir,
        '--launcher', 'none'
    ]

    # Execute the command
    print(f"Testing with model: {model}")
    subprocess.run(command)
    print(f"Finished testing with model: {model}\n")

