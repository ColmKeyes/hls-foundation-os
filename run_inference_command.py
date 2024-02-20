import subprocess

# Define command parameters
config_path = "E:\hls-foundation-os\configs/forest_distrubances_config.py" #burn_scars_config.py"
ckpt_path = r"E:\PycharmProjects\hls-foundation-os\Prithvi-100m\best_mIoU_iter_1000.pth"#"E:/burn_scars_Prithvi_100M.pth"
input_path = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\inference_test\\"#burn_scars\\"
output_path = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\inference_test"
input_type = "tif"
bands = "[0,1,2,3,4,5]"
out_channels = 1

# Construct the command
command = [
    'python', 'model_inference.py',
    '-config', config_path,
    '-ckpt', ckpt_path,
    '-input', input_path,
    '-output', output_path,
    '-input_type', input_type,
    '-bands', bands,
    # '-out_channels', str(out_channels)
]

# Run the command
subprocess.run(command)