import subprocess

# Command and arguments
command = "mim"
args = ["train", "mmsegmentation", "--launcher", "none", "--gpus", "1", "E:\hls-foundation-os\configs/forest_distrubances_config.py"]

# Running the command
subprocess.run([command] + args)