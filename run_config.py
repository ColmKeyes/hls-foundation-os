import os
from mmseg.apis import train_segmentor, init_segmentor, inference_segmentor
from mmcv import Config

# Update the config file
cfg_path = "E:\\PycharmProjects\\hls-foundation-os\\configs\\burn_scars.py"
cfg = Config.fromfile(cfg_path)

# Update data root directory
cfg.data_root = "/path/to/your/data/root/directory"

# Update pre-trained weights path
cfg.pretrained_weights_path = "/path/to/your/pretrained/weights"

# Update experiment name
cfg.experiment = "my_experiment"

# Update project directory
cfg.project_dir = "/path/to/your/project/directory"

# Update number of samples per GPU
cfg.samples_per_gpu = 4

# Update image normalization config
cfg.img_norm_cfg = dict(
    means=[0.485, 0.456, 0.406],
    stds=[0.229, 0.224, 0.225],
)

# Update bands
cfg.bands = [0, 1, 2]

# Update classes
cfg.CLASSES = ("Unburnt land", "Burn scar")

# Update optimizer settings
cfg.optimizer = dict(type="Adam", lr=1e-4)

# Update learning rate scheduler config
cfg.lr_config = dict(
    policy="poly",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
    power=1.0,
)

# Save the updated config file
cfg.dump(os.path.join(cfg.project_dir, cfg.experiment, "updated_config.py"))

# Initialize and train the segmentor
model = init_segmentor(cfg, device='cuda:0')
train_segmentor(model, cfg.data.train)

# For inference, you can use the following
# results = inference_segmentor(model, "/path/to/your/test/image")



