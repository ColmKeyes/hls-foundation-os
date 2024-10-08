# -*- coding: utf-8 -*-
"""
This configuration script sets up the parameters for running the Prithvi-100m model, specifically tailored for burn scar detection using Sentinel-2 imagery. Adapted from the original Prithvi model code.
"""
"""
@Time    : [Time of Creation, e.g., 07/12/2023 10:30]
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : burn_scars_config
"""

import os

custom_imports = dict(imports=["geospatial_fm"])

# base options
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
cudnn_benchmark = True

dataset_type = "GeospatialDataset"

# TO BE DEFINED BY USER: data directory ok
data_root = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\globalnorm\15000_minalerts_backscatter"

num_frames = 1
img_size = 224
num_workers = 1
samples_per_gpu = 18
img_norm_cfg = dict(
# means = [1, 1,1,1,1,1],
# stds = [.1,.1,.1,.1,.1,.1]
# )
# means = [0.01317752, 0.02213574 ,0.0166708,  0.11811387, 0.03907045 ,0.01535661],
# stds= [0.01072161, 0.01190746, 0.01725778, 0.0295262 , 0.02203361 ,0.01510126])

means = [0.11349006921130662, 0.18013922190434992, 0.09824706480197842, 0.5221330166878408, 0.03025329, 0.07961972], #0.31299707170569835, 0.15978379676331667],
stds = [0.073205091376172, 0.08175555163630563, 0.07270076782812802, 0.11422985057904868,0.01407666, 0.03714384] #0.10197667925518768, 0.08313405470923486]
)

# COH:
# Global means after normalization: [0.3583233  0.34587905]
# Global standard deviations after normalization: [0.16327385 0.15160857]
#
# BSC:
# Global means after normalization: [0.03025329 0.07961972]
# Global standard deviations after normalization: [0.01407666 0.03714384]


#means=[190.25926138433516, 429.89401106101735, 263.64892182260917, 2914.742437197362, 1436.1499327360475, 552.0405896092034],
         #stds=[147.77000967917303, 181.20171057982083, 215.94822142698277, 560.9504799450765, 380.71293248398223, 229.565368633047])




    # means=[
    #     0.033349706741586264,
    #     0.05701185520536176,
    #     0.05889748132001316,
    #     0.2323245113436119,
    #     0.1972854853760658,
    #     0.11944914225186566,
    # ],
    # stds=[
    #     0.02269135568823774,
    #     0.026807560223070237,
    #     0.04004109844362779,
    #     0.07791732423672691,
    #     0.08708738838140137,
    #     0.07241979477437814,
    # ],)  # change the mean and std of all the bands

#####
##
#####
bands = [0, 1, 2, 3, 4, 5]
tile_size = 224
orig_nsize = 512
crop_size = (tile_size, tile_size)
img_suffix = "_sentinel_agb_normalized_bsc_masked_normalized.tif"#"_sentinel_agb_normalized.tif"  #"radd_modified_sentinel_normalized_agb.tif"
seg_map_suffix = "_radd_labelled_agb.tif" # "radd_modified_radd_agb.tif"
ignore_index = -1
label_nodata = -9999
image_nodata = -9999
image_nodata_replace = -1
image_to_float32 = True

# model
# TO BE DEFINED BY USER: model path
pretrained_weights_path = r"E:\burn_scars_Prithvi_100M_reset.pth" ## burn_scars_Prithvi_100M.pth#E:\Prithvi_100M.pt
num_layers = 12
patch_size = 16
embed_dim = 768
num_heads = 12
tubelet_size = 1
output_embed_dim = num_frames*embed_dim
max_intervals = 500
evaluation_interval = 100

# TO BE DEFINED BY USER: model path
experiment = "Prithvi-100m_backscatter"
project_dir = "E:\PycharmProjects\hls-foundation-os"
work_dir = os.path.join(project_dir, experiment)
save_path = work_dir

save_path = work_dir
train_pipeline = [
    dict(
        type="LoadGeospatialImageFromFile",
        to_float32=image_to_float32,
        channels_last=True,
        nodata = image_nodata,
        nodata_replace = image_nodata_replace
),
    dict(type="LoadGeospatialAnnotations",
        reduce_zero_label=False,
        nodata=label_nodata,
        nodata_replace=ignore_index),
    dict(type="BandsExtract", bands=bands),
    dict(type="RandomFlip", prob=0.5),
    dict(type="ToTensor", keys=["img", "gt_semantic_seg"]),
    # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(type="TorchRandomCrop", crop_size=(tile_size, tile_size)),
    dict(
        type="Reshape_prithvi",
        keys=["img"],
        new_shape=(
            len(bands),
            num_frames,
            tile_size,
            tile_size
        )
    ),
    dict(
        type="Reshape_prithvi",
        keys=["gt_semantic_seg"],
        new_shape=(1, tile_size, tile_size)
    ),
    dict(
        type="CastTensor",
        keys=["gt_semantic_seg"],
        new_type="torch.LongTensor"
    ),
    dict(type="Collect", keys=["img", "gt_semantic_seg"])
]
test_pipeline = [
    dict(
        type="LoadGeospatialImageFromFile",
        to_float32=image_to_float32,
        channels_last=True,
        nodata=image_nodata,   #  so, there is a discrepency in the indexes utilised in the test pipeline and in the test dataset. values are set differently to be 0 and ignore -1.
        nodata_replace=ignore_index # Needs to be the same as the test_data ignore index!!!
    ),
    dict(type="BandsExtract", bands=bands),
    dict(type="ToTensor", keys=["img"]),
    # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(
        type="Reshape_prithvi",
        keys=["img"],
        new_shape=(len(bands),
                   num_frames,
                   -1,
                   -1),
        look_up=dict({
            "2": 1,
            "3": 2
        })),
    dict(type="CastTensor", keys=["img"], new_type="torch.FloatTensor"),
    dict(
        type="CollectTestList",
        keys=["img"],
        meta_keys=[
            "img_info",
            "seg_fields",
            "img_prefix",
            "seg_prefix",
            "filename",
            "ori_filename",
            "img",
            "img_shape",
            "ori_shape",
            "pad_shape",
            "scale_factor"
            ,"img_norm_cfg"
        ]
    )
]

CLASSES = ("Forest", "Disturbed_Forest")

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=num_workers,
    train=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir="train",#r"training_narrow\test",
        ann_dir="train",#r"training_narrow\test",
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=train_pipeline,
        ignore_index=ignore_index),
    val=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir="val",#r"validation_narrow\test",
        ann_dir="val", #r"validation_narrow\test",
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=test_pipeline,
        ignore_index=ignore_index,
        gt_seg_map_loader_cfg=dict(nodata=label_nodata, nodata_replace=ignore_index)
    ),

    test=dict(
        type=dataset_type,
        CLASSES=CLASSES,
        data_root=data_root,
        img_dir="test",#r"inference_test\best_sample",#"test",
        ann_dir="test",#r"inference_test\best_sample",
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=test_pipeline,
        ignore_index=ignore_index,
gt_seg_map_loader_cfg = dict(nodata=label_nodata, nodata_replace=ignore_index)

)
)

optimizer = dict(type="Adam", lr=1.3e-05, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="poly",
    warmup="linear",
    warmup_iters=100,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)
log_config = dict(
    interval=20,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=True),
        dict(type="TensorboardLoggerHook", by_epoch=True)
    ]
)

custom_hooks = [dict(
    type='PlotIterationHook',
    interval=10)
]

checkpoint_config = dict(
    by_epoch=True,
    interval=10,
    out_dir=save_path
)
evaluation = dict(
    interval=evaluation_interval,
    metric="mIoU",
    pre_eval=True,
    save_best="mIoU",
    by_epoch=True
)

loss_func = dict(
    ## So these values below are compleetly different to what we see as potential variables in
    ## the dice_loss mmseg Docs...

    type="FocalLoss",
    use_sigmoid=True,
    gamma=1.0,  # Decreasing gamma to reduce focus on hard-to-classify examples
    alpha=0.5,  # Balanced focus on both classes
    reduction='mean',
    class_weight=[1, 100],  # Less aggressive weighting on the disturbed class
    loss_weight=1.0,
    loss_name='loss_focal'
)

#     type="FocalLoss",
#     use_sigmoid=True,
#     gamma=5.0,  # Increasing gamma to focus more on hard-to-classify examples
#     alpha=0.25,  # Shift more focus to the disturbed class
#     reduction='mean',
#     class_weight=[1, 20000],  # Increase weight for the disturbed class
#     loss_weight=1.0,
#     loss_name='loss_focal'
# )
  # Slightly more importance to the positive class
    #     reduction='mean',
    #     class_weight=[1, 100],  # Less extreme class weighting
    #     loss_weight=1.0,
    #     loss_name='loss_focal'
    # )


    # type="DiceLoss",
    # use_sigmoid=False,
    # class_weight=[1,200],
    # loss_weight=1,
    # ignore_index=-1)



runner = dict(type="IterBasedRunner", max_iters=max_intervals)#, custom_hooks= custom_hooks)
workflow = [("train", 1)]
norm_cfg = dict(type="BN", requires_grad=True)
model = dict(
    type="TemporalEncoderDecoder",
    frozen_backbone=True, #False,
    backbone=dict(
        type="TemporalViTEncoder",
        pretrained=pretrained_weights_path,
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans=len(bands),
        embed_dim=embed_dim,
        depth=12,
        num_heads=num_heads,
        mlp_ratio=4.0,
        norm_pix_loss=False
    ),
    neck=dict(
        type="ConvTransformerTokensToEmbeddingNeck",
        embed_dim=embed_dim*num_frames,
        output_embed_dim=output_embed_dim,
        drop_cls_token=True,
        Hp=14,
        Wp=14
    ),
    decode_head=dict(
        num_classes=len(CLASSES),
        in_channels=output_embed_dim,
        type="FCNHead",
        in_index=-1,
        ignore_index=ignore_index,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=loss_func
    ),
    auxiliary_head=dict(
        num_classes=len(CLASSES),
        in_channels=output_embed_dim,
        type="FCNHead",
        in_index=-1,
        ignore_index=ignore_index,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=loss_func
    ),
    train_cfg=dict(),
    test_cfg=dict(
        mode="slide",
        stride=(int(tile_size / 2), int(tile_size / 2)),
        crop_size=(tile_size, tile_size),
    ),


)
gpu_ids = range(0, 1)
auto_resume = False