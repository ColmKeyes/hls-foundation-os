set PL_TORCH_DISTRIBUTED_BACKEND = gloo
set NCCL_DEBUG=INFO

mim train mmsegmentation --help
mim train mmsegmentation --launcher none --gpus 1 configs/burn_scars.py
mim train mmsegmentation --launcher pytorch --gpus 1 configs/burn_scars.py
mim train mmsegmentation --launcher none --gpu 0 configs/burn_scars.py



python model_inference.py -config "C:/Users/Lord Colm/PycharmProjects/hls-foundation-os/configs/burn_scars.py" -ckpt "E:/PycharmProjects/hls-fo
undation-os/experiment 1/best_mIoU_iter_900.pth" -input "C:/Users/Lord Colm/Downloads/S2B_MSIL2A_20230828T210929_N0509_R057_T04QGJ_20230828T230535.SAFE/GRANULE/L2A_T04QGJ_A033831_20230828T210926/IMG_D
ATA/" -output "C:/Users/Lord Colm/Downloads/S2B_MSIL2A_20230828T210929_N0509_R057_T04QGJ_20230828T230535.SAFE/GRANULE/L2A_T04QGJ_A033831_20230828T210926/IMG_DATA/" -input_type tif -bands "[0,1,2,3,4,5
]" -out_channels = 1



