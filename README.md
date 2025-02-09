# Modifications for Forest Disturbance Detection

#### This project, undertaken as part of an internship with VITO Remote Sensing, focuses on fine-tuning the Prithvi-100m model to determine its capabilities in forest disturbance detection, as well as the model's capacity to fine tune to unseen SAR and InSAR data. Writeup found under Internship____Colm_Keyes.pdf

## Overview
### This repository details the extension of the Prithvi-100m model, pre-trained on contiguous US data by teams at NASA and IBM. Thus much of the documentation below is kept from the original repo. Included in this research is model config, preprocessing and running and analysis of the Prithvi model along with a U-Net model used for comparison.

## Methods Flow Chart
<p align="center">
  <img src="Methods Flow Chart.png" alt="Image Description" width="100%">
</p>


## Results
### While the Prithvi model marginally outperformed a U-Net model for the task of forest disturbance detection, the performance of both models were limited by data quality. A significant result which appeared from this research was that the highest and second highest performing models were U-Net models utilising InSAR and SAR data respectively. This result is in agreement with research conducted on InSAR data in my MSc Thesis, which showed the potential of InSAR coherence measures for forest disturbance detection. 

## Performance Metrics
Performance evaluation results on unseen test data. Avg.: Average. Acc: Accuracy. mAcc: mean Accuracy mIoU: mean Intersection over Union.
<p align="center">
  <img src="Capture_prithvi_results.PNG" alt="Image Description" width="100%">
</p>

## Example Outputs

Below are sample comparison outputs between the prithvi model output and RADD alert labels, showing model inference on test data tile T49MDU, with input RGB, inference output, RADD labels, and confusion matrix map. 
<p align="center">
  <img src="2023076_T49MET_agb_radd_fmask_stack_512_512_comparison_Prithvi_backscatter.png" alt="Image Description" width="100%">
</p>
<p align="center">
  <img src="2023076_T49MET_agb_radd_fmask_stack_512_512_comparison_prithvi_coherence.png" alt="Image Description" width="100%">
</p>



## Project Repository Structure
This project utilises a source-bin folder structure, with functions found in source and processing/analysis in bin. 
Preprocessing steps are labelled file 1-3 for HLS and Sentienl-1 data. 
Model run commands are found in the main project directory.




<!--
## Usage Instructions


## Project Repository Structure
This project utilises a source-bin folder structure, with functions found in source and processing/analysis in bin. 
Preprocessing steps are labelled file 1-3 for HLS and Sentienl-1 data. 
Model run commands are found in the main project directory.

- **bin/**: Main scripts for data preprocessing and analysis
  - **data_preprocessing_hls**:
    - `1_hls_run_processing_prep.py`: Step 1 - Prepare HLS processing
    - `2_dataset_run_management.py`: Step 2 - Manage dataset runs for HLS
    - `3_model_run_input_processor.py`: Step 3 - Process inputs for model runs
  - **data_preprocessing_sar**:
    - `1_sentinel1slc_bsc_coh_processing.py`: Step 1 - Sentinel-1 SLC, BSC, and Coherence processing
    - `2_sar_model_run_input_processor.py`: Step 2 - Process inputs for SAR model runs
    - `3_sar_run_processing_prep.py`: Step 3 - Prepare SAR processing runs
  - **run_analysis**: 
    - `inference_run_analysis.py`: Script for analyzing inference runs
    - `model_run_analysis.py`: Script for analyzing model runs
    - `test_run_analysis.py`: Script for analyzing test runs
- **configs**: Configuration files for OpenMM 
  - `burn_scars_config.py`: Config for burn scars model
  - `forest_disturbances_config.py`: Config for forest disturbances model
  - `multi_temporal_crop_classification.py`: Config for multi-temporal crop classification
  - `sen1floods11_config.py`: Config for Sentinel-1 floods model
- **geospatial_fm**: Geospatial Foundation Model
  - `datasets.py`: Datasets utilities and functions
  - `geospatial_fm.py`: Geospatial feature management core functions
  - `geospatial_pipelines.py`: Pipelines for geospatial processing
  - `temporal_encoder_decoder.py`: Temporal encoder-decoder models
- **src**: Source code with functions 
  - `custom_hooks.py`: Custom hooks for models
  - `dataset_management.py`: Dataset management utilities
  - `hls_stacks_prep.py`: HLS stack preparation utilities
  - `model_analysis.py`: Functions for model analysis
  - `model_input_processor.py`: Process model input data
  - `model_management.py`: Manage model training and inference
  - `sar_model_input_processor.py`: SAR model input processor
  - `sar_processing_prep.py`: Prepare SAR data processing
  - `test_analysis.py`: Analyze test metrics
  - `utility_functions.py`: General utility functions for data processing

- `dump_file.py`: File for deprecated or unused code
- `model_inference.py`: Script for model inference
- `model_inference_terminal_commands.py`: Commands for terminal-based model inference
- `run_conda_command.py`: Script for running conda environment commands
- `run_config.py`: Script for running configuration-based model training
- `run_inference_command.py`: Script for running model inference commands
- `run_test_command_prithvi.py`: Script for testing Prithvi model
- `run_test_command_unet.py`: Script for testing UNet model
- `run_unet_model_command.py`: Script for running UNet model training
- `PREVIOUSLY_USED_COMMANDS.txt`: Text file with previously used commands


# Image segmentation by foundation model finetuning

This repository shows three examples of how [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M) can be finetuned for downstream tasks. The examples include flood detection using Sentinel-2 data from the [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) dataset, burn scars detection using the [NASA HLS fire scars dataset](https://huggingface.co/datasets/nasa-impact/hls_burn_scars) and multi-temporal crop classification using the [NASA HLS multi-temporal crop classification dataset](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification).

## The approach
### Background
To finetune for these tasks in this repository, we make use of [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/), which provides an extensible framework for segmentation tasks. 

[MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) allows us to concatenate necks and heads appropriate for any segmentation downstream task to the encoder, and then perform the finetuning. This only requires setting up a config file detailing the desired model architecture, dataset setup and training strategy. 

We build extensions on top of [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) to support our encoder and provide classes to read and augment remote sensing data (from .tiff files) using [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) data pipelines. These extensions can be found in the [geospatial_fm](./geospatial_fm/) directory, and they are installed as a package on the top of [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) for ease of use. If more advanced functionality is necessary, it should be added there.

### The pretrained backbone
The pretrained model we work with is a [ViT](https://arxiv.org/abs/2010.11929)operating as a [Masked Autoencoder](https://arxiv.org/abs/2111.06377), trained on [HLS](https://hls.gsfc.nasa.gov/) data. The encoder from this model is made available as the backbone and the weights can be downloaded from Hugging Face [here](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/blob/main/Prithvi_100M.pt).


### The architectures
We use a simple architecture that adds a neck and segmentation head to the backbone. The neck concatenates and processes the transformer's token based embeddings into an embedding that can be fed into convolutional layers. The head processes this embedding into a segmentation mask. The code for the architecture can be found in [this file](./geospatial_fm/geospatial_fm.py).

### The pipeline
Additionally, we provide extra components for data loading pipelines in [geospatial_pipelines.py](./geospatial_fm/geospatial_pipelines.py). These are documented in the file.
We observe the MMCV convention that all operations assumes a channel-last format. Our tiff loader also assumes this is the format in which files are written, and offers a flag to automatically transpose a to channel-last format if this is not the case.
*However*, we also introduce some components with the prefix `Torch`, such as `TorchNormalize`. These components assume the torch convention of channel-first.

At some point during the pipeline, before feeding the data to the model, it is necessary to change to channel-first format.
We reccomend implementing the change after the `ToTensor` operation (which is also necessary at some point), using the `TorchPermute` operation.
## Setup
### Dependencies
1. Clone this repository
2. `conda create -n <environment-name> python==3.9`
3. `conda activate <environment-name>`
4. Install torch (tested for >=1.7.1 and <=1.11.0) and torchvision (tested for >=0.8.2 and <=0.12). May vary with your system. Please check at: https://pytorch.org/get-started/previous-versions/.
    1. e.g.: `pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 --extra-index-url https://download.pytorch.org/whl/cu115`
5. `cd` into the cloned repo
5. `pip install -e .`
6. `pip install -U openmim`
7. `mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/{cuda_version}/{torch_version}/index.html`. Note that pre-built wheels (fast installs without needing to build) only exist for some versions of torch and CUDA. Check compatibilities here: https://mmcv.readthedocs.io/en/v1.6.2/get_started/installation.html
    1. e.g.: `mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11.0/index.html`

### Data

The flood detection dataset can be downloaded from [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11). Splits in the `mmsegmentation` format are available in the `data_splits` folders.


The [NASA HLS fire scars dataset](https://huggingface.co/datasets/nasa-impact/hls_burn_scars) can be downloaded from Hugging Face.

The [NASA HLS multi-temporal crop classification dataset](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification) can be downloaded from Hugging Face.


## Running the finetuning
1. In the `configs` folder there are three config examples for the three segmentation tasks. Complete the configs with your setup specifications. Parts that must be completed are marked with `#TO BE DEFINED BY USER`. They relate to the location where you downloaded the dataset, pretrained model weights, the test set (e.g. regular one or Bolivia out of bag data) and where you are going to save the experiment outputs.

2. 
    a. With the conda env created above activated, run:
    
    `mim train mmsegmentation --launcher pytorch configs/sen1floods11_config.py` or 
    
    `mim train mmsegmentation --launcher pytorch configs/burn_scars.py` or
    
    `mim train mmsegmentation --launcher pytorch configs/multi_temporal_crop_classification.py`
    
    b. To run testing: 
    
    `mim test mmsegmentation configs/sen1floods11_config.py --checkpoint /path/to/best/checkpoint/model.pth --eval "mIoU"` or 
    
    `mim test mmsegmentation configs/burn_scars.py --checkpoint /path/to/best/checkpoint/model.pth --eval "mIoU"` or
    
    `mim test mmsegmentation configs/multi_temporal_crop_classification.py --checkpoint /path/to/best/checkpoint/model.pth --eval "mIoU"`

## Checkpoints on Hugging Face
We also provide checkpoints on Hugging Face for the [burn scars detection](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M-burn-scar) and the [multi temporal crop classification tasks](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M-multi-temporal-crop-classification).

## Running the inference
We provide a script to run inference on new data in GeoTIFF format. The data can be of any shape (e.g. height and width) as long as it follows the bands/channels of the original dataset. An example is shown below.

```
python model_inference.py -config /path/to/config/config.py -ckpt /path/to/checkpoint/checkpoint.pth -input /input/folder/ -output /output/folder/ -input_type tif -bands "[0,1,2,3,4,5]"
```

The `bands` parameter is useful in case the files used to run inference have the data in different orders/indexes than the original dataset.

## Additional documentation
This project builds on [MMSegmentation](https://mmsegmentation.readthedocs.io/en/0.x/) and [MMCV](https://mmcv.readthedocs.io/en/v1.5.0/). For additional documentation, consult their docs (please note this is currently version 0.30.0 of MMSegmentation and version 1.5.0 of MMCV, not latest).

## Citation

If this repository helped your research, please cite `HLS foundation` in your publications. Here is an example BibTeX entry:

```
@software{HLS_Foundation_2023,
    author          = {Jakubik, Johannes and Chu, Linsong and Fraccaro, Paolo and Bangalore, Ranjini and Lambhate, Devyani and Das, Kamal and Oliveira Borges, Dario and Kimura, Daiki and Simumba, Naomi and Szwarcman, Daniela and Muszynski, Michal and Weldemariam, Kommy and Zadrozny, Bianca and Ganti, Raghu and Costa, Carlos and Watson, Campbell and Mukkavilli, Karthik and Roy, Sujit and Phillips, Christopher and Ankur, Kumar and Ramasubramanian, Muthukumaran and Gurung, Iksha and Leong, Wei Ji and Avery, Ryan and Ramachandran, Rahul and Maskey, Manil and Olofossen, Pontus and Fancher, Elizabeth and Lee, Tsengdar and Murphy, Kevin and Duffy, Dan and Little, Mike and Alemohammad, Hamed and Cecil, Michael and Li, Steve and Khallaghi, Sam and Godwin, Denys and Ahmadi, Maryam and Kordi, Fatemeh and Saux, Bertrand and Pastick, Neal and Doucette, Peter and Fleckenstein, Rylie and Luanga, Dalton and Corvin, Alex and Granger, Erwan},
    doi             = {10.57967/hf/0952},
    month           = aug,
    title           = {{HLS Foundation}},
    repository-code = {https://github.com/nasa-impact/hls-foundation-os},
    year            = {2023}
}
```

-->
