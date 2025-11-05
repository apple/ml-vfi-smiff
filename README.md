# Structure Motion based Iterative Feature Fusion (SMIFF) for Video frame interpolation

This software project accompanies the research paper, [Video Frame Interpolation via Structure Motion based Iterative Feature Fusion](https://arxiv.org/abs/2105.05353)

This work proposes an end-to-end structure-motion based iterative fusion method for video frame interpolation.

## Documentation

In this project, we propose a video frame interpolation method via structure-motion based iterative fusion, which aims to provide results with a clear and reasonable appearance. To achieve this goal, a two-stage framework is established. Given two adjacent frames, we encode images by structure and motion based learning branches respectively in the first stage. Then, the temporal information alignment unit and spatial feature based rectifier unit is introduced in the second stage, which achieves further enhancement based on adjacent frames and hierarchical context. Here, iterative learning structure is utilized to integrate spatial and temporal feature based optimization, and hence to generate video results with higher quality. To learn more about this work, please refer [link](https://sid.onlinelibrary.wiley.com/doi/abs/10.1002/sdtp.14635).

The Code structure of this project is listed below.

```bash
# Datas/
  # vimeo_90k_interpolation.py # dataloader of vimeo-90k

# Models/
  # SmiffNet.py # network building of the proposed SMIFF

# my_package/ # c++/cu functions for projection calculation

# Utils/ # other functions
  # average_meter.py: evaluation function
  # loss_functions.py: loss functions
  # lr_scheduler.py: training scheduler

# inference.py: inference pipeline

# train.py: training pipeline
```

## Getting Start

### Requirements:

1. pytorch: version==1.2.0

2. mmdetection: version==1.1

### Installation:

1. clone this repo:
```bash
git clone git@github.com:apple/ml-vfi-smiff.git
```

2. build the dependencies:

```bash
cd my_package/
sh build.sh
cd ../

cd Models/correlation_package_pytorch1_0/
sh build.sh
cd ../
```

### Usage:

#### Training:

1. Prepare your training dataset. Vimeo-90k triplet set is widely use, you can download it from http://toflow.csail.mit.edu/index.html#triplet

2. Set your path to dataset and the txt files for "train_list" and "test_list" in "train.sh"

3. Download the pretrained_weight from "https://www.icloud.com.cn/iclouddrive/0aaPyenXEametIepzSqXARpPQ#SMIFF_weights" into "Weights/", and set the weight path also in "train.sh".

4. run the training precess by

```bash
sh train.sh
```

#### Inference:

1. Prepare your inference dataset. A video shoud be split to several frames, and you should put each pair of adjacent frame into a folder. The dataset should be list as:

```yaml
video/
  1/
    frame_00.png
    frame_02.png
  ...
  n/
    frame_00.png
    frame_02.png
```

2. Set your path to dataset in "inference.sh"

3. Download the trained_weight from "https://www.icloud.com.cn/iclouddrive/0aaPyenXEametIepzSqXARpPQ#SMIFF_weights" into "Weights/", and set the weight path also in "inference.sh".

4. run the inference precess by

```bash
sh inference.sh
```

### Tips:

This work is mainly built on DAIN [1] and FeatureFlow [2]. If you want to follow this work [3], please cite the following papers:

[1] Bao W, Lai W S, Ma C, et al. Depth-aware video frame interpolation[C]//CVPR 2019.

[2] Gui S, Wang C, Chen Q, et al. Featureflow: Robust video interpolation via structure-to-texture generation[C]//CVPR 2020.

[3] Li X, Cao M, Tang Y, et al. 13‐3: Invited Paper: Video Frame Interpolation via Structure Motion based Iterative Feature Fusion[C]//SID 2021.
