# LaDI-VTON-Shoes refer to [**LaDI-VTON**](https://github.com/miccunifi/ladi-vton)

### Latent Diffusion Textual-Inversion Enhanced Virtual Try-On


## Overview
<img width="412" alt="image" src="https://github.com/sheep8888/ladi-vton-shoes/assets/92249359/ee38981f-2a03-4fea-a8b9-2d9fc5ea6509">
<img width="478" alt="image" src="https://github.com/sheep8888/ladi-vton-shoes/assets/92249359/0a1c09d5-2fac-4321-817c-cd8a68f2c17d">



### Installation

1. Clone the repository

```sh
git clone https://github.com/sheep8888/ladi-vton-shoes
```

2. Install Python dependencies

```sh
conda env create -n ladi-vton -f environment.yml
conda activate ladi-vton
```

Alternatively, you can create a new conda environment and install the required packages manually:

```sh
conda create -n ladi-vton -y python=3.10
conda activate ladi-vton
pip install torch==2.0.1 torchvision==0.15.2 opencv-python==4.7.0.72 diffusers==0.14.0 transformers==4.27.3 accelerate==0.18.0 clean-fid==0.1.35 torchmetrics[image]==0.11.4 wandb==0.14.0 matplotlib==3.7.1 tqdm xformers
```

### Data Preparation

#### ShoesData

Download the [ShoesData](https://github.com/sheep8888/ladi-vton-shoes/releases/download/untagged-b97e7db945f89bd95228/VTON-S.zip) dataset
Once the dataset is downloaded, the folder structure should look like this:

```
├── DressCode
│   ├── [train | test]
│   │   ├── ip
│   │   │   ├── [0.jpg | 1.jpg | 2.jpg | 3.jpg | ...]
│   │   ├── ia
│   │   │   ├── [0.jpg| 2.jpg | ...]
│   │   ├── jp
│   │   │   ├── [0.json | 1.json | ...]
│   │   ├── ic
│   │   │   ├── [0.jpg | 1.jpg | ...]
│   │   ├── im
│   │   │   ├── [0.jpg | 1.jpg | ...]
```


### Train
```sh
!python src/train_vto.py --dataset shoes  --shoes_dataroot <path> --output_dir <path> --inversion_adapter_dir <path>
```

### Inference
```sh
!python src/inference.py --dataset shoes --shoes_dataroot <path> --output_dir <name> --test_order [paired|unpaired] 
```

### Metrics computation
The metric calculation code section has not been modified

**Please refer to  [**LaDI-VTON**](https://github.com/miccunifi/ladi-vton) for more details**



