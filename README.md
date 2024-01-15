# LaDI-VTON-Shoes refer to [**LaDI-VTON**](https://github.com/miccunifi/ladi-vton)

### Latent Diffusion Textual-Inversion Enhanced Virtual Try-On


## Overview
<img width="412" alt="image" src="https://github.com/mortal-163/ladi-vton/assets/92249359/ca6a9d0c-0105-4642-98c6-57c7fb271a46">


### Installation

1. Clone the repository

```sh
git clone https://github.com/mortal-163/ladi-vton
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

Download the [ShoesData]() dataset
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


</details>




