# MTDAPre

This repository provides the PyTorch implementation of MTDAPre for multivariable meteorological field prediction.

## Installation

```bash
conda create -n mtdapre python=3.10 -y

conda activate mtdapre

pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126

pip install hickle xarray==0.19.0 decord fvcore lpips nni pandas scikit-image==0.19.3 timm==0.6.11 tqdm

pip install tensorboard einops matplotlib

pip install natten==0.21.6+torch2110cu126 -f https://whl.natten.org

pip install opencv-python==4.8.1.78 numpy==1.24.3

pip install netCDF4 h5py h5netcdf dask
```

## Overview

The main structure of this repository is organized as follows:

```
MTDAPre/
├── configs/          # Configuration files
├── datasets/         # Dataset processing codes
├── models/           # Model implementation
├── tools/            # Training and evaluation tools
├── train.py          # Training script
├── test.py           # Evaluation script
├── quick_test.py     # Quick test example
├── requirements.txt
└── README.md
```

`models/` contains the implementation of MTDAPre and related modules.  
`datasets/` contains dataset loading and preprocessing codes.  
`configs/` contains training configuration files.  
`tools/` contains auxiliary training and evaluation scripts.

## Train

Modify the dataset paths and experiment configurations before training.

Example:

```bash
python train.py --config configs/example.yaml
```

The training logs and model checkpoints will be saved in the specified output directory.

## Quicktest

A quick test script is provided to verify that MTDAPre can be initialized and perform inference successfully.

Run:

```bash
python quick_test.py
```

The quick test uses randomly generated inputs and does not require downloading the complete datasets.

## Datasets

The experiments in this repository use dual-polarization radar datasets and the WeatherBench dataset.

### Dual-polarization Radar Datasets

**Shijiazhuang Radar Dataset**

The Shijiazhuang dataset contains:

- $Z_H$: horizontal reflectivity
- $Z_{DR}$: differential reflectivity
- $K_{DP}$: specific differential phase
- $\rho_{HV}$: cross-correlation coefficient

The reflectivity field $Z_H$ is used as the target variable, while other polarimetric variables are used as auxiliary inputs.

**Nanjing Radar Dataset (NJUCPOL)**

The Nanjing dataset contains:

- $Z_H$
- $Z_{DR}$
- $K_{DP}$

The reflectivity field $Z_H$ is used as the prediction target.

**WeatherBench**

WeatherBench dataset can be downloaded from:

https://github.com/pangeo-data/WeatherBench

The experiments use the following meteorological variables:

- $u10m$
- $v10m$
- $t2m$
- $tcc$
- $rh$

## Acknowledgments

Our code is based on [OpenSTL](https://github.com/chengtan9907/OpenSTL) and [PredFormer](https://github.com/yyyujintang/PredFormer).

We greatly appreciate the code bases they provided for this project.
