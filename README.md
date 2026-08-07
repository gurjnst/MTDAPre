# MTDAPre
MTDAPre is a target guided model designed for MISO (Multiple-Input Single-Output) meteorological field prediction, where multiple meteorological variables are used to forecast a specific target variable. The repository also supports SISO (Single-Input Single-Output) and MIMO (Multiple-Input Multiple-Output) settings.

## Installation
```bash
conda create -n mtdapre python=3.10 -y
conda activate mtdapre
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
pip install hickle xarray==0.19.0 decord fvcore lpips nni pandas scikit-image==0.19.3 timm==0.6.11 tqdm
pip install tensorboard einops matplotlib
pip install natten==0.21.6+torch2110cu126 -f https://whl.natten.org
pip install opencv-python==4.8.1.78 numpy==1.24.3
pip install netCDF4 h5py h5netcdf dask thop
```

## Overview
`models/` contains the implementation of MTDAPre and related modules.  
`datasets/` contains dataset loading and preprocessing codes.  
`configs/` contains training configuration files.  
`tools/` contains auxiliary training and evaluation scripts.

## Train
python train.py

## Quicktest
python quick_test.py

## Datasets
The experiments in this repository use dual-polarization radar datasets and the WeatherBench dataset.

## Acknowledgments
Our code is based on [OpenSTL](https://github.com/chengtan9907/OpenSTL) and [PredFormer](https://github.com/yyyujintang/PredFormer).
We greatly appreciate the code bases they provided for this project.
