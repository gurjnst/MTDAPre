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
- `configs/weather/`: contains training configs for different meteorological forecasting settings.
- `openstl/models/mtdapre_t.py`: contains the MTDAPre model for target meteorological field prediction.
- `openstl/models/mtdapre_m.py`: contains the MTDAPre model with middle fusion strategy.
- `openstl/models/mtdapre_c.py`: contains the MTDAPre model with early fusion settings.
- `openstl/modules/mtdapre_modules.py`: contains the basic modules used in MTDAPre.
- `openstl/datasets/dataloader_weather.py`: contains the data loading and preprocessing code for WeatherBench experiments.

## Quicktest
Run the quick test script to verify the initialization and inference of MTDAPre.
```bash
python quick_test.py
```

## Train
Modify the configuration path in `train.py` and run:
```bash
python train.py
```

## Datasets
- [WeatherBench](https://arxiv.org/abs/2002.00469) (ArXiv'2020) [[download](https://github.com/pangeo-data/WeatherBench)]
- [NJUCPOL](https://doi.org/10.1029/2021GL095302) (GRL'2021) [[download](https://zenodo.org/records/5109403)]

## Acknowledgments
Our code is based on [OpenSTL](https://github.com/chengtan9907/OpenSTL) and [PredFormer](https://github.com/yyyujintang/PredFormer).
We greatly appreciate the code bases they provided for this project.
