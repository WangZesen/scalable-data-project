# **Decent-RSC**: Project of Scalable Data Science and Distributed ML

## Members

Zesen Wang, Palatip Jopanya, Sheng Liu

## Introduction
Distributed training for deep neural networks has become a mainstream approach for consensus algorithms due to its efficiency and scalability, with the All-Reduce algorithm being a common example. However, in practice, a high-speed network connection between workers is essential to achieve high training performance in a distributed workflow. Decentralized training, on the other hand, is a promising alternative for training deep neural networks. The key difference between decentralized and distributed training is that, in decentralized training, only immediate worker pairs (or pre-defined selected worker pairs) communicate. This reduces communication overhead and accelerates training time.

In this project, we investigate the performance of decentralized training on the Road Surface Classification Dataset (RSCD), which classifies surfaces into 27 labeled classes based on their evenness, friction, and material. The deep neural network architecture used is Deep Residual Networks (ResNet-50). Training is conducted with an independently and identically distributed (IID) dataset (non-IID data distribution can be further investigated if time allows).

## Method

## Results

## Reproduce Experiments

### Setup Python Environment

First, setup a virtual Python environment using `virtualenv` or `Anaconda` with `Python>=3.11`. Then install the dependencies as follows.

```
# install latest PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# install dali with cuda-11.0 as it has cuda dependency included
pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda110
# other auxiliary library
pip install wandb seaborn loguru scipy tqdm tomli-w pydantic
# wrapper library for decentralized training
pip install decent-dp
```

One has to login to wandb for uploading the training/testing statistics before runing the experiments ([detailed instruction](https://docs.wandb.ai/quickstart/)).
```
wandb login
```

### Prepare Data

Download the Road Surface Dataset from [here](https://thu-rsxd.com/).

Put the data under `./data/RSCD` and arrage the data like below for training and validation set.
```
data/RSCD/
├── train
│   ├── dry_asphalt_severe
│   ├── dry_asphalt_slight
│   ├── dry_asphalt_smooth
│   ├── dry_concrete_severe
│   ├── dry_concrete_slight
│   ├── ...
├── val
│   ├── dry_asphalt_severe
│   ├── dry_asphalt_slight
│   ├── dry_asphalt_smooth
│   ├── dry_concrete_severe
│   ├── dry_concrete_slight
│   ├── ...
```

### Train

The experiments are conducted on a data center using Slurm as the scheduler. To run the training with four A40 GPUs, 

```
sbatch -A <PROJECT_ACCOUNT> script/train/4xA40.sh $(which torchrun) config/data/rscd.toml config/train/resnet50.toml
```
where `<PROJECT_ACCOUNT>` is the slurm project account.

The evaulation on the validation set is done along with the training, and it's logged in both the log files and the wandb.

#### Configs
- `config/data/rscd.toml`: config file for RSC dataset.
- `config/train/resnet50.toml`: config file for typical distributed training with ResNet-50.
- `config/train/decent_resnet50.toml`: config file for decentralized training with ResNet-50.


