# **Decent-RSC**: Project of Scalable Data Science and Distributed ML

## Members

Zesen Wang, Palatip Jopanya, Sheng Liu

## Introduction
Distributed training for deep neural networks has become a mainstream approach for consensus algorithms due to its efficiency and scalability, with the All-Reduce algorithm being a common example. However, in practice, a high-speed network connection between workers is essential to achieve high training performance in a distributed workflow. Decentralized training, on the other hand, is a promising alternative for training deep neural networks. The key difference between decentralized and distributed training is that, in decentralized training, only immediate worker pairs (or pre-defined selected worker pairs corresponding to the communication topology) communicate. This reduces communication overhead and speeds up per-iteration runtime to accelerate training time eventually.

In this project, we implement distributed training (including federated learning in data heterogeneity) to a practical autonomous driving scenario. Specifically, we investigate the performance of decentralized training on the Road Surface Classification Dataset ([RSCD](https://thu-rsxd.com/rscd/)), which classifies surfaces into 27 labeled classes based on their evenness, friction, and material. The sample images, detailed name of subclasses, and number of images for each class are shown in the following three figures. The deep neural network architecture used is Deep Residual Networks (ResNet-50). Training is conducted with an independently and identically distributed (IID) setting and the non-IID setting (by using dirichlet distribution).

In the realistic scenario, the images captured by cameras on automated vehicles are senstive as they may contain private information. Such that, the centralized learning paradigm that require data collection maybe infeasible considering the data protection regurations. Federated learning is a promising solution to bridge such data islands by only transfering model parameters. However, since data sharing is forbidden, federated learning also faces the data heterigeneity issue, i.e., the data distribution of a client/vehicle is different from each other, which may reduce the model performance. That is why this project also explores federated learning in non-IID setting.

<p align="center">
  <img src="https://github.com/WangZesen/scalable-data-project/blob/main/Fig/classification_sample.png?raw=true" alt="Sample image of RSCD" />
  <br>
  Figure 1. Sample images of RSCD.
</p>

<p align="center">
  <img src="https://github.com/WangZesen/scalable-data-project/blob/main/Fig/class-RSCD.png?raw=true" alt="Sublime's custom image"/>
  <br>
  Figure 2. Subclasses of RSCD.
</p>

<p align="center">
  <img src="https://github.com/WangZesen/scalable-data-project/blob/main/Fig/counts-RSCD.jpeg?raw=true" alt="Sublime's custom image"/>
  Figure 3. Number of images for each class.
</p>

## Method

### Multi-node Decentralized Training
The main task for decentralized training is to update the global parameter, $`x \in \Re^d`$, by the local parameter $`x_i \in \Re^d`$ at each worker (computing node) $` i = 1,2,3, ..., N`$ to reach the consensus. The optimization is to minimize the aggregated local training loss 

$$
\underset{x_1, x_2, ..., x_N}{\text{minimize}}  \ \frac{1}{N} \sum_{i=q}^N F_i(x_i)  \ , \ x_i = x_j \ \forall \ i,j 
$$

where $` F_i(x_i) `$ is the expected training loss of the local model $`x_i`$ at worker $`i`$. By doing so, we use adaptive momentum versions of decentralized gradient descent as 

<p align="center">
  <img src="https://github.com/WangZesen/scalable-data-project/blob/main/Fig/eq2.png?raw=true" alt="Sublime's custom image"/>
</p>


where the first term is an update that reduces the value of $` F_i(x_i) `$ and the final term accounts for the averaged model parameters of neighboring workers and drives the local parameters towards the same first-order stationary point.

### Overlapping communication and computation
The system splits the execution of the backward pass into buckets and interleaves the corresponding
local model updates. After updating a bucket, the corresponding communication can be initiated, and
its results are not needed by the worker and its neighbors until the same bucket requires updating in
the next iteration. By doing so, the decentralized updates are independent of neighbor information within the same iteration as in Figure 1.

<p align="center">
  <img src="https://github.com/WangZesen/scalable-data-project/blob/main/Fig/timelines.png?raw=true" alt="Sublime's custom image"/>
  Figure 1. Timelines of two workers in decentralized training.
</p>


### Heterogeneous Communication Cost

In All-Reduce, all workers must participate in each iteration. On the other hand, decentralized training allows workers to communicate only with their immediate neighbors (also called gossip communication). In the practical environment, the bandwidth between the nodes is limited and can be varied. Hence, the decentralized training could speed up the training time because of reduced communication costs. The setting of network topology influences the training time. Herein the alternating Exponential Ring topology is considered.

<p align="center">
  <img src="https://github.com/WangZesen/scalable-data-project/blob/main/Fig/ring.png?raw=true" alt="Sublime's custom image"/>
  Figure 2. The alternating Exponential Ring topology
</p>

## Results 
### 2x4xA40 GPUs with IID data distribution
<p align="center">
  <img src="https://github.com/WangZesen/scalable-data-project/blob/main/Fig/A40_iid_acc1.png?raw=true" alt="Sublime's custom image"/>
  
</p>
<p align="center">
  <img src="https://github.com/WangZesen/scalable-data-project/blob/main/Fig/A40_iid_acc5.png?raw=true" alt="Sublime's custom image"/>
  
</p>

<p align="center">
  <img src="https://github.com/WangZesen/scalable-data-project/blob/main/Fig/A40_iid_val.png?raw=true" alt="Sublime's custom image"/>
  
</p>

### 2x4xA100 GPUs with IID data distribution 
<p align="center">
  <img src="https://github.com/WangZesen/scalable-data-project/blob/main/Fig/A100-iid-acc1.png?raw=true" alt="Sublime's custom image"/>
  
</p>
<p align="center">
  <img src="https://github.com/WangZesen/scalable-data-project/blob/main/Fig/A100-iid-acc5.png?raw=true" alt="Sublime's custom image"/>
  
</p>

<p align="center">
  <img src="https://github.com/WangZesen/scalable-data-project/blob/main/Fig/A100-iid-val.png?raw=true" alt="Sublime's custom image"/>
  
</p>

<table align="center">
  <tr>
    <th>Setting</th>
    <th>Total train time (s): All-Reduce</th>
    <th>Total train time (s): Decen</th>
    <th>Speedup (s)</th>
  </tr>
  <tr>
    <td>IID 2x4xA40</td>
    <td>1220.96</td>
    <td>940.18</td>
    <td bgcolor="green">280.78</td>
  </tr>
  <tr>
    <td>IID 2x4xA100</td>
    <td>546.64</td>
    <td>508.78</td>
    <td bgcolor="green">37.86</td>
  </tr>
</table>

### Hardware comparison A400 & A40 in IID
<p align="center">
  <img src="https://github.com/WangZesen/scalable-data-project/blob/main/Fig/A100-A40-iid-all-reduce-acc5.png?raw=true" alt="Sublime's custom image"/>
</p>
<p align="center">
  <img src="https://github.com/WangZesen/scalable-data-project/blob/main/Fig/A100-A40-iid-decen-acc5.png?raw=true" alt="Sublime's custom image"/>
</p>

### Compare alphas in Non-IID
<p align="center">
  <img src="https://github.com/WangZesen/scalable-data-project/blob/main/Fig/compare-alphas_A100_both.png?raw=true" alt="Sublime's custom image"/>
</p>

### Alphas of All-Reduce of 2x4xA100 in Non-IID
<p align="center">
  <img src="https://github.com/WangZesen/scalable-data-project/blob/main/Fig/AllReduce 2x4xA100 Non-IID alphas.png?raw=true" alt="Sublime's custom image"/>
</p>

### Alphas of Decen of 2x4xA100 in Non-IID
<p align="center">
  <img src="https://github.com/WangZesen/scalable-data-project/blob/main/Fig/Decen 2x4xA100 Non-IID alphas.png?raw=true" alt="Sublime's custom image"/>
</p>

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


