# Training with More In-Distribution Data Benefits Out-of-Distribution Detection

In this work we identify that simply training with more in-distribution (ID) data helps with Out-of-Distribution (OOD) detection. Specifically, we propose a training framework named **SIO**, which uses additional synthetic ID samples produced by generative models (e.g., GANs), together with the real ID training data, to train the base classifier. SIO is compatible with multiple OOD detection methods (either inference techniques or training algorithms) and yields immediate performance gains on top of them.

<p align="center">
    <img src='/sio_related/figures/cifar_results_screenshot.png' width='870'>
</p>

Paper coming soon!

## Overview
Our code uses [OpenOOD](https://github.com/Jingkang50/OpenOOD). This repo should be multiple commits ahead of the OpenOOD's main branch, as we fixed some bugs and added custom code for running SIO. We might be able to integrate SIO to the official OpenOOD implementation in the near future.


## Setup
After cloning this repo, please follow the instructions [here](https://github.com/Jingkang50/OpenOOD/blob/main/README.md). We were using `torch==1.10.0+cu111` and `torchvision==0.11.0+cu111`.

The part that is unique to SIO is downloading the synthetic ID samples, which can be done with [this script](/sio_related/download_synthetic_id.sh).

## Reproducing results
We provide [here](./sio_related/sio_scripts.md) a list of bash scripts for reproducing all the main experiments in the paper (including training and evaluation).

There are a few notable differences in the experimental setting between this work and OpenOOD (for CIFAR experiments):
- For all methods we train the model for 200 epochs instead of 100 epochs, as longer training is beneficial for OOD detection.
- We report the average AUROC over 3 independent runs, while OpenOOD reported a single run's result.
- We remove Tiny ImageNet (TIN) images from the near-OOD split for CIFAR-10/CIFAR-100 since TIN is used as the training OOD data for Outlier Exposure. Not removing it makes a trivial problem where training and test OOD distribution overlap.

## What if I use my own implementation?
Integrating SIO into your own method/implementation is easy. Below we show how to do this with a few lines for CIFAR-10.
```python
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# the original real training set
train_set = CIFAR10(root, train=True)
real_ids = list(range(len(train_set)))

# load synthetic data
syn_images = np.load(syn_images_path)
syn_labels = np.load(syn_labels_path)

# append synthetic data to the dataset
train_set.data = np.concatenate(
    (train_set.data, syn_images), axis=0)
train_set.targets = np.concatenate(
    (np.array(train_set.targets), syn_labels), axis=0)
syn_ids = list(set(range(len(train_set))) - set(real_ids))

# dataloader
# TwoSourceSampler is a custom sampler for SIO
# which can be found in openood/datasets/imglist_extradata_dataset.py
batch_size = 128
real_ratio = 0.8
batch_sampler = TwoSourceSampler(
    real_ids, syn_ids,
    batch_size, real_ratio=real_ratio
)
train_loader = DataLoader(train_set, batch_sampler=batch_sampler)
```
That's it! The rest things (model, optimizer, loss function, training loop, etc.) are just as usual.


## Citation
If you find our paper/code helpful, please consider citing:
```
Coming soon!
```
