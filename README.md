# Evaluating Fairness in Chest Radiography Without Demographic Data

[Dilermando Queiroz Neto](https://www.dilermando.site), [André Anjos](https://anjos.ai), Lilian Berton.


![](images/workflow-features.png)

## Table of Contents
- [Evaluating Fairness in Chest Radiography Without Demographic Data](#evaluating-fairness-in-chest-radiography-without-demographic-data)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Getting Started](#getting-started)
    - [Configuration](#configuration)
    - [Model](#model)
    - [Data](#data)
  - [Citing](#citing)

## Introduction
This repository contains the implementation for our paper “Using Backbone Foundation Model for Evaluating Fairness in Chest Radiography Without Demographic Data.” The study addresses the challenge of assessing fairness in medical imaging models, particularly when demographic data is unavailable. We propose a novel approach using the backbone of a Foundation Model to create representative groups that approximate sensitive attributes, such as gender and age, without directly utilizing protected demographic data. Our method demonstrates significant improvements in gender fairness in both in-distribution and out-of-distribution scenarios, highlighting the potential of Foundation Models to promote equitable healthcare diagnostics.

## Installation
To use this code, clone the repository and install the necessary dependencies using Conda:

```bash
git clone https://github.com/your-repo/fairness-chest-radiography.git
cd fairness-chest-radiography
conda env create -f environment.yml
conda activate fairness
```

## Getting Started

### Configuration
To configure the project “Using Backbone Foundation Model for Evaluating Fairness in Chest Radiography Without Demographic Data,” you should define important parameters in a config.ini configuration file. Below is an example of how you can structure this file, with specific values filled in:

```ini
[general]
project_name = Fairness
base_dir = [BASE_DIRECTORY]
experiment_dir = [EXPERIMENT_DIRECTORY]

[data]
num_workers = 4
batch_size = 32

[nih]
labels = [NIH_LABELS_PATH]
data_dir = [NIH_DATA_DIRECTORY]

[chexpert]
labels = [CHEXPERT_LABELS_PATH]
data_dir = [CHEXPERT_DATA_DIRECTORY]

[brax]
labels = [BRAX_LABELS_PATH]
data_dir = [BRAX_DATA_DIRECTORY]

[model]
model_dir = [MODEL_DIRECTORY]
input_size = 224
num_classes = 10
learning_rate = 0.001
weight_decay = 0.0001
num_epochs = 10

[training]
optimizer = adam
loss_function = cross_entropy

[pca]
pca_dir = [PCA_DIRECTORY]

[cluster]
cluster_dir = [CLUSTER_DIRECTORY]

[tsne]
tsne_dir = [TSNE_DIRECTORY]
```
### Model

This framework uses the [REMEDIS](https://arxiv.org/pdf/2205.09723) model. If you choose to use this model, you will need to download it. You can follow the steps described in the [official project](https://github.com/google-research/medical-ai-research-foundations?tab=readme-ov-file) to download and set up the REMEDIS model.

After downloading, make sure to save the model weights in the `model/weights/cxr-50x1-remedis-m` folder within the project directory:


### Data

To run the project, you'll need to download the required datasets. This project relies on two key datasets for evaluation:

- **CheXpert Dataset**: The CheXpert dataset is a large dataset of chest X-rays labeled for 14 different observations. You can download this dataset from the [Stanford ML Group's CheXpert page](https://stanfordmlgroup.github.io/competitions/chexpert/). Follow the instructions on the page to request access and download the data.

- **NIH Chest X-ray Dataset**: This dataset, provided by the NIH Clinical Center, is one of the largest publicly available chest X-ray datasets. It contains over 100,000 anonymized images of chest X-rays, labeled with the associated conditions. You can download the dataset from the [NIH News Release page](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community). 

After downloading the datasets, place them in the appropriate directories as defined in your `config.ini` file under the `[nih]` and `[chexpert]` sections. This setup ensures that the data is properly linked and ready for processing by the project scripts.

## Citing
