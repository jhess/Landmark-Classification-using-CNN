# Landmark Classification using a Convolutional Neural Network (CNN)

This project using a deep CNN model that classifies different landmarks after being pretrained on many different images of world landmarks.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Setup](#setup)
- [Features](#features)

## Installation

Install the dependencies from the requirements.txt using the command below:
```shell
pip install -r requirements.txt
```

Originally, this project was run with the above package versions with Python 3.7.6, but it had to be updated to 3.8.1 to be compatible with Windows 11.

If there are issues installing the torch versions with CUDA support enabled, you will need to run `pip install` on each of them separately in the command line: torch, torchvision, and torchaudio:
```shell
pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```
```shell
pip install torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```
```shell
pip install torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
Then install the rest of the packages in the requirements.txt file (comment out temporarily the three above packages).

If you get a long error when it installs the psutil package that has to do with Visual Studio C++ 14.0, you will need to install the latest C++ MSVC at this [link](https://visualstudio.microsoft.com/visual-cpp-build-tools/), and restart your computer after installing them again in VS Support Tools.

Use this [link](https://pytorch.org/get-started/previous-versions/) to see what versions of CUDA enabled PyTorch to install.

## Usage

Run the `cnnmodel.ipynb` to train a new instance of the network, which gets saved in the checkpoints folder as .pt.

Run the cells in `transferlearning.ipynb` to generate the other .pt model files to use in the app.ipynb notebook.

## Setup

The first cell in the `cnnmodel.ipynb notebook` checks if CUDA GPU support is available:

```python
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
workers = multiprocessing.cpu_count()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    torch.cuda.empty_cache()
    print(f'Memory allocated: {torch.cuda.memory_allocated()}')
    print(f'Max memory allocated: {torch.cuda.max_memory_allocated()}')

# If running locally, this will download dataset
setup_env()
```

The test and training landmark image sets will also be downloaded after creating a `data/landmark_images` directory from the main workspace directory when the `setup_env()` function is called. It downloads the data from [here](https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip) if this is the first time running the notebook and no data images are present.

## Features

Run the `cnnmodel.ipynb` Jupyter notebook to run through the steps of training and testing the CNN against a large data set of landmark images. The source files, including the CNN model, are in the `src` folder.

The `transferlearning.ipynb` Jupyter notebook shows how to use a pretrained ResNet18 model that was already trained on many different images and use its backbone with the fully connected (FC) linear layer 'head' of our network to improve our accuracy even more on the test set of images.
