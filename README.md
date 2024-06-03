# Landmark Classification using a Convolutional Neural Network (CNN)

This project classifies different landmarks after being pretrained on many different images of world landmarks.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
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

If you get a long error when it installs the psutil package that has to do with Visual Studio C++ 14.0, you will need to install the latest C++ MVSC at this [link](https://visualstudio.microsoft.com/visual-cpp-build-tools/), and restart your computer after installing them again in VS Support Tools.

Use this [link](https://pytorch.org/get-started/previous-versions/) to see what versions of CUDA enabled PyTorch to install.

## Usage

Run the `cnnmodel.ipynb` to train a new instance of the networ, which gets saved in the checkpoints folder as .pt.

Run the cells in `transferlearning.ipynb` to generate the other .pt model files to use in the app.ipynb notebook.

## Features

Run the `cnnmodel.ipynb` Jupyter notebook to run through the steps of training and testing the CNN against a large data set of landmark images. The source files, including the CNN model, are in the `src` folder.

The `transferlearning.ipynb` Jupyter notebook shows how to use a pretrained ResNet18 model that was already trained on many different images and use its backbone with the fully connected (FC) linear layer 'head' of our network to improve our accuracy even more on the test set of images.