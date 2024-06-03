from src.helpers import setup_env, compute_mean_and_std, plot_confusion_matrix
from src.data import get_data_loaders, visualize_one_batch
from src.train import optimize, one_epoch_test
from src.optimization import get_optimizer, get_loss
from src.predictor import Predictor, predictor_test
from src.model import CNNet
from src.lrfinder import lr_finder
import matplotlib.pyplot as plt
import multiprocessing
import torch
import torch.optim as optim
import torchvision
import torchvision.models as models
import numpy as np
from ipywidgets import VBox, Button, FileUpload, Output, Label
from PIL import Image
from IPython.display import display
import io