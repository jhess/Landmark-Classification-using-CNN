import torch
import torch.nn as nn
import numpy as np

# define the CNN architecture
class CNNet(nn.Module):
    def __init__(self, numClasses: int = 50, dropout: float = 0.5, imgSize = 224*224) -> None:

        super(CNNet, self).__init__()

        # convolutional layer 1. It sees 3x224x224 image tensor
        # and produces 16 feature maps 32x32 (i.e., a tensor 16x32x32)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(16)

        self.relu = nn.ReLU()
        # 2x2 pooling with stride 2. It sees tensors 16x128x128
        # and halves their size, i.e., the output will be 16x64x64
        self.pool = nn.MaxPool2d(2, 2)

        # convolutional layer 2 -> creates 32 layers from 16 inputs
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(32)

        # convolutional layer 3 -> creates 64 layers from 32 inputs
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(64)

        # flatten (64 * 28 * 28 -> 50176)
        self.flatten = nn.Flatten()

        # 224 * 224, 1/2 * 1/2 * 1/2 for each MaxPool layer, but we lose the original 3 channels (should be 64 * 28 * 28)
        self.numPixels = imgSize

        # get mean between intermediate image size here and number of classes to determine number of fc1 hidden nodes
        h1Nodes = int(np.floor(imgSize/40))

        # linear layer 1 (50176 -> 1254)
        self.fc1 = nn.Linear(imgSize, h1Nodes)
        self.dp = nn.Dropout(dropout)
        self.batchNorm4 = nn.BatchNorm1d(h1Nodes)

        # linear layer 2 (1254 -> 627)
        self.fc2 = nn.Linear(h1Nodes, numClasses)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.pool(self.relu(self.batchNorm1(self.conv1(x)))) # -> 16x112x112
        x = self.pool(self.relu(self.batchNorm2(self.conv2(x)))) # -> 32x56x56
        x = self.pool(self.relu(self.batchNorm3(self.conv3(x)))) # -> 64x28x28

        #x = x.view(-1, self.numPixels) 64 * 28 * 28)
        x = self.flatten(x)

        x = self.relu(self.batchNorm4(self.dp(self.fc1(x))))

        x = self.fc2(x)
        
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = CNNet(numClasses=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
