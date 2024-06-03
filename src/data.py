import math
import torch
import torch.utils.data
from pathlib import Path
from torchvision import datasets, transforms
import os
import multiprocessing
from .helpers import compute_mean_and_std, get_data_location
import matplotlib.pyplot as plt


def get_data_loaders(
    batch_size: int = 20, valid_size: float = 0.2, num_workers: int = 0, limit: int = -1
):
    """
    Create and returns the train_one_epoch, validation and test data loaders.

    :param batch_size: size of the mini-batches
    :param valid_size: fraction of the dataset to use for validation. For example 0.2
                       means that 20% of the dataset will be used for validation
    :param num_workers: number of workers to use in the data loaders. Use num_workers=1. 
            0 - disable multiprocessing
    :param limit: maximum number of data points to consider
    :return a dictionary with 3 keys: 'train_one_epoch', 'valid' and 'test' containing respectively the
            train_one_epoch, validation and test data loaders
    """

    # We will fill this up later
    dataLoaders = {"train": None, "valid": None, "test": None}

    base_path = Path(get_data_location())

    # Compute mean and std of the dataset
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")

    # Create 3 sets of data transforms: one for the training dataset,
    # containing data augmentation, one for the validation dataset
    # (without data augmentation) and one for the test set (again
    # without augmentation)
    # Resize the image to 256 first, then crop them to 224, then add the
    # appropriate transforms for that step
    dataTransforms = {
        "train": transforms.Compose(
            [
                # Images are 32x32. We enlarge them a bit so we can then take a random crop
                transforms.Resize(256),
                
                # take a random crop of the image
                transforms.RandomCrop(224, padding_mode = "reflect", pad_if_needed = True),
                
                # Horizontal flip is not part of RandAugment according to RandAugment paper
                #transforms.RandomHorizontalFlip(0.5),
                
                # RandAugment has 2 main parameters: how many transformations should be
                # applied to each image, and the strength of these transformations. This
                # latter parameter should be tuned through experiments: the higher the more
                # the regularization effect
                transforms.RandAugment(
                    #transforms.AutoAugmentPolicy.IMAGENET
                    num_ops = 2,
                    magnitude = 9,
                    interpolation = transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        ),
        "valid": transforms.Compose(
            [
                # Both of these are useless, but we keep them because
                # in a non-academic dataset you will need them
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        ),
        "test": transforms.Compose(
            [
                # Both of these are useless, but we keep them because
                # in a non-academic dataset you will need them
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        ),
    }

    # Get train and test subfolder paths from landmark_images base path
    trainPath = os.path.join(base_path, 'train')
    testPath = os.path.join(base_path, 'test')

    # Create train and validation datasets
    train_data = datasets.ImageFolder(
        trainPath,
        transform = dataTransforms["train"]
    )

    # The validation dataset is a split from the train_one_epoch dataset, so we read
    # from the same folder, but we apply the transforms for validation
    valid_data = datasets.ImageFolder(
        trainPath,
        transform = dataTransforms["valid"]
    )

    # obtain training indices that will be used for validation
    n_tot = len(train_data)
    indices = torch.randperm(n_tot)

    # If requested, limit the number of data points to consider
    if limit > 0:
        indices = indices[:limit]
        n_tot = limit

    split = int(math.ceil(valid_size * n_tot))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler  = torch.utils.data.SubsetRandomSampler(valid_idx)

    # prepare data loaders
    dataLoaders["train"] = torch.utils.data.DataLoader(
        train_data,
        batch_size = batch_size,
        sampler = train_sampler,
        #shuffle = True - don't need shuffle since using Sampler
        num_workers = num_workers
    )
    dataLoaders["valid"] = torch.utils.data.DataLoader(
        valid_data,
        batch_size = batch_size,
        sampler = valid_sampler,
        #shuffle = False - don't need shuffle since using Sampler
        num_workers = num_workers
    )

    # Now create the test data loader
    test_data = datasets.ImageFolder(
        testPath,
        transform = dataTransforms["test"]
    )

    test_shuffle = False
    if limit > 0:
        indices = torch.arange(limit)
        test_sampler = torch.utils.data.SubsetRandomSampler(indices)
    else:
        test_sampler = None
        test_shuffle = True

    dataLoaders["test"] = torch.utils.data.DataLoader(
        test_data,
        batch_size = batch_size,
        sampler = test_sampler,
        shuffle = test_shuffle,
        num_workers = num_workers
    )

    return dataLoaders


def visualize_one_batch(dataLoaders, max_n: int = 5):
    """
    Visualize one batch of data.

    :param dataLoaders: dictionary containing data loaders
    :param max_n: maximum number of images to show
    :return: None
    """

    # obtain one batch of training images
    # First obtain an iterator from the train dataloader
    dataiter  = iter(dataLoaders["train"])

    images, labels  = dataiter.next()

    # Undo the normalization (for visualization purposes)
    mean, std = compute_mean_and_std()
    invTrans = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )

    images = invTrans(images)

    # Get class names from the train data loader
    trainloader = dataLoaders["train"]
    traindata = trainloader.dataset
    # Get class name from the folder names (which are stored in traindata.classes)
    class_names  = [name.split('.')[1] for name in traindata.classes]

    # Convert from BGR (the format used by pytorch) to
    # RGB (the format expected by matplotlib)
    images = torch.permute(images, (0, 2, 3, 1)).clip(0, 1)

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in range(max_n):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        # print out the correct label for each image
        # .item() gets the value contained in a Tensor
        ax.set_title(class_names[labels[idx].item()])


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    return get_data_loaders(batch_size=2, num_workers=0)


def test_data_loaders_keys(data_loaders):

    assert set(data_loaders.keys()) == {"train", "valid", "test"}, "The keys of the dataLoaders dictionary should be train, valid and test"


def test_data_loaders_output_type(data_loaders):
    # Test the data loaders
    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    assert isinstance(images, torch.Tensor), "images should be a Tensor"
    assert isinstance(labels, torch.Tensor), "labels should be a Tensor"
    assert images[0].shape[-1] == 224, "The tensors returned by your dataloaders should be 224x224. Did you " \
                                       "forget to resize and/or crop?"


def test_data_loaders_output_shape(data_loaders):
    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    assert len(images) == 2, f"Expected a batch of size 2, got size {len(images)}"
    assert (
        len(labels) == 2
    ), f"Expected a labels tensor of size 2, got size {len(labels)}"


def test_visualize_one_batch(data_loaders):

    visualize_one_batch(data_loaders, max_n=2)
