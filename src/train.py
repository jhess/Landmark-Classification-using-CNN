import tempfile
import torch.optim as optim
import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from src.helpers import after_subplot


def train_one_epoch(train_dataloader, model, optimizer, loss, train_on_gpu):
    """
    Performs one train_one_epoch epoch
    """

    if torch.cuda.is_available() & train_on_gpu:
        model.cuda()

    model.train()
    
    trainLoss = 0.0

    # Go through each batch
    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80):

        # move data to GPU
        if torch.cuda.is_available() & train_on_gpu:
            data, target = data.cuda(), target.cuda()

        # Clear the gradients of all optimized variables
        optimizer.zero_grad()
        # Forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # Calculate the loss
        lossValue = loss(output, target)
        # Backward pass: compute gradient of the loss with respect to model parameters
        lossValue.backward()
        # Perform a single optimization step (parameter update)
        optimizer.step()

        # update average training loss
        trainLoss = trainLoss + ((1 / (batch_idx + 1)) * (lossValue.data.item() - trainLoss))

    return trainLoss


def valid_one_epoch(valid_dataloader, model, loss, train_on_gpu):
    """
    Validate at the end of one epoch
    """

    with torch.no_grad():

        # set the model to evaluation mode
        model.eval()

        if torch.cuda.is_available() & train_on_gpu:
            model.cuda()

        validLoss = 0.0

        # Go through each batch
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80):

            # move data to GPU
            if torch.cuda.is_available() & train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # Calculate the loss
            lossValue = loss(output, target)

            # Calculate average validation loss
            validLoss = validLoss + ((1 / (batch_idx + 1)) * (lossValue.data.item() - validLoss))

    return validLoss


def optimize(dataLoaders, model, optimizer, loss, n_epochs, save_path, interactive_tracking=False, train_on_gpu=True):
    # initialize tracker for minimum validation loss
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    validLossMin = None
    logs = {}

    # Learning rate scheduler: setup a learning rate scheduler that
    # reduces the learning rate when the validation loss reaches a plateau
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, verbose=True, patience=5)

    for epoch in range(1, n_epochs + 1):

        trainLoss = train_one_epoch(dataLoaders["train"], model, optimizer, loss, train_on_gpu)

        validLoss = valid_one_epoch(dataLoaders["valid"], model, loss, train_on_gpu)

        # print training/validation statistics
        print("Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(epoch, trainLoss, validLoss))

        # If the validation loss decreases by more than 1%, save the model
        if validLossMin is None or ((validLossMin - validLoss) / validLossMin > 0.01):
            print(f"New minimum validation loss: {validLoss:.6f}. Saving model ...")

            # Save the weights to save_path
            torch.save(model.state_dict(), save_path)

            validLossMin = validLoss

        # Update learning rate, i.e., make a step in the learning rate scheduler
        scheduler.step(validLoss)

        # Log the losses and the current learning rate
        if interactive_tracking:
            logs["loss"] = trainLoss
            logs["val_loss"] = validLoss
            logs["lr"] = optimizer.param_groups[0]["lr"]

            liveloss.update(logs)
            liveloss.send()


def one_epoch_test(test_dataloader, model, loss, train_on_gpu=True):
    # monitor test loss and accuracy
    testLoss = 0.
    correct = 0.
    total = 0.

    # set the module to evaluation mode
    with torch.no_grad():

        # set the model to evaluation mode
        model.eval()

        if torch.cuda.is_available() & train_on_gpu:
            model = model.cuda()

        for batch_idx, (data, target) in tqdm(
                enumerate(test_dataloader),
                desc='Testing',
                total=len(test_dataloader),
                leave=True,
                ncols=80):
            
            # move data to GPU
            if torch.cuda.is_available() & train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Forward pass: compute predicted outputs by passing inputs to the model
            logits = model(data)
            # Calculate the loss
            lossValue = loss(logits, target)

            # update average test loss
            testLoss = testLoss + ((1 / (batch_idx + 1)) * (lossValue.data.item() - testLoss))

            # convert logits to predicted class - the predicted class is the index of the max of the logits
            _ , pred = torch.max(logits.data, 1)

            # compare predictions to true label
            correct += torch.sum(torch.squeeze(pred.eq(target.data.view_as(pred))).cpu())
            total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(testLoss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))

    return testLoss


    
######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=50, limit=200, valid_size=0.5, num_workers=0)


@pytest.fixture(scope="session")
def optim_objects():
    from src.optimization import get_optimizer, get_loss
    from src.model import CNNet

    model = CNNet(50)

    return model, get_loss(), get_optimizer(model)


def test_train_one_epoch(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lt = train_one_epoch(data_loaders['train'], model, optimizer, loss)
        assert not np.isnan(lt), "Training loss is nan"


def test_valid_one_epoch(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lv = valid_one_epoch(data_loaders["valid"], model, loss)
        assert not np.isnan(lv), "Validation loss is nan"

def test_optimize(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    with tempfile.TemporaryDirectory() as temp_dir:
        optimize(data_loaders, model, optimizer, loss, 2, f"{temp_dir}/hey.pt")


def test_one_epoch_test(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    tv = one_epoch_test(data_loaders["test"], model, loss)
    assert not np.isnan(tv), "Test loss is nan"
