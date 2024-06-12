import torch
import torchvision

from torch.utils.data import Dataset

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

from matplotlib import pyplot as plt
import numpy as np


def load_mnist(transform: transforms.Compose, batch_size: int) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
        Returns data loaders for the mnist datasets (a tuple with train set and one with dev set)
    """
    trainset = torchvision.datasets.MNIST(
        root='./data/', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(
        root='./data/', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


def show_data(features, labels):
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    for feature, label in zip(train_features, train_labels):
        img = feature.squeeze()
        print(f"Label: {label}")
        plt.imshow(img, cmap="gray")
        plt.show()


if __name__ == "__main__":

    batch_size = 1
    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    transform = ToTensor()

    trainloader, testloader = load_mnist(transform, batch_size)
    train_features, train_labels = next(iter(trainloader))
    test_features, test_labels = next(iter(testloader))

    show_data(train_features, train_labels)
    # show_data(test_features, test_labels)
