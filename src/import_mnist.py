import torch
import torchvision.datasets as datasets


def load_mnist():
    mnist_trainset = datasets.MNIST(
        root='./data', train=True, download=True, transform=None)

    mnist_testset = datasets.MNIST(
        root='./data', train=False, download=True, transform=None)
    return mnist_trainset, mnist_testset


if __name__ == "__main__":
    train, test = load_mnist()
    print(len(train))
    print(len(test))
