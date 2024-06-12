import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import v2
from torchvision.transforms import Compose, Lambda
import matplotlib.pyplot as plt


# source: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

def erase_middle(image: torch.Tensor) -> torch.Tensor:
    all, height, width = image.size()
    image[:, 10:20, :] = image[:, 14, :]

    return image


def load_data(data_set, transformer) -> tuple:
    special = Compose(
        [
            # First transform it to a tensor
            ToTensor(),
            # Then erase the middle
            Lambda(erase_middle),
        ]
    )

    training_data = data_set(
        root="data",
        train=True,
        download=True,
        transform=special
    )

    test_data = data_set(
        root="data",
        train=False,
        download=True,
        transform=transformer
    )

    return training_data, test_data


def plot_data(data, fig_size, cols, rows):
    figure = plt.figure(figsize=(fig_size, fig_size))
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label = data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(f'{label}')
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


if __name__ == "__main__":
    transformer = ToTensor()
    training_data, test_data = load_data(
        datasets.MNIST, transformer=transformer)

    # Constants
    COLS, ROWS = 3, 3
    IMG_SIZE = 5

    plot_data(training_data, IMG_SIZE, COLS, ROWS)
