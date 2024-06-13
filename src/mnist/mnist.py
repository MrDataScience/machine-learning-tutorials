import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms import v2
import matplotlib.pyplot as plt


# source: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

def change_pix_values(image: torch.Tensor, row_i: int, row_j: int, pix_val: float) -> torch.Tensor:
    """
    lambda that will change all pixels from `row_i:row_j` to a new value, given by `pix_val`
    """
    _, height, _ = image.size()

    # image[:, row_i:row_j, :] = image[:, int(height/2), :]
    image[:, row_i:row_j, :] = pix_val
    return image


class MyLambda(Lambda):
    """custom lambda class that will support arguments"""

    def __init__(self, lambd, row_i, row_j, pix_val):
        super().__init__(lambd)
        self.row_i = row_i
        self.row_j = row_j
        self.pix_val = pix_val

    def __call__(self, img, ):
        return self.lambd(img, self.row_i, self.row_j, self.pix_val)


def load_data(data_set, transformer) -> tuple:

    training_data = data_set(
        root="data",
        train=True,
        download=True,
        transform=transformer
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


def get_transformers():
    """
    return two transformers, second one supports v2
    """
    t1 = Compose([
        # First transform it to a tensor
        ToTensor(),
        # Then erase the middle
        MyLambda(change_pix_values, 10, 20, 0.5),
    ])

    t2 = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        MyLambda(change_pix_values, 10, 20, 0.5),
    ])

    return t1, t2


if __name__ == "__main__":
    transformer, transformer_v2 = get_transformers()

    training_data, test_data = load_data(
        datasets.MNIST, transformer=transformer_v2)

    # Constants
    COLS, ROWS = 3, 3
    IMG_SIZE = 5

    plot_data(training_data, IMG_SIZE, COLS, ROWS)
