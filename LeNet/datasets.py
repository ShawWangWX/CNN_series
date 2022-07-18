#!env
# Author
# Create Time: 2022/7/4
# License: None
"""
Info
"""

# import
import os

import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def data_loader(path: str, batch_size=64):
    print("init")
    train_data = datasets.MNIST(
        root=path,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    test_data = datasets.MNIST(
        root=path,
        train=False,
        download=False,
        transform=transforms.ToTensor()
    )

    train_data_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        drop_last=True
    )
    test_data_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        drop_last=True
    )

    return train_data_loader, test_data_loader


if __name__ == '__main__':
    print(torch.cuda.is_available())

    path = "./data/"
    train_loader, test_loader = data_loader(path)

    img, label = next(iter(train_loader))
    print(img.size())
    # images = torchvision.utils.make_grid(img)

    # images = images.numpy().transpose((1, 2, 0))

    '''
    plt.figure("MNIST batch sample")
    plt.imshow(images)
    plt.show()
    '''
