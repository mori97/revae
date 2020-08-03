from itertools import chain
import random

import torch
from torchvision import datasets, transforms


def get_celeba():
    """Get and preprocess the CelebA dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()])
    # Use only 18/40 labels as described in Appendix C.1
    mask = torch.tensor(
        [False, True, False, True, False, True, False, False, True, True,
         False, True, True, True, False, True, False, False, True, False,
         True, False, False, False, True, False, True, False, True, False,
         False, True, False, True, False, False, False, False, True, True])

    train = datasets.CelebA('./data', split='train', transform=transform,
                            download=True)
    valid = datasets.CelebA('./data', split='valid', transform=transform,
                            download=True)
    test = datasets.CelebA('./data', split='test', transform=transform,
                           download=True)

    # Use 'train' split and 'valid' split as the training set
    train = [(img, label[mask]) for img, label in chain(train, valid)]
    test = [(img, label[mask]) for img, label in test]

    return train, test


def get_fashion_mnist(supervision_rate=0.06, seed=0):
    """Get and preprocess the Fashion-MNIST dataset.
    """
    random.seed(seed)

    train = datasets.FashionMNIST('./data', train=True, download=True,
                                  transform=transforms.ToTensor())
    test = datasets.FashionMNIST('./data', train=False, download=True,
                                 transform=transforms.ToTensor())

    # Mask some labels as unsupervised data
    for t in range(10):
        mask = (train.targets == t)
        ts = train.targets[mask]
        uns = [True if i >= ts.size(0) * supervision_rate else False
               for i in range(ts.size(0))]
        random.shuffle(uns)
        uns = torch.tensor(uns)
        mask[mask] = uns
        train.targets[mask] = -1

    return train, test
