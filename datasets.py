from itertools import chain

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
