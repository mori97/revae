import torch
from torchvision import datasets, transforms

CELEBA_MEAN = (0.5063, 0.4258, 0.3832)
CELEBA_STD = (0.3043, 0.2839, 0.2834)


def get_celeba():
    """Get and preprocess the CelebA dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(CELEBA_MEAN, CELEBA_STD)])
    celeba = datasets.CelebA('./data', transform=transform,
                             download=True)
    # Use only 18/40 labels as described in Appendix C.1
    mask = torch.tensor(
        [False, True, False, True, False, True, False, False, True, True,
         False, True, True, True, False, True, False, False, True, False,
         True, False, False, False, True, False, True, False, True, False,
         False, True, False, True, False, False, False, False, True, True])
    ret = [(img, label[mask]) for img, label in celeba]
    return ret
