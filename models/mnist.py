import torch
import torch.nn as nn
import torch.nn.functional as F


class _Encoder(nn.Module):

    def __init__(self, z_dim):
        super(_Encoder, self).__init__()

        self.fc1 = nn.Linear(784, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 2 * z_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu, logvar = torch.chunk(self.fc3(h), 2, dim=1)
        return mu, logvar


class _Decoder(nn.Module):

    def __init__(self, z_dim):
        super(_Decoder, self).__init__()

        self.fc1 = nn.Linear(z_dim, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 784)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = torch.sigmoid(self.fc3(h))
        return h


class _Classifier(nn.Module):

    def __init__(self, z_c_dim):
        super(_Classifier, self).__init__()

        self.fc1 = nn.Linear(z_c_dim, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, z_c):
        h = F.relu(self.fc1(z_c))
        h = self.fc2(h)
        return h


class _ConditionalPrior(nn.Module):

    def __init__(self, z_c_dim):
        super(_ConditionalPrior, self).__init__()

        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 2 * z_c_dim)

    def forward(self, y):
        h = F.relu(self.fc1(y))
        mu, logvar = torch.chunk(self.fc2(h), 2, dim=1)
        return mu, logvar
