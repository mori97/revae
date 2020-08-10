import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import datasets
import models
from utils import AverageMeter

# Plot reconstructions every PLOT_INTERVAL epochs in the test time
PLOT_INTERVAL = 5
PLOT_IDX = (19, 2, 1, 13, 14, 8, 4, 9, 18, 0)


def _collate_train(batch):
    """A collate function for train data loader.
    This function makes unsupervised samples come before supervised samples.
    """
    batch.sort(key=lambda sample: sample[1])
    x, t = list(zip(*batch))
    x = torch.stack(x, dim=0)
    t = torch.tensor(t)
    return x, t


def train(model, data_loader, optimizer, device, epoch, writer):
    model.train()

    lb_meter = AverageMeter()

    for x, t in data_loader:
        batch_size = x.size(0)
        x, t = x.view(-1, 784).to(device), t.to(device)

        loss = torch.mean(model(x, t))
        lb_meter.update(-loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    writer.add_scalar('train/lower bound', lb_meter.avg, epoch)


def test(model, data_loader, device, epoch, writer, plot_data):
    model.eval()

    recon_loss_meter = AverageMeter()
    accu_meter = AverageMeter()

    with torch.no_grad():
        for x, t in data_loader:
            batch_size = x.size(0)
            x, t = x.view(-1, 784).to(device), t.to(device)

            z_mu, _ = model.encoder(x)
            recon = model.decoder(z_mu)
            recon_loss = F.binary_cross_entropy(recon, x, reduction='none')
            recon_loss = torch.mean(torch.sum(recon_loss, dim=1), dim=0)
            recon_loss_meter.update(recon_loss.item(), batch_size)

            z_c_mu = z_mu[:, :model.z_c_dim]
            y = torch.argmax(model.classifier(z_c_mu), dim=1)
            accu = (y == t).sum().float() / batch_size
            accu_meter.update(accu.item(), batch_size)

        writer.add_scalar('test/recon loss', recon_loss_meter.avg, epoch)
        writer.add_scalar('test/accuracy', accu_meter.avg, epoch)

        if epoch % PLOT_INTERVAL == 0:
            plot_data = plot_data.view(10, 784).to(device)
            z_mu, _ = model.encoder(plot_data)
            recon = model.decoder(z_mu).cpu().view(-1, 28, 28).numpy()

            fig, ax = plt.subplots(1, 10)
            for i in range(10):
                ax[i].axis('off')
                ax[i].imshow(recon[i], cmap='binary', vmin=0, vmax=1)
            writer.add_figure('recon', fig, epoch)


def main():
    parser = argparse.ArgumentParser(
        description='Train the Reparameterized VAE',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', '-b',
                        help='Batch size.',
                        type=int, default=512)
    parser.add_argument('--epochs', '-e',
                        help='Number of epochs.',
                        type=int, default=100)
    parser.add_argument('--gpu', '-g',
                        help='GPU id. (Negative number indicates CPU)',
                        type=int, default=0)
    parser.add_argument('--learning-rate', '-l',
                        help='Learning Rate.',
                        type=float, default=5e-4)
    parser.add_argument('--log-dir',
                        help='TensorBoard save directory location.',
                        type=str, default=None)
    parser.add_argument('--supervision-rate', '-s',
                        help='Supervision rate.',
                        type=float, default=0.2)
    args = parser.parse_args()

    if_use_cuda = torch.cuda.is_available() and args.gpu >= 0
    device = torch.device('cuda:{}'.format(args.gpu) if if_use_cuda else 'cpu')

    train_dataset, test_dataset =\
        datasets.get_fashion_mnist(supervision_rate=args.supervision_rate)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate_train, num_workers=2, pin_memory=if_use_cuda)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=if_use_cuda)

    model = models.REVAEMNIST(z_c_dim=4, z_exc_dim=10)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    writer = SummaryWriter(log_dir=args.log_dir)

    plot_data = torch.stack(
        [test_dataset[i][0][0, ...] for i in PLOT_IDX], dim=0)
    fig, ax = plt.subplots(1, 10)
    for i in range(10):
        ax[i].axis('off')
        ax[i].imshow(plot_data[i], cmap='binary', vmin=0, vmax=1)
    writer.add_figure('origin', fig)

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, device, epoch, writer)
        test(model, test_loader, device, epoch, writer, plot_data)

    writer.close()


if __name__ == '__main__':
    main()
