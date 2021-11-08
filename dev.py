import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST

PATH_DATASETS = 'data'
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)


class FMNISTDataModule(LightningDataModule):

    def __init__(
            self,
            data_dir: str = PATH_DATASETS,
            batch_size: int = BATCH_SIZE,
            num_workers: int = NUM_WORKERS,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose({
            # convert image or numpy.array (HxWxC) in the range [0,225] to in the range[0,1]
            transforms.ToTensor(),
            # Normalize a tensor image with mean and standard deviation
            transforms.Normalize((0.1307,), (0.3081,)),
        })

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # download
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            feminist_full = FashionMNIST(
                self.data_dir, train=True,
                transform=self.transform)
            self.femnist_train, self.femnist_val = random_split(feminist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.fmnist_test = FashionMNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.femnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.femnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.fmnist_test, batch_size=self.batch_size, num_workers=self.num_workers)


class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.img_shape = 100
        self.embedding = nn.Embedding(10, 10)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(110, 256, normalize=False),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 28 * 28),
            nn.Tanh(),
        )

    def forward(self, z, y):
        # pass the labels into a embedding layer
        labels_embedding = self.embedding(y)
        # concat the embedded labels and the noise tensor
        # z is a tensor of size (batch_size, latent_dim + dim_label_encode)
        z = torch.cat([z, labels_embedding], dim=-1)
        img = self.model(z)
        # model returns three tensors in its forward method: return self.decode(z), mu, logvar
        # so this return tensor and shape
        return img


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(28 * 28 + 10, 512),

            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z, y):
        # pass the labels into a embedding layer
        labels_embedding = self.embedding(y)
        # concat the embedded labels and the noise tensor
        # z is a tensor of size (batch_size, latent_dim + dim_label_encode)
        z = z.view(z.size(0), -1)
        z = torch.cat([z, labels_embedding], dim=-1)
        img = self.model(z)
        # model returns three tensors in its forward method: return self.decode(z), mu, logvar
        # so this return tensor and shape
        return img


class CGAN(LightningModule):

    def __init__(
            self,
    ):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self, z, y):
        """
        Generates an image using the generator
        given input noise z and labels y
        """
        return self.generator(z, y)

    def generator_step(self, x):
        """
        Training step for generator
        1. Sample random noise and labels
        2. Pass noise and labels to generator to
           generate images
        3. Classify generated images using
           the discriminator
        4. Backprop loss
        """

        # Sample random noise and labels
        z = torch.randn(x.shape[0], 100, device=device)
        y = torch.randint(0, 10, size=(x.shape[0],), device=device)

        # Generate images
        generated_imgs = self(z, y)

        # Classify generated image using the discriminator
        d_output = torch.squeeze(self.discriminator(generated_imgs, y))

        # Backprop loss. We want to maximize the discriminator's
        # loss, which is equivalent to minimizing the loss with the true
        # labels flipped (i.e. y_true=1 for fake images). We do this
        # as PyTorch can only minimize a function instead of maximizing
        g_loss = nn.BCELoss()(d_output,
                              torch.ones(x.shape[0], device=device))

        return g_loss

    def discriminator_step(self, x, y):
        """
        Training step for discriminator
        1. Get actual images and labels
        2. Predict probabilities of actual images and get BCE loss
        3. Get fake images from generator
        4. Predict probabilities of fake images and get BCE loss
        5. Combine loss from both and backprop
        """

        # Real images
        d_output = torch.squeeze(self.discriminator(x, y))
        loss_real = nn.BCELoss()(d_output,
                                 torch.ones(x.shape[0], device=device))

        # Fake images
        z = torch.randn(x.shape[0], 100, device=device)
        y = torch.randint(0, 10, size=(x.shape[0],), device=device)

        generated_imgs = self(z, y)
        d_output = torch.squeeze(self.discriminator(generated_imgs, y))
        loss_fake = nn.BCELoss()(d_output,
                                 torch.zeros(x.shape[0], device=device))

        return loss_real + loss_fake

    def training_step(self, batch, batch_idx, optimizer_idx):
        X, y = batch

        # train generator
        if optimizer_idx == 0:
            loss = self.generator_step(X)

        # train discriminator
        if optimizer_idx == 1:
            loss = self.discriminator_step(X, y)

        return loss

    def configure_optimizers(self):
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
        return [g_optimizer, d_optimizer], []


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dm = FMNISTDataModule()
    model = CGAN()
    dm.setup('fit')
    mnist_dataloader = dm.train_dataloader()
    trainer = pl.Trainer(max_epochs=1, gpus=1 if torch.cuda.is_available() else 0, progress_bar_refresh_rate=50)
    trainer.fit(model, mnist_dataloader)
    print(1)
