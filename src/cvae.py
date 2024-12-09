from os import PathLike
from typing import List

import torch
from torch import Tensor, nn
from torch.nn import functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(
        self, latent_dim: int, hidden_dims: List[int], num_categories: int
    ) -> None:
        super(Encoder, self).__init__()

        layers = []
        self.latent_dims = latent_dim
        self.num_categories = num_categories
        self.input = nn.Sequential(
            nn.Conv2d(1, hidden_dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(),
        )

        # Iterate over hidden layers two at a time
        # i.e. hidden_dims = [1, 2, 4, 8, 16, ...]
        # (1, 2)
        # (2, 4)
        # (4, 8)
        # ...
        for h1, h2 in zip(hidden_dims, hidden_dims[1:]):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(h1, h2, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h2),
                    nn.ReLU(),
                )
            )

        self.encoder = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.fc_z = nn.Linear(hidden_dims[-1] * 16, num_categories * latent_dim)

    def forward(self, x: Tensor):
        x = self.input(x)
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc_z(x)
        return x.view(-1, self.latent_dims, self.num_categories)


class Decoder(nn.Module):
    def __init__(
        self, latent_dim: int, hidden_dims: List[int], num_categories: int
    ) -> None:
        super(Decoder, self).__init__()
        self.input = nn.Linear(latent_dim * num_categories, hidden_dims[0] * 4 * 4)
        self.hidden_dims = hidden_dims
        self.latent_dims = latent_dim
        self.num_categories = num_categories
        layers = []

        for h1, h2 in zip(hidden_dims, hidden_dims[1:]):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        h1, h2, kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    nn.BatchNorm2d(h2),
                    nn.ReLU(),
                )
            )

        self.decoder = nn.Sequential(*layers)
        self.output = nn.Sequential(
            # Upsample from [C x 16 x 16] to [C x 28 x 28]
            # Where C is the number of channels in the final layer
            nn.ConvTranspose2d(
                in_channels=hidden_dims[-1],
                out_channels=hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=3,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            # Reduce channels from 32 to 1
            nn.Conv2d(in_channels=hidden_dims[-1], out_channels=1, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, z: Tensor):
        z = z.view(-1, self.latent_dims * self.num_categories)
        z = self.input(z)
        z = z.view(-1, self.hidden_dims[0], 4, 4)
        z = self.decoder(z)
        z = self.output(z)
        return z


class CVAE(nn.Module):
    def __init__(
        self,
        latent_dims: int,
        num_categories: int = 10,
        hidden_dimensions: List[int] = [32, 64, 128],
        state_dict_path: PathLike | None = None,
        temperature: float = 0.5,
        anneal_interval: int = 100,
        anneal_rate: float = 3e-5,
    ) -> None:
        super(CVAE, self).__init__()
        if state_dict_path is not None:
            self.load_state_dict(torch.load(state_dict_path))
            exit()

        self.anneal_interval = anneal_interval
        self.anneal_rate = anneal_rate
        self.temp = temperature
        self.min_temp = self.temp
        self.latent_dims = latent_dims
        self.num_categories = num_categories
        self.encoder = Encoder(latent_dims, hidden_dimensions, num_categories)
        self.decoder = Decoder(latent_dims, hidden_dimensions[::-1], num_categories)

    def reparameterize(self, z: Tensor) -> Tensor:
        u = torch.rand_like(z)
        g = -torch.log(-torch.log(u + 1e-7) + 1e-7)
        s = F.softmax((z + g) / self.temp, dim=-1)
        return s.view(-1, self.latent_dims * self.num_categories)

    def loss_function(
        self,
        reconstruction,
        original,
        logits,
        batch_idx: int,
        alpha: float = 30.0,
        eps: float = 1e-7,
        kld_weight: float = 0.001,
    ):
        logits = F.softmax(logits, dim=-1)

        if batch_idx % self.anneal_interval == 0 and self.training:
            self.temp = np.maximum(
                self.temp * np.exp(-self.anneal_rate * batch_idx), self.min_temp
            )

        reconstruction_loss = F.mse_loss(reconstruction, original, reduction="mean")
        h1 = logits * torch.log(logits + eps)
        h2 = logits * np.log(1.0 / self.num_categories + eps)
        kld_loss = torch.mean(torch.sum(h1 - h2, dim=(1, 2)), dim=0)

        loss = alpha * reconstruction_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "kld_loss": -kld_loss,
        }

    def encode(self, input: Tensor):
        return self.encoder(input)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, input):
        q = self.encode(input)
        z = self.reparameterize(q)
        return [self.decode(z), input, q]


if __name__ == "__main__":
    c = CVAE(10, 10)
    i = torch.rand((1, 1, 28, 28))
    z = c(i)
    print(z[0])
    print(c.loss_function(*z))
