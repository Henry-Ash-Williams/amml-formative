from os import PathLike
from typing import List

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: List[int]) -> None:
        super(Encoder, self).__init__()

        layers = []
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
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1] * 4 * 4, latent_dim)
        self.flatten = nn.Flatten()

    def forward(self, x: Tensor):
        x = self.input(x)
        x = self.encoder(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: List[int]) -> None:
        super(Decoder, self).__init__()
        self.input = nn.Linear(latent_dim, hidden_dims[0] * 4 * 4)
        self.hidden_dims = hidden_dims
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
            nn.ReLU(),
            # Reduce channels from 32 to 1
            nn.Conv2d(in_channels=hidden_dims[-1], out_channels=1, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, z: Tensor):
        z = self.input(z)
        z = z.view(-1, self.hidden_dims[0], 4, 4)
        z = self.decoder(z)
        z = self.output(z)
        return z


class VAE(nn.Module):
    def __init__(
        self,
        latent_dims: int,
        hidden_dimensions: List[int] = [32, 64, 128],
        state_dict_path: PathLike | None = None,
    ) -> None:
        super(VAE, self).__init__()
        if state_dict_path is not None:
            self.load_state_dict(torch.load(state_dict_path))
            exit()

        self.encoder = Encoder(latent_dims, hidden_dimensions)
        self.decoder = Decoder(latent_dims, hidden_dimensions[::-1])

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    @staticmethod
    def loss_function(
        reconstruction: Tensor,
        input: Tensor,
        mu: Tensor,
        log_var: Tensor,
        kld_weight=0.001,
    ):
        recons_loss = F.mse_loss(input=reconstruction, target=input)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )
        loss = recons_loss + kld_weight * kld_loss

        return {
            "loss": loss,
            "reconstruction_loss": recons_loss.detach(),
            "kld_loss": -kld_loss.detach(),
        }

    def encode(self, input: Tensor):
        return self.reparameterize(*self.encoder(input))

    def decode(self, z: Tensor):
        return self.decoder(z)

    def forward(self, input: Tensor):
        mu, log_var = self.encoder(input)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), input, mu, log_var
