import types

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import Callback
from utils import str2bool


class SigmaVAE_PL(pl.LightningModule):

    def __init__(
        self, hparams, normalization_func=None, inverse_normalization_func=None
    ):
        super().__init__()

        # set self.hparams implicitly:
        self.save_hyperparameters(
            hparams
        )  # required to avoid issues related with https://github.com/PyTorchLightning/pytorch-lightning/issues/3998

        self.normalization_func = normalization_func
        self.inverse_normalization_func = inverse_normalization_func
        if ("lr_scheduler_fractional_epochs" in self.hparams) and self.hparams[
            "lr_scheduler_fractional_epochs"
        ]:
            self.automatic_optimization = False
            self.training_step = self.fractional_scheduler_training_step

    def training_step_(self, batch, batch_idx):

        images, labels = batch

        # Run VAE
        recon_batch, mu, logvar = self(images)

        # Compute loss
        rec, kl = self.loss_function(recon_batch, images, mu, logvar)
        rec = rec / len(images)
        kl = kl / len(images)
        loss = rec + self.hparams["beta"] * kl

        # logging
        prefix = "train"
        self.log(
            f"{prefix}_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}_rec",
            rec,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}_kl",
            kl,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}_log_sigma",
            self.log_sigma,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}_mse",
            torch.mean(torch.square(images - recon_batch)),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        # automatic optimization
        return self.training_step_(batch, batch_idx)

    def fractional_scheduler_training_step(self, batch, batch_idx):
        """a manual optimization loop for CosineAnnealingWarmRestarts"""

        opt = self.optimizers()
        opt.zero_grad()
        loss = self.training_step_(batch, batch_idx)
        self.manual_backward(loss)
        opt.step()

        sch = self.lr_schedulers()
        sch.step(
            epoch=self.trainer.current_epoch
            + batch_idx / self.trainer.num_training_batches
        )

    def validation_step(self, batch, batch_idx):

        images, labels = batch

        # Run VAE
        recon_batch, mu, logvar = self(images)

        # Compute loss
        rec, kl = self.loss_function(recon_batch, images, mu, logvar)
        loss = (rec + self.hparams["beta"] * kl) / len(images)

        # logging
        prefix = "val"
        self.log(
            f"{prefix}_loss",
            loss,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}_rec",
            rec,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}_kl",
            kl,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}_log_sigma",
            self.log_sigma,
            prog_bar=False,
            logger=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}_mse",
            torch.mean(torch.square(images - recon_batch)),
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        if batch_idx == 0:
            if self.inverse_normalization_func is not None:
                images = self.inverse_normalization_func(images)
            self.save_recons(images, recon_batch)

        return loss

    @rank_zero_only
    def save_recons(self, images, recons):
        # if hasattr(self.trainer.datamodule,'unpreprocess'):
        #     images=self.trainer.datamodule.unpreprocess(images)
        #     recons=self.trainer.datamodule.unpreprocess(recons)

        n = min(images.size(0), 12)
        comparison = torch.cat(
            [
                torchvision.utils.make_grid(
                    images[:n],
                    nrow=12,
                    padding=2,
                    normalize=False,
                    scale_each=False,
                    pad_value=0,
                ),
                torchvision.utils.make_grid(
                    recons[:n],
                    nrow=12,
                    padding=2,
                    normalize=False,
                    scale_each=False,
                    pad_value=0,
                ),
            ],
            axis=1,
        )
        torchvision.utils.save_image(
            comparison,
            f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"
            f"reconstructions_{self.logger.name}_{self.current_epoch}.png",
        )

    @rank_zero_only
    def sample_images(self):
        # Get sample reconstruction image
        samples = self.sample(64)
        torchvision.utils.save_image(
            samples,
            f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"
            f"samples_{self.logger.name}_{self.current_epoch}.png",
            nrow=8,
            padding=2,
            normalize=False,
            scale_each=False,
            pad_value=0,
        )

    def configure_callbacks(self):
        class ImageSampling(Callback):
            def on_validation_epoch_start(self, trainer, pl_module):
                pl_module.sample_images()

        return [ImageSampling()]

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.adam_eps,
        )
        scheduler_opts = {
            key.replace(self.hparams.scheduler_name + "_", ""): value
            for (key, value) in self.hparams.items()
            if key.startswith(self.hparams.scheduler_name + "_")
        }
        scheduler_cls = getattr(lr_scheduler, self.hparams.scheduler_name)
        print("initializing scheduler", scheduler_cls, "with opts", scheduler_opts)
        scheduler = getattr(lr_scheduler, self.hparams.scheduler_name)(
            optimizer, **scheduler_opts
        )
        return [optimizer], [scheduler]


# The sigma-VAE code was adapted from https://github.com/orybkin/sigma-vae-pytorch/blob/master/model.py


def softclip(tensor, min):
    """Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials"""
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, n_channels):
        super(UnFlatten, self).__init__()
        self.n_channels = n_channels

    def forward(self, input):
        size = int((input.size(1) // self.n_channels) ** 0.5)
        return input.view(input.size(0), self.n_channels, size, size)


def gaussian_nll(mu, log_sigma, x):
    return (
        0.5 * torch.pow((x - mu) / log_sigma.exp(), 2)
        + log_sigma
        + 0.5 * np.log(2 * np.pi)
    )


from contextlib import contextmanager


@contextmanager
def evaluating(net):
    """Temporarily switch to evaluation mode."""
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()


class sVAE_module(SigmaVAE_PL):
    """a sigma-VAE module. The encoder can be VGG16 or a simple convnet"""

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VAE_opts")
        parser.add_argument(
            "--z_dim",
            type=int,
            default=20,
            metavar="N",
            help="latent representation dim (default: 20)",
        )
        parser.add_argument(
            "--beta",
            type=float,
            default=1.0,
            metavar="N",
            help="tradeoff between reconstruction and D-KL terms (default: 1.0)",
        )
        parser.add_argument(
            "--estimate_type",
            type=str,
            default="sigma_vae",
            metavar="N",
        )
        parser.add_argument(
            "--encoder_name",
            type=str,
            default="vgg16",
            metavar="N",
        )
        parser.add_argument(
            "--decoder_name",
            type=str,
            default="flexible_decoder",
            metavar="N",
        )
        parser.add_argument(
            "--img_channels",
            type=int,
            default=3,
            metavar="N",
        )
        parser.add_argument(
            "--n_decoder_channels",
            type=int,
            default=16,
            metavar="N",
        )
        parser.add_argument(
            "--n_decoder_blocks",
            type=int,
            default=4,
            metavar="N",
        )
        parser.add_argument(
            "--fc_layers",
            type=int,
            default=2,
            metavar="N",
        )
        parser.add_argument(
            "--dropout",
            type=str2bool,
            default=False,
            metavar="N",
        )
        return parent_parser

    def __init__(
        self,
        hparams,
        *args,
        normalization_func=None,
        inverse_normalization_func=None,
        **kwargs,
    ):
        super().__init__(
            hparams,
            *args,
            normalization_func=normalization_func,
            inverse_normalization_func=inverse_normalization_func,
            **kwargs,
        )

        self.img_size = self.hparams["img_size"]
        self.img_channels = self.hparams["img_channels"]
        self.z_dim = self.hparams["z_dim"]
        self.estimate_type = self.hparams["estimate_type"]
        self.encoder_name = self.hparams["encoder_name"]
        self.decoder_name = self.hparams["decoder_name"]
        self.n_decoder_channels = self.hparams["n_decoder_channels"]
        self.n_decoder_blocks = self.hparams["n_decoder_blocks"]
        self.fc_layers = self.hparams["fc_layers"]
        self.dropout = self.hparams["dropout"]

        # prepare encoder
        if self.encoder_name.startswith("vgg"):
            self.encoder = self.get_VGG_encoder(
                self.encoder_name, fc_layers=self.fc_layers, dropout=self.dropout
            )
        elif (
            self.encoder_name == "simple_CNN"
        ):  # the example CNNs from the sigma-VAE code
            self.encoder = self.get_simple_encoder(self.img_channels, filters_m=16)
        elif (
            self.encoder_name == "simple_CNN_bn"
        ):  # the example CNNs from the sigma-VAE code
            self.encoder = self.get_simple_encoder_bn(self.img_channels, filters_m=16)
        else:
            raise ValueError

        # prepare latent
        with evaluating(self):
            demo_input = torch.ones(
                [1, self.img_channels, self.img_size, self.img_size]
            )
            h_dim = self.encoder(demo_input).shape[1]
        # print('h_dim', h_dim)

        # map to latent z
        self.fc11 = nn.Linear(h_dim, self.z_dim)
        self.fc12 = nn.Linear(h_dim, self.z_dim)
        self.fc12.weight.data.fill_(0)  # https://stackoverflow.com/a/52005955
        self.fc12.bias.data.fill_(0)

        # prepare decoder

        if self.decoder_name == "simple_decoder":
            self.decoder, decoder_input_size = self.get_simple_decoder(
                16, self.img_channels, img_size=self.img_size
            )
        elif self.decoder_name == "flexible_decoder":
            self.decoder, decoder_input_size = self.get_flexible_decoder(
                self.n_decoder_channels,
                self.img_size,
                n_blocks=self.n_decoder_blocks,
                out_channels=3,
            )

        # print('decoder_input_size=',decoder_input_size)
        self.fc2 = nn.Linear(self.z_dim, decoder_input_size)

        self.log_sigma = 0.0
        if self.estimate_type == "sigma_vae":
            self.log_sigma = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc11(h), self.fc12(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(self.fc2(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def sample(self, n, seed=1234):
        with evaluating(self):
            with torch.no_grad():
                with torch.random.fork_rng(devices=[self.device]):
                    torch.random.manual_seed(seed)
                    sample = torch.randn(n, self.z_dim, device=self.device)
                    samples = self.decode(sample)
        return samples

    def reconstruction_loss(self, x_hat, x):
        """Computes the likelihood of the data given the latent variable,
        in this case using a Gaussian distribution with mean predicted by the neural network and variance = 1
        """

        if (
            self.normalization_func is not None
        ):  # when the input image channels have been centered and normalized, we'd like to do the same to the reconstructions.
            x_hat = self.normalization_func(x_hat)

        if self.estimate_type == "gaussian_vae":
            # Naive gaussian VAE uses a constant variance
            log_sigma = torch.zeros([], device=x_hat.device)
        elif self.estimate_type == "sigma_vae":
            # Sigma VAE learns the variance of the decoder as another parameter
            log_sigma = self.log_sigma
        elif self.estimate_type == "optimal_sigma_vae":
            log_sigma = ((x - x_hat) ** 2).mean([0, 1, 2, 3], keepdim=True).sqrt().log()
            self.log_sigma = log_sigma.item()
        else:
            raise NotImplementedError

        # Learning the variance can become unstable in some cases. Softly limiting log_sigma to a minimum of -6
        # ensures stable training.
        log_sigma = softclip(log_sigma, -6)
        rec = gaussian_nll(x_hat, log_sigma, x).sum()
        return rec

    def loss_function(self, recon_x, x, mu, logvar):
        # Important: both reconstruction and KL divergence loss have to be summed over all element!
        # Here we also sum the over batch and divide by the number of elements in the data later

        rec = self.reconstruction_loss(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return rec, kl

    @staticmethod
    def get_VGG_encoder(encoder_name, fc_layers=2, dropout=False):
        encoder = getattr(torchvision.models, encoder_name)(pretrained=False)
        del encoder.classifier

        if fc_layers >= 1:
            encoder.fc1 = nn.Linear(512 * 7 * 7, 4096)
            encoder.fc1_relu = nn.ReLU()
            if dropout:
                encoder.fc1_dropout = nn.Dropout()
        if fc_layers >= 2:
            encoder.fc2 = nn.Linear(4096, 4096)
            encoder.fc2_relu = nn.ReLU()
            if dropout:
                encoder.fc2_dropout = nn.Dropout()

        def vgg_forward_without_a_classifier_(self, x: torch.Tensor) -> torch.Tensor:
            """a modified version of torchvision's VGG forward, omitting the classifier"""
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            if fc_layers >= 1:
                x = self.fc1(x)
                x = self.fc1_relu(x)
                if dropout:
                    x = self.fc1_dropout(x)
            if fc_layers >= 2:
                x = self.fc2(x)
                x = self.fc2_relu(x)
                if dropout:
                    x = self.fc2_dropout(x)
            return x

        encoder.forward = types.MethodType(vgg_forward_without_a_classifier_, encoder)
        return encoder

    @staticmethod
    def get_simple_encoder(img_channels, filters_m):
        return nn.Sequential(
            nn.Conv2d(img_channels, filters_m, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters_m, 2 * filters_m, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * filters_m, 4 * filters_m, (5, 5), stride=2, padding=2),
            nn.ReLU(),
            Flatten(),
        )

    @staticmethod
    def get_simple_encoder_bn(img_channels, filters_m):
        return nn.Sequential(
            nn.Conv2d(img_channels, filters_m, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(filters_m),
            nn.ReLU(),
            nn.Conv2d(filters_m, 2 * filters_m, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(2 * filters_m),
            nn.ReLU(),
            nn.Conv2d(2 * filters_m, 4 * filters_m, (5, 5), stride=2, padding=2),
            nn.BatchNorm2d(4 * filters_m),
            nn.ReLU(),
            Flatten(),
        )

    @staticmethod
    def get_simple_decoder(filters_m, out_channels, img_size):
        n_input_units = (4 * filters_m) * int(img_size * 2**-2) ** 2

        return (
            nn.Sequential(
                UnFlatten(4 * filters_m),
                nn.ConvTranspose2d(
                    4 * filters_m, 2 * filters_m, (6, 6), stride=2, padding=2
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    2 * filters_m, filters_m, (6, 6), stride=2, padding=2
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    filters_m, out_channels, (5, 5), stride=1, padding=2
                ),
                nn.Sigmoid(),
            ),
            n_input_units,
        )

    @staticmethod
    def get_flexible_decoder(filters_m, img_size, n_blocks, out_channels=3):
        """
        filters_m (int) the number of filters in the final layer (just before the image)
        out_channels (int) number of image_channels
        img_size (int) the dimensionality of the output image (e.g. 224)
        n_blocks (int) how many resolution reduction/channel doubling operations
        """

        in_channels = filters_m * int(
            2**n_blocks
        )  # the number of channels is doubled in each block

        input_size = img_size  # simulate image dimension reduction
        for i_block in range(n_blocks):
            input_size = int(input_size / 2)

        n_input_units = in_channels * input_size**2

        sequential = nn.Sequential(UnFlatten(in_channels))

        for i_block in range(n_blocks):
            sequential.add_module(
                f"B{i_block}_conv1",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=int(in_channels * 4 / 2),
                    kernel_size=(3, 3),
                    stride=1,
                    padding=1,
                ),
            )
            sequential.add_module(
                f"B{i_block}_pixelshuffle1", nn.PixelShuffle(upscale_factor=2)
            )
            in_channels = int(in_channels / 2)
            # sequential.add_module(f'B{i_block}_batchnorm1',
            #     nn.BatchNorm2d(in_channels))
            sequential.add_module(f"B{i_block}_ReLU1", nn.ReLU())

            sequential.add_module(
                f"B{i_block}_conv2",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=(3, 3),
                    stride=1,
                    padding=1,
                ),
            )
            sequential.add_module(f"B{i_block}_ReLU2", nn.ReLU())

        sequential.add_module(
            "final_conv",
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1),
        )
        sequential.add_module("final_sigmoid", nn.Sigmoid())

        # print(sequential)
        return sequential, n_input_units
