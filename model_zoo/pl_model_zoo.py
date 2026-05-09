import torch
import torchvision.models
from torch.optim import lr_scheduler
import torch.nn as nn
import pytorch_lightning as pl


class myLightningModule(pl.LightningModule):

    def __init__(self, hparams, class_weights=None):
        super().__init__()

        # set self.hparams implicitly:
        self.save_hyperparameters(
            hparams
        )  # required to avoid issues related with https://github.com/PyTorchLightning/pytorch-lightning/issues/3998

        if class_weights is None:
            class_weights = torch.ones(self.hparams["num_classes"])
            class_weights = class_weights / class_weights.sum()
            self.register_buffer("class_weights", class_weights)
        else:
            self.register_buffer("class_weights", class_weights)

        if (
            "lr_scheduler_fractional_epochs" in hparams
            and self.hparams.lr_scheduler_fractional_epochs
        ):
            self.automatic_optimization = False
            self.training_step = self.fractional_scheduler_training_step

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = self._shared_eval(batch, batch_idx, "train")
        return loss

    def fractional_scheduler_training_step(self, batch, batch_idx):
        """a manual optimization loop for CosineAnnealingWarmRestarts"""

        opt = self.optimizers()
        opt.zero_grad()
        loss = self._shared_eval(batch, batch_idx, "train")
        self.manual_backward(loss)
        opt.step()

        sch = self.lr_schedulers()
        sch.step(
            epoch=self.trainer.current_epoch
            + batch_idx / self.trainer.num_training_batches
        )

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        batch_size = len(labels)
        outputs = self(imgs)

        loss = nn.functional.cross_entropy(outputs, labels, weight=self.class_weights)

        top1_cls = torch.argmax(outputs, dim=1)
        top1_acc = torch.sum(top1_cls == labels).item() / (batch_size * 1.0)

        _, top5_clss = torch.topk(outputs, k=5, dim=1)
        correct_pred = 0
        for k in range(5):
            predicted = top5_clss[:, k]
            correct_pred += torch.sum(predicted == labels)
        top5_acc = correct_pred / (batch_size * 1.0)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_top1_acc",
            top1_acc,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_top5_acc",
            top5_acc,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def _shared_eval(self, batch, batch_idx, prefix):
        imgs, labels = batch
        batch_size = len(labels)
        outputs = self(imgs)

        loss = nn.functional.cross_entropy(outputs, labels)
        top1_cls = torch.argmax(outputs, dim=1)
        top1_acc = torch.sum(top1_cls == labels).item() / (batch_size * 1.0)

        _, top5_clss = torch.topk(outputs, k=5, dim=1)
        correct_pred = 0
        for k in range(5):
            predicted = top5_clss[:, k]
            correct_pred += torch.sum(predicted == labels)
        top5_acc = correct_pred / (batch_size * 1.0)

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
            f"{prefix}_top1_acc",
            top1_acc,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}_top5_acc",
            top5_acc,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
            nesterov=self.hparams.nesterov,
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


class VGG16(myLightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VGG16")
        parser.add_argument("--num_classes", type=int, default=8631)
        return parent_parser

    def __init__(self, hparams, class_weights=None, model_name=None):
        super().__init__(hparams, class_weights=class_weights)

        self.model_name = model_name
        self.model = torchvision.models.vgg16(
            pretrained=False, num_classes=self.hparams.num_classes
        )


class BFM_MODEL(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BFM_MODEL")
        parser.add_argument("--model_class", type=str, default="vgg16_bn")
        parser.add_argument("--num_latents", type=int, default=508)
        return parent_parser

    def __init__(
        self,
        hparams,
        quat_var=None,
        light_direction_var=None,
        light_intensity_var=None,
        latent_keys=None,
        latent_cumdims=None,
    ):
        super().__init__()

        self.save_hyperparameters(hparams)
        self.model_name = "VGG16_BFM"
        model = torchvision.models.vgg16(
            pretrained=False, num_classes=self.hparams.num_latents
        )
        dropout_layers = [2, 5]
        for dropout_layer in dropout_layers:
            list(model.modules())[0].classifier[dropout_layer] = nn.Dropout(p=0.0)
        self.model = model

        if quat_var is None:
            quat_var = torch.ones([4])
        else:
            quat_var = torch.tensor(quat_var)
        self.register_buffer("quaternions_var", quat_var)

        if light_direction_var is None:
            light_direction_var = torch.ones([3])
        else:
            light_direction_var = torch.tensor(light_direction_var)
        self.register_buffer("light_direction_var", light_direction_var)

        if light_intensity_var is None:
            light_intensity_var = torch.ones([3])
        else:
            light_intensity_var = torch.tensor(light_intensity_var)
        self.register_buffer("light_intensity_var", light_intensity_var)

        self.keys = latent_keys
        self.cumdims = latent_cumdims

        if (
            "lr_scheduler_fractional_epochs" in hparams
            and self.hparams.lr_scheduler_fractional_epochs
        ):
            self.automatic_optimization = False
            self.training_step = self.fractional_scheduler_training_step

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = self._shared_eval(batch, batch_idx, "train")
        return loss

    def fractional_scheduler_training_step(self, batch, batch_idx):
        """a manual optimization loop for CosineAnnealingWarmRestarts"""

        opt = self.optimizers()
        opt.zero_grad()
        loss = self._shared_eval(batch, batch_idx, "train")
        self.manual_backward(loss)
        opt.step()

        sch = self.lr_schedulers()
        sch.step(
            epoch=self.trainer.current_epoch
            + batch_idx / self.trainer.num_training_batches
        )

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval(batch, batch_idx, "val")
        return loss

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def weighted_latent_mse(self, output, label, prefix):
        log_step = True if prefix == "train" else False
        mean_latent_mse = 0

        coef_indices = [
            i_latent for i_latent, latent in enumerate(self.keys) if "coefs" in latent
        ]
        other_indices = [
            i_latent
            for i_latent, latent in enumerate(self.keys)
            if "coefs" not in latent
        ]

        for i_latent in coef_indices:
            latent_name = self.keys[i_latent]
            start, end = self.cumdims[i_latent], self.cumdims[i_latent + 1]
            per_latent_mse = torch.mean(
                (output[:, start:end] - label[:, start:end]) ** 2
            )
            self.log(
                f"{latent_name}_mse_loss_{prefix}",
                per_latent_mse,
                prog_bar=True,
                logger=True,
                on_step=log_step,
                on_epoch=True,
                sync_dist=True,
            )
            mean_latent_mse += per_latent_mse

        for i_latent in other_indices:
            latent_name = self.keys[i_latent]
            latent_var = latent_name + "_var"
            start, end = self.cumdims[i_latent], self.cumdims[i_latent + 1]

            total_se = (output[:, start:end] - label[:, start:end]) ** 2
            mean_se_per_dim = torch.mean(total_se, axis=0) / getattr(self, latent_var)
            per_latent_mse = torch.mean(mean_se_per_dim)
            mean_latent_mse += per_latent_mse

            self.log(
                f"{latent_name}_mse_loss_{prefix}",
                per_latent_mse,
                prog_bar=True,
                logger=True,
                on_step=log_step,
                on_epoch=True,
                sync_dist=True,
            )

        return mean_latent_mse

    def _shared_eval(self, batch, batch_idx, prefix):
        imgs, labels = batch
        outputs = self(imgs)  # [num_imgs,num_latents]

        weighted_mse_loss = self.weighted_latent_mse(outputs, labels, prefix)

        log_step = True if prefix == "train" else False
        self.log(
            f"{prefix}_mse_loss",
            weighted_mse_loss,
            prog_bar=True,
            logger=True,
            on_step=log_step,
            on_epoch=True,
            sync_dist=True,
        )

        return weighted_mse_loss

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            print(f"initializing Adams with lr = {self.hparams.lr}...")
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer_name == "SGD":
            print(f"initializing SGD with lr = {self.hparams.lr}...")
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
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
