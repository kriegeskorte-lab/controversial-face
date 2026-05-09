import os, sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download

from model_zoo.pl_model_zoo import BFM_MODEL, VGG16
from model_zoo.sigma_VAE import sVAE_module
from model_zoo.preprocessing_funs import preprocessing_funs

HF_MODEL_REPO_ID = "wenx-guo/controversial-face-model-checkpoints"
DEFAULT_CHECKPOINT_DIR = "model_checkpoints"
REPO_ROOT = Path(__file__).resolve().parents[1]


VGG16_LAYER_NAMES = [
    "conv1_1",
    "relu1_1",
    "conv1_2",
    "relu1_2",
    "pool_1",
    "conv2_1",
    "relu2_1",
    "conv2_2",
    "relu2_2",
    "pool_2",
    "conv3_1",
    "relu3_1",
    "conv3_2",
    "relu3_2",
    "conv3_3",
    "relu3_3",
    "pool_3",
    "conv4_1",
    "relu4_1",
    "conv4_2",
    "relu4_2",
    "conv4_3",
    "relu4_3",
    "pool_4",
    "conv5_1",
    "relu5_1",
    "conv5_2",
    "relu5_2",
    "conv5_3",
    "relu5_3",
    "pool_5",
    "avgpool_vgg",
    "fc_6",
    "relu_6",
    "fc_7",
    "relu_7",
    "fc_8",
]


MODEL_MAP = {
    "VGG16_VGGFace2_128": "faceID-VGGFace2",
    "VGG16_VGGFace2_VAE_encoder_128": "autoenc-VGGFace2",
    "VGG16_BFM_identity_128": "faceID-BFM",
    "VGG16_BFM_VAE_encoder_128": "autoenc-BFM",
    "VGG16_ImageNet_128": "objCat-ImageNet",
    "VGG16_BFM_128": "invRend-BFM",
}


class VisionModel:
    def __init__(self, instance_id=0, model_class=None):
        super().__init__()
        self.instance_id = instance_id
        self.model_class = model_class

    def load(self, device):
        raise NotImplementedError

    def preprocess(self, x: torch.Tensor):
        """input image scaling and color normalization
        If image size is different than im_size, rescale it to fit im_size.
        Then, the image is normalized according to channel_normalization_fun.

        Face cropping is not applied by this function.

        args:
        x (torch.tensor) float image tensor (NCHW)
        """

        if x.ndim == 3:  # add a missing batch dimension
            x = x.unsqueeze(0)

        # image rescaling
        N, C, H, W = x.shape
        assert H == W, "preprocess only supports square images."
        if H != self.input_im_size:
            # print('mismatching image size:',x.shape)
            x = torch.nn.functional.interpolate(
                x, size=self.input_im_size, mode="bilinear", align_corners=True
            )
            # print('downscaled to',x.shape)

        if not hasattr(self, "channel_normalization_fun"):
            self.channel_normalization_fun = preprocessing_funs[
                self.channel_normalization_fun_name
            ]

        # color normalization
        x = self.channel_normalization_fun(x)
        return x

    def forward(self, x):
        return self.model(self.preprocess(x.to(self.device)))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def modules(self, *args, **kwargs):
        return self.model.modules(*args, **kwargs)


class LightningModel(VisionModel):
    def __init__(
        self,
        instance_id=0,
        model_class=None,
        pl_model_class=None,
        pl_model_class_kwd=None,
    ):
        super().__init__(instance_id=instance_id, model_class=model_class)
        self.pl_model_class = pl_model_class

        if pl_model_class_kwd is None:
            pl_model_class_kwd = {}
        self.pl_model_class_kwd = pl_model_class_kwd

    def _get_checkpoint_filename(self):
        """
        Return the checkpoint filename inside the Hugging Face repository.

        Expected repository structure:
            faceID-BFM/instance00.ckpt
            faceID-BFM/instance01.ckpt
            faceID-BFM/instance02.ckpt
            ...
        """
        class_name = type(self).__name__

        if class_name not in MODEL_MAP:
            raise KeyError(
                f"No Hugging Face checkpoint folder is defined for {class_name}. "
                f"Add this class to MODEL_MAP."
            )

        if self.instance_id not in [0, 1, 2]:
            raise ValueError(
                f"instance_id must be 0, 1, or 2, but got {self.instance_id}."
            )

        model_folder = MODEL_MAP[class_name]
        return f"{model_folder}/instance{self.instance_id:02d}.ckpt"

    def get_checkpoint_path(
        self,
        checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
        repo_id=HF_MODEL_REPO_ID,
        force_download=False,
    ):
        """
        Download the checkpoint from Hugging Face if needed, and return
        the local checkpoint path.

        The file is saved under:
            model_checkpoints/<model_folder>/instanceXX.ckpt
        """
        checkpoint_dir = Path(checkpoint_dir).expanduser()
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = REPO_ROOT / checkpoint_dir
        filename = self._get_checkpoint_filename()
        local_path = checkpoint_dir / filename

        if local_path.exists() and not force_download:
            return str(local_path)

        local_path.parent.mkdir(parents=True, exist_ok=True)

        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            local_dir=str(checkpoint_dir),
            force_download=force_download,
        )

    def load(
        self,
        device,
        checkpoint_path=None,
        checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
        repo_id=HF_MODEL_REPO_ID,
        force_download=False,
    ):
        if checkpoint_path is None:
            checkpoint_path = self.get_checkpoint_path(
                checkpoint_dir=checkpoint_dir,
                repo_id=repo_id,
                force_download=force_download,
            )

        self.model = self.pl_model_class.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
            **self.pl_model_class_kwd,
        ).to(device)

        self.model.eval()
        self.device = device
        self.checkpoint_path = checkpoint_path


class VGG16_pl_model(LightningModel):
    """a parent class for VGG16-based models trained with pytorch-lightning"""

    def load(self, device, checkpoint_path=None, **kwargs):
        super().load(
            device,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )

        self.layer_names = VGG16_LAYER_NAMES

        self.layer_modules = (
            [*self.model.model.features]
            + [self.model.model.avgpool]
            + [
                layer
                for layer in self.model.model.classifier
                if not "Dropout" in str(layer)
            ]
        )

        self.layer_nums = [
            list(self.model.modules()).index(m) for m in self.layer_modules
        ]
        assert len(self.layer_names) == len(self.layer_modules)


class VGG16_VGGFace2_128(VGG16_pl_model):
    def __init__(self, instance_id=0):
        super().__init__(
            instance_id=instance_id, model_class="VGGFace2", pl_model_class=VGG16
        )
        self.model_name = "VGG16_Face2"
        self.channel_normalization_fun_name = "vggface2_transform"
        self.input_im_size = 128


class VGG16_ImageNet_128(VGG16_pl_model):
    def __init__(self, instance_id=0):
        super().__init__(
            instance_id=instance_id, model_class="ImageNet", pl_model_class=VGG16
        )
        self.model_name = "VGG16_Object"
        self.channel_normalization_fun_name = "torchvision"
        self.input_im_size = 128


class VGG16_BFM_128(VGG16_pl_model):
    def __init__(self, instance_id=0):
        super().__init__(
            instance_id=instance_id,
            model_class="BFM_vgg16",
            pl_model_class=BFM_MODEL,
            pl_model_class_kwd={"model_class": "vgg16"},
        )
        self.model_name = "VGG16_BFM"
        self.channel_normalization_fun_name = "bfm_transform"
        self.input_im_size = 128


class VGG16_BFM_identity_128(VGG16_pl_model):
    def __init__(self, instance_id=0):
        super().__init__(
            instance_id=instance_id,
            model_class="BFM_identity_vgg16",
            pl_model_class=VGG16,
            pl_model_class_kwd={"model_class": "vgg16"},
        )
        self.model_name = "VGG16_BFMidentity"
        self.channel_normalization_fun_name = "bfm_transform"
        self.input_im_size = 128


class VGG16_VGGFace2_VAE_encoder_128(LightningModel):
    def __init__(self, instance_id=0):
        super().__init__(
            instance_id=instance_id,
            model_class="sVAE_module",
            pl_model_class=sVAE_module,
        )
        self.model_name = "sVAE_module"
        self.channel_normalization_fun_name = "vggface2_transform"
        self.input_im_size = 128

    def load(self, device, checkpoint_path=None, **kwargs):
        super().load(
            device,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )

        # we use only the VAE encoder
        del self.model.decoder, self.model.fc12, self.model.fc2, self.model.log_sigma

        self.layer_names = VGG16_LAYER_NAMES
        self.layer_modules = (
            list(self.model.encoder.features)
            + [self.model.encoder.avgpool]
            + [
                self.model.encoder.fc1,
                self.model.encoder.fc1_relu,
                self.model.encoder.fc2,
                self.model.encoder.fc2_relu,
            ]
            + [self.model.fc11]
        )

        assert len(self.layer_names) == len(self.layer_modules)

        # get layer numbers in .modules() according to the old convention
        self.layer_nums = [
            list(self.model.modules()).index(m) for m in self.layer_modules
        ]

        # print layer names and modules for debugging:
        # [print(n,'|',m) for n,m in zip(self.layer_names, self.layer_modules)]

        # change the forward function (originally, it includes decoding)
        self.model.forward = lambda x: self.model.fc11(self.model.encoder(x))
        self.model.encoder.eval()


class VGG16_BFM_VAE_encoder_128(LightningModel):
    def __init__(self, instance_id=0):
        super().__init__(
            instance_id=instance_id,
            model_class="sVAE_module",
            pl_model_class=sVAE_module,
        )
        self.model_name = "sVAE_module"
        self.channel_normalization_fun_name = "bfm_transform"
        self.input_im_size = 128

    def load(self, device, checkpoint_path=None, **kwargs):
        super().load(
            device,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )

        # we use only the VAE encoder
        del self.model.decoder, self.model.fc12, self.model.fc2, self.model.log_sigma

        self.layer_names = VGG16_LAYER_NAMES
        self.layer_modules = (
            list(self.model.encoder.features)
            + [self.model.encoder.avgpool]
            + [
                self.model.encoder.fc1,
                self.model.encoder.fc1_relu,
                self.model.encoder.fc2,
                self.model.encoder.fc2_relu,
            ]
            + [self.model.fc11]
        )

        assert len(self.layer_names) == len(self.layer_modules)

        # get layer numbers in .modules() according to the old convention
        self.layer_nums = [
            list(self.model.modules()).index(m) for m in self.layer_modules
        ]

        # print layer names and modules for debugging:
        # [print(n,'|',m) for n,m in zip(self.layer_names, self.layer_modules)]

        # change the forward function (originally, it includes decoding)
        self.model.forward = lambda x: self.model.fc11(self.model.encoder(x))
        self.model.encoder.eval()
