import os
from pathlib import Path
from typing import Dict, List, Optional

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from omegaconf import DictConfig, OmegaConf


def get_env(env_name: str, default: Optional[str] = None) -> str:
    """
    Safely read an environment variable.
    Raises errors if it is not defined or it is empty.
    :param env_name: the name of the environment variable
    :param default: the default (optional) value for the environment variable
    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        if default is None:
            raise KeyError(f"{env_name} not defined and no default value is present!")
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            raise ValueError(
                f"{env_name} has yet to be configured and no default value is present!"
            )
        return default

    return env_value


def load_envs(env_file: Optional[str] = None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.
    It is possible to define all the system specific variables in the `env_file`.
    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


def render_images(
    batch: torch.Tensor, nrow=8, title: str = "Images", autoshow: bool = True
) -> np.ndarray:
    """
    Utility function to render and plot a batch of images in a grid
    :param batch: batch of images
    :param nrow: number of images per row
    :param title: title of the image
    :param autoshow: if True calls the show method
    :return: the image grid
    """
    image = (
        torchvision.utils.make_grid(
            batch.detach().cpu(), nrow=nrow, padding=2, normalize=True
        )
        .permute((1, 2, 0))
        .numpy()
    )

    if autoshow:
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title(title)
        plt.imshow(image)
        plt.show()
    return image


def iterate_elements_in_batches(
    outputs: List[Dict[str, torch.Tensor]], batch_size: int, n_elements: int
) -> Dict[str, torch.Tensor]:
    """
    Iterate over elements across multiple batches in order, independently to the
    size of each batch
    :param outputs: a list of outputs dictionaries
    :param batch_size: the size of each batch
    :param n_elements: the number of elements to iterate over
    :return: yields one element at the time
    """
    count = 0
    for output in outputs:
        for i in range(batch_size):
            count += 1
            if count >= n_elements:
                return
            yield {
                key: value if len(value.shape) == 0 else value[i]
                for key, value in output.items()
            }


STATS_KEY: str = "stats"


# Adapted from https://github.com/hobogalaxy/lightning-hydra-template/blob/6bf03035107e12568e3e576e82f83da0f91d6a11/src/utils/template_utils.py#L125
def log_hyperparameters(
    cfg: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionally saves:
        - sizes of train, val, test dataset
        - number of trainable model parameters
    Args:
        cfg (DictConfig): [description]
        model (pl.LightningModule): [description]
        trainer (pl.Trainer): [description]
    """
    hparams = OmegaConf.to_container(cfg, resolve=True)

    # save number of model parameters
    hparams[f"{STATS_KEY}/params_total"] = sum(p.numel() for p in model.parameters())
    hparams[f"{STATS_KEY}/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams[f"{STATS_KEY}/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # (this is just a trick to prevent trainer from logging hparams of model, since we already did that above)
    trainer.logger.log_hyperparams = lambda params: None


# Load environment variables
load_envs()

# Set the cwd to the project root
PROJECT_ROOT: Path = Path(get_env("PROJECT_ROOT"))
assert (
    PROJECT_ROOT.exists()
), "You must configure the PROJECT_ROOT environment variable in a .env file!"

os.chdir(PROJECT_ROOT)