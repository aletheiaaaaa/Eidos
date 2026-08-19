"""Lumina: a latent diffusion transformer trained with mean-flow matching.

Only the config types are re-exported here, so that `lumina --help` does not pay
for importing torch. The heavy pieces live in submodules and are imported
directly:

    from lumina.nn.model import DiT, Diffuser
    from lumina.data import H5Dataset, process_data
    from lumina.train import train
"""

from .configs import (
    Config,
    DataConfig,
    DiffuserConfig,
    DiTConfig,
    TrainConfig,
    load_config,
)

__all__ = [
    "Config",
    "DataConfig",
    "DiTConfig",
    "DiffuserConfig",
    "TrainConfig",
    "load_config",
]
