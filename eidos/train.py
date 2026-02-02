import torch
import einops
import wandb
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from .model import Diffuser
from .configs import TrainConfig
from .data import H5Dataset

accel = Accelerator(mixed_precision="bf16" if torch.cuda.is_available() else "no")
device = accel.device

