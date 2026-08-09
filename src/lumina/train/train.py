import torch
from accelerate import Accelerator

accel = Accelerator(mixed_precision="fp8" if torch.cuda.is_available() else "no")
device = accel.device
