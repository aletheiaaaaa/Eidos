import torch
from model import MaskDiT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

