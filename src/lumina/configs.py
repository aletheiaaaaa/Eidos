from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DiTConfig:
    d_model: int = 512
    n_heads: int = 8
    d_head: int = 64
    d_mlp: int = 2048
    n_layers: int = 6


@dataclass
class DiffuserConfig:
    img_size: int = 64  # latent resolution, i.e. data.resolution // vae downsample
    patch_size: int = 2
    d_caption: int = 768
    n_channels: int = 4
    vae: str = "stabilityai/sd-vae-ft-mse"
    clip: str = "openai/clip-vit-large-patch14"
    model_path: str = ""

    dit: DiTConfig = field(default_factory=DiTConfig)


@dataclass
class DataConfig:
    dataset: str = "Fhrozen/relaion-art"
    split: str = "train"
    resolution: int = 512
    stream_batch_size: int = 16384
    save_dir: str = "./data"
    samples_per_shard: int = 10000
    vae: str = "stabilityai/sd-vae-ft-mse"
    clip: str = "openai/clip-vit-large-patch14"


@dataclass
class TrainConfig:
    p_mean: float = -0.4
    p_std: float = 1.0
    p_ratio: float = 0.25
    n_warmup: int = 5
    n_epochs: int = 200
    batch_size: int = 128
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-2
    mixed_precision: str = "bf16"
    save_interval: int = 50
    output_dir: str = "./checkpoints"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    diffuser: DiffuserConfig = field(default_factory=DiffuserConfig)


def _build(cls: type, raw: Any, path: str) -> Any:
    """Recursively instantiate the dataclass `cls` from a mapping."""
    if not isinstance(raw, dict):
        raise TypeError(f"{path or 'config'}: expected a mapping, got {type(raw).__name__}")

    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(
            f"{path or 'config'}: unknown key(s) {sorted(unknown)}; "
            f"expected any of {sorted(known)}"
        )

    kwargs = {}
    for name, value in raw.items():
        f = known[name]
        child = f"{path}.{name}" if path else name

        if is_dataclass(f.type) or (isinstance(f.type, str) and f.type in _NESTED):
            kwargs[name] = _build(_NESTED[f.type] if isinstance(f.type, str) else f.type, value, child)
            continue

        kwargs[name] = _coerce(f.type, value, child)

    return cls(**kwargs)


def _coerce(typ: Any, value: Any, path: str) -> Any:
    """Validate a scalar against its annotation, widening int -> float.

    Strings are parsed against the target type so that both a YAML 1.1 scalar
    that stays a string (`1e-4`) and a raw `--set` value land as the right type.
    """
    name = typ if isinstance(typ, str) else getattr(typ, "__name__", str(typ))

    if name == "float":
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise TypeError(f"{path}: expected a number, got {value!r}") from None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{path}: expected a number, got {value!r}")
        return float(value)

    if name == "int":
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                raise TypeError(f"{path}: expected an integer, got {value!r}") from None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{path}: expected an integer, got {value!r}")
        return value

    if name == "bool":
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        if not isinstance(value, bool):
            raise TypeError(f"{path}: expected a boolean, got {value!r}")
        return value

    if name == "str":
        if not isinstance(value, str):
            raise TypeError(f"{path}: expected a string, got {value!r}")
        return value

    return value


_NESTED = {
    "DataConfig": DataConfig,
    "TrainConfig": TrainConfig,
    "DiffuserConfig": DiffuserConfig,
    "DiTConfig": DiTConfig,
}


def load_config(path: str | Path) -> Config:
    """Load `config.yaml`. Missing keys fall back to the dataclass defaults."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no config at {path}; pass --config or create one at the project root"
        )

    raw = yaml.safe_load(path.read_text()) or {}
    return _build(Config, raw, "")
