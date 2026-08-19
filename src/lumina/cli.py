import argparse
from pathlib import Path

from .configs import Config, load_config

DEFAULT_CONFIG = "config.yaml"


def _cmd_data(cfg: Config, args: argparse.Namespace) -> None:
    from .data import process_data

    process_data(cfg.data)


def _cmd_train(cfg: Config, args: argparse.Namespace) -> None:
    from .nn.model import DiT
    from .train import train

    model = DiT(cfg.diffuser)

    if cfg.diffuser.model_path and not args.resume:
        import torch

        state = torch.load(
            cfg.diffuser.model_path, map_location="cpu", weights_only=True
        )
        for key in ("ema", "model"):
            if key in state:
                state = state[key]
                break
        model.load_state_dict(state)
        print(f"initialized from {cfg.diffuser.model_path}")

    train(model, cfg.train, cfg.data, diffuser=cfg.diffuser, resume=args.resume)


def _cmd_generate(cfg: Config, args: argparse.Namespace) -> None:
    import torch
    from torchvision.utils import save_image

    from .nn.model import Diffuser

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint:
        cfg.diffuser.model_path = args.checkpoint
    if not cfg.diffuser.model_path:
        raise SystemExit(
            "no denoiser weights: pass --checkpoint or set diffuser.model_path"
        )

    diffuser = Diffuser(cfg.diffuser, device=device)
    images = diffuser.generate(
        args.prompt,
        num_images=args.num_images,
        num_steps=args.num_steps,
        guidance=args.guidance,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        save_image(image.float(), out_dir / f"sample_{i:03d}.png")

    print(f"wrote {len(images)} image(s) to {out_dir}")


def _cmd_config(cfg: Config, args: argparse.Namespace) -> None:
    import dataclasses
    import json

    print(json.dumps(dataclasses.asdict(cfg), indent=2))


def _apply_overrides(cfg: Config, overrides: list[str]) -> None:
    from .configs import _build

    for override in overrides:
        if "=" not in override:
            raise SystemExit(f"--set expects section.key=value, got {override!r}")

        dotted, _, raw = override.partition("=")
        keys = dotted.strip().split(".")
        if len(keys) < 2:
            raise SystemExit(f"--set expects a section-qualified key, got {dotted!r}")

        target = cfg
        for key in keys[:-1]:
            if not hasattr(target, key):
                raise SystemExit(f"--set: unknown key {dotted!r}")
            target = getattr(target, key)

        leaf = keys[-1]
        if not hasattr(target, leaf):
            raise SystemExit(f"--set: unknown key {dotted!r}")

        parent = ".".join(keys[:-1])
        try:
            checked = _build(type(target), {leaf: raw}, parent)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"--set: {exc}") from None

        setattr(target, leaf, getattr(checked, leaf))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lumina", description="Train and sample the Lumina latent diffusion model."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=DEFAULT_CONFIG,
        help=f"path to the YAML config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a config value, e.g. --set train.lr=3e-4 (repeatable)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("data", help="encode the source dataset into latent shards")
    p_data.set_defaults(func=_cmd_data)

    p_train = sub.add_parser("train", help="train the denoiser on the latent shards")
    p_train.add_argument(
        "--resume",
        help="checkpoint to restore model, optimizer, scheduler and EMA from",
    )
    p_train.set_defaults(func=_cmd_train)

    p_gen = sub.add_parser("generate", help="sample images from a trained denoiser")
    p_gen.add_argument("prompt", help="text prompt")
    p_gen.add_argument("-n", "--num-images", type=int, default=4)
    p_gen.add_argument("-s", "--num-steps", type=int, default=2)
    p_gen.add_argument(
        "-g", "--guidance", type=float, default=3.0, help="cfg scale, 1.0 disables"
    )
    p_gen.add_argument("-o", "--output-dir", default="./samples")
    p_gen.add_argument("--checkpoint", help="overrides diffuser.model_path")
    p_gen.add_argument("--device", help="torch device (default: cuda if available)")
    p_gen.set_defaults(func=_cmd_generate)

    p_cfg = sub.add_parser("config", help="print the resolved config and exit")
    p_cfg.set_defaults(func=_cmd_config)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_config(args.config)
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"config error: {exc}") from None

    _apply_overrides(cfg, args.overrides)

    args.func(cfg, args)


if __name__ == "__main__":
    main()
