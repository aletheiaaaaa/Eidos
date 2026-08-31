import argparse
from pathlib import Path

from .configs import Config, latest_checkpoint, load_config

DEFAULT_CONFIG = "config.yaml"


def _cmd_train_denoiser(cfg: Config, args: argparse.Namespace) -> None:
    from .nn.model import DiT
    from .train import train_denoiser

    model = DiT(cfg.diffuser)

    train_denoiser(
        model,
        cfg.diffuser.train,
        cfg.data,
        diffuser=cfg.diffuser,
        decoder=cfg.decoder,
        resume=args.resume,
    )


def _cmd_train_decoder(cfg: Config, args: argparse.Namespace) -> None:
    from .nn.latents import Decoder
    from .train import train_decoder

    decoder = Decoder(cfg.decoder, cfg.diffuser.img_size)

    stats = args.stats or latest_checkpoint(cfg.diffuser.train.output_dir)

    train_decoder(
        decoder,
        cfg.decoder.train,
        cfg.data,
        resume=args.resume,
        stats=stats or None,
    )


def _cmd_generate(cfg: Config, args: argparse.Namespace) -> None:
    import torch
    from torchvision.utils import save_image

    from .nn.model import Diffuser

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    path = args.checkpoint or latest_checkpoint(cfg.diffuser.train.output_dir)
    if not path:
        raise SystemExit(
            f"no denoiser weights in {cfg.diffuser.train.output_dir}; pass --checkpoint"
        )

    if not args.checkpoint:
        print(f"using {path}")

    diffuser = Diffuser(cfg.diffuser, cfg.decoder, device=device, model_path=path)
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


def _cmd_bench_data(cfg: Config, args: argparse.Namespace) -> None:
    import time

    from datasets import load_dataset
    from torch.utils.data import DataLoader

    from .data import Stream, collate_pixels

    data = cfg.data

    start = time.perf_counter()
    raw = load_dataset(data.dataset, split=data.split, streaming=True)
    print(
        f"resolved {data.dataset}:{data.split} in "
        f"{time.perf_counter() - start:.1f}s, num_shards={raw.num_shards}"
    )
    if raw.num_shards < args.workers:
        print(
            f"warning: num_shards={raw.num_shards} < workers={args.workers}; "
            f"{args.workers - raw.num_shards} workers will sit idle"
        )

    stream = iter(raw)
    start = time.perf_counter()
    next(stream)
    print(f"first raw example in {time.perf_counter() - start:.1f}s")

    start = time.perf_counter()
    for _ in range(args.num_examples):
        next(stream)
    elapsed = time.perf_counter() - start
    print(
        f"{args.num_examples} raw examples in {elapsed:.1f}s "
        f"({args.num_examples / elapsed:.1f}/s)"
    )

    start = time.perf_counter()
    next(iter(raw.shuffle(seed=0, buffer_size=data.shuffle_buffer)))
    print(
        f"first shuffled example (buffer {data.shuffle_buffer}) in "
        f"{time.perf_counter() - start:.1f}s"
    )

    loader = DataLoader(
        Stream(data),
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        collate_fn=collate_pixels,
    )
    start = time.perf_counter()
    pixels = next(iter(loader))[0]
    print(
        f"first batch of {args.batch_size} via {args.workers} workers in "
        f"{time.perf_counter() - start:.1f}s {tuple(pixels.shape)}"
    )


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

    sub = parser.add_subparsers(dest="command", required=True)

    p_denoise = sub.add_parser(
        "train-denoiser", help="train the denoiser on the streamed dataset"
    )
    p_denoise.add_argument(
        "--resume",
        help="checkpoint to restore model, optimizer, scheduler and EMA from",
    )
    p_denoise.set_defaults(func=_cmd_train_denoiser)

    p_dec = sub.add_parser(
        "train-decoder",
        help="train the ViT decoder that turns latents back into pixels",
    )
    p_dec.add_argument(
        "--resume",
        help="checkpoint to restore decoder, optimizer, scheduler and EMA from",
    )
    p_dec.add_argument(
        "--stats",
        help="checkpoint to copy encoder statistics from "
        "(defaults to the latest denoiser checkpoint, else fits its own)",
    )
    p_dec.set_defaults(func=_cmd_train_decoder)

    p_gen = sub.add_parser("generate", help="sample images from a trained denoiser")
    p_gen.add_argument("prompt", help="text prompt")
    p_gen.add_argument("-n", "--num-images", type=int, default=4)
    p_gen.add_argument("-s", "--num-steps", type=int, default=2)
    p_gen.add_argument(
        "-g", "--guidance", type=float, default=3.0, help="cfg scale, 1.0 disables"
    )
    p_gen.add_argument("-o", "--output-dir", default="./samples")
    p_gen.add_argument(
        "--checkpoint", help="overrides the latest checkpoint in the output dir"
    )
    p_gen.add_argument("--device", help="torch device (default: cuda if available)")
    p_gen.set_defaults(func=_cmd_generate)

    p_cfg = sub.add_parser("config", help="print the resolved config and exit")
    p_cfg.set_defaults(func=_cmd_config)

    p_bench = sub.add_parser(
        "bench-data", help="time the streaming pipeline stage by stage"
    )
    p_bench.add_argument("-n", "--num-examples", type=int, default=200)
    p_bench.add_argument("-b", "--batch-size", type=int, default=32)
    p_bench.add_argument("-w", "--workers", type=int, default=8)
    p_bench.set_defaults(func=_cmd_bench_data)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_config(args.config)
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"config error: {exc}") from None

    args.func(cfg, args)


if __name__ == "__main__":
    main()
