import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms

from .configs import DataConfig


def class_prompt(name: str, template: str) -> str:
    label = name.split(",")[0].strip().replace("_", " ")

    return template.format(label)


class Stream(IterableDataset):
    def __init__(self, cfg: DataConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.seed = seed
        self.epoch = 0

        self.transform = transforms.Compose(
            [
                transforms.Resize(cfg.resolution),
                transforms.CenterCrop(cfg.resolution),
                transforms.ToTensor(),
            ]
        )

        self.stream = load_dataset(cfg.dataset, split=cfg.split, streaming=True)

        features = self.stream.features
        if features is None or cfg.label_key not in features:
            raise ValueError(
                f"{cfg.dataset} exposes no '{cfg.label_key}' feature; "
                "set data.label_key to the class column"
            )

        self.names = features[cfg.label_key].names

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        stream = self.stream.shuffle(
            seed=self.seed, buffer_size=self.cfg.shuffle_buffer
        )

        info = get_worker_info()
        if info is not None and info.num_workers > 1:
            stream = stream.shard(num_shards=info.num_workers, index=info.id)

        stream.set_epoch(self.epoch)

        for example in stream:
            image = example[self.cfg.image_key]
            label = example[self.cfg.label_key]

            yield (
                self.transform(image.convert("RGB")),
                class_prompt(self.names[label], self.cfg.prompt_template),
            )


def collate(batch, tokenizer, max_tokens: int):
    pixels = torch.stack([item[0] for item in batch])
    inputs = tokenizer(
        [item[1] for item in batch],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_tokens,
    )

    return pixels, inputs["input_ids"], inputs["attention_mask"]


def collate_pixels(batch):
    return (torch.stack([item[0] for item in batch]),)
