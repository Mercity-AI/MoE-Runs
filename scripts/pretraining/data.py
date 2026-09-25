"""Shared streaming FineWeb packing for the baseline and LongCat runs."""

import hashlib
from typing import Optional

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def is_eval_document(example: dict, holdout_fraction: float) -> bool:
    """Assign a document to validation using a stable content/ID hash."""
    identity = example.get("id") or example.get("url") or example.get("text", "")
    digest = hashlib.blake2b(
        str(identity).encode("utf-8", errors="ignore"), digest_size=8
    ).digest()
    bucket = int.from_bytes(digest, byteorder="big") / float(2**64)
    return bucket < holdout_fraction


def is_document_in_partition(
    example: dict, holdout_fraction: float, partition: str
) -> bool:
    selected_for_eval = is_eval_document(example, holdout_fraction)
    return selected_for_eval if partition == "eval" else not selected_for_eval


class PackedFineWebDataset(IterableDataset):
    """Stream, partition, tokenize, and pack FineWeb into fixed token blocks."""

    def __init__(
        self,
        config: dict,
        tokenizer,
        seed: int = 42,
        max_examples: Optional[int] = None,
        start_batch: int = 0,
        partition: str = "train",
        rank: int = 0,
        world_size: int = 1,
        base_dataset=None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = config["max_seq_len"]
        self.eos_id = tokenizer.eos_id
        self.seed = seed
        self.epoch = 0
        self.buffer_size = config["streaming_buffer_size"]
        self.batch_size = config["per_device_batch_size"]
        self.max_examples = max_examples
        self.start_batch = start_batch
        self.partition = partition
        if partition == "eval":
            self.buffer_size = config.get("eval_streaming_buffer_size", 1)
        self.holdout_fraction = config.get("eval_holdout_fraction", 0.005)
        self.rank = rank
        self.world_size = world_size

        if partition not in {"train", "eval"}:
            raise ValueError(f"partition must be 'train' or 'eval', got {partition!r}")
        if not 0.0 < self.holdout_fraction < 1.0:
            raise ValueError("eval_holdout_fraction must be between 0 and 1.")

        if base_dataset is not None:
            self.base_dataset = base_dataset
        else:
            from datasets import load_dataset

            self.base_dataset = load_dataset(
                config["dataset_name"],
                name=config["dataset_config"],
                split="train",
                streaming=True,
            )

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        from datasets.distributed import split_dataset_by_node

        # HF IterableDataset handles DataLoader worker sharding itself. Split
        # only across training processes here to avoid dropping data twice.
        dataset = split_dataset_by_node(
            self.base_dataset, rank=self.rank, world_size=self.world_size
        )
        dataset = dataset.filter(
            is_document_in_partition,
            fn_kwargs={
                "holdout_fraction": self.holdout_fraction,
                "partition": self.partition,
            },
        )
        dataset = dataset.shuffle(
            seed=self.seed + self.epoch,
            buffer_size=self.buffer_size,
        )
        dataset_iter = iter(dataset)

        buffer = []
        examples_yielded = 0
        examples_skipped = 0
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        def worker_share(total: Optional[int]) -> Optional[int]:
            if total is None:
                return None
            quotient, remainder = divmod(total, num_workers)
            return quotient + int(worker_id < remainder)

        local_max_examples = worker_share(self.max_examples)
        skipped_batches, extra_workers = divmod(self.start_batch, num_workers)
        local_start_example = (
            skipped_batches + int(worker_id < extra_workers)
        ) * self.batch_size

        while True:
            if local_max_examples is not None and examples_yielded >= local_max_examples:
                return

            while len(buffer) < self.max_seq_len:
                try:
                    doc = next(dataset_iter)
                except StopIteration:
                    return

                text = doc.get("text", "")
                if not text or not text.strip():
                    continue
                tokens = self.tokenizer.encode(text, add_bos=False, add_eos=False)
                if tokens:
                    buffer.extend(tokens)
                    buffer.append(self.eos_id)

            chunk = buffer[: self.max_seq_len]
            buffer = buffer[self.max_seq_len :]
            if examples_skipped < local_start_example:
                examples_skipped += 1
                continue

            yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}
            examples_yielded += 1


def build_dataloader(
    dataset: IterableDataset, batch_size: int, config: dict
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "pin_memory": torch.cuda.is_available(),
    }
    if config["dataloader_workers"] > 0:
        kwargs.update(
            num_workers=config["dataloader_workers"],
            persistent_workers=True,
            prefetch_factor=config["dataloader_prefetch_factor"],
        )
    return DataLoader(dataset, **kwargs)


__all__ = [
    "PackedFineWebDataset",
    "build_dataloader",
    "is_document_in_partition",
    "is_eval_document",
]
