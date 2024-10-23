"""
Script for tokenizing the tiny stories dataset and saving it to huggingface
It's built to handle large datasets by using multiprocessing and streaming
That said, the actual TinyStories dataset is small enough that it's not necessary -
this script is more of a proof of concept for larger datasets

Usage:
    python src/data/make_tokenized_data.py \
        --hf_out_repo TinyStories \
        --hf_in_repo roneneldan/TinyStories \
"""

import argparse
from dataclasses import dataclass
import functools
import io
import multiprocessing as mp
from itertools import batched
from typing import Generator
from datasets import load_dataset, IterableDataset, IterableDatasetDict
from huggingface_hub import (
    CommitOperationAdd,
    preupload_lfs_files,
    create_commit,
    create_repo,
    login,
)
import polars as pl
from tqdm import tqdm
from transformers import GPT2Tokenizer


# CLI Arguments
@dataclass
class Namespace(argparse.Namespace):
    num_processes: int
    hf_out_repo: str
    hf_in_repo: str
    data_dir: str
    examples_per_shard: int
    hf_token: str | None
    private: bool
    exist_ok: bool
    split: str | None


parser = argparse.ArgumentParser()
parser.add_argument("--num_processes", type=int, default=4)
parser.add_argument("--hf_out_repo", type=str, default="TinyStories")
parser.add_argument("--hf_in_repo", type=str, default="roneneldan/TinyStories")
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--examples_per_shard", type=int, default=500000)
parser.add_argument("--hf_token", type=str, default=None)
parser.add_argument("--private", action="store_true")
parser.add_argument("--exist_ok", action="store_true")
parser.add_argument("--split", type=str, default=None)


# Globals
class ProcessGlobals:
    def __init__(self, namespace: type[Namespace]):
        self.tqdm_lock = mp.RLock()
        self.progress_bar_slots = mp.Array("B", namespace.num_processes)


def init_globals(globals: ProcessGlobals):
    global PROGRESS_BAR_SLOTS
    PROGRESS_BAR_SLOTS = globals.progress_bar_slots
    tqdm.set_lock(globals.tqdm_lock)


# Help with progress bar slots
class ProgressBarSlot:
    slot: int

    def __enter__(self) -> int:
        global PROGRESS_BAR_SLOTS

        with PROGRESS_BAR_SLOTS.get_lock():
            for i in range(len(PROGRESS_BAR_SLOTS)):
                if PROGRESS_BAR_SLOTS[i] == 0:
                    PROGRESS_BAR_SLOTS[i] = 1
                    self.slot = i
                    return i
        raise RuntimeError("No available progress bar slots")

    def __exit__(self, *args):
        global PROGRESS_BAR_SLOTS
        with PROGRESS_BAR_SLOTS.get_lock():
            PROGRESS_BAR_SLOTS[self.slot] = 0


# Dataset utils
def num_examples(dataset: IterableDataset) -> int:
    assert dataset.info.splits is not None
    val = dataset.info.splits[dataset.split].num_examples
    assert isinstance(
        val, int
    ), f"Expected int, got {type(val)}"  # Some datasets don't have this info
    return val


def calculate_num_shards(dataset: IterableDataset, examples_per_shard: int) -> int:
    return (num_examples(dataset) // examples_per_shard) + 1


# Tokenization and uploading
@dataclass
class ExampleShard:
    shard_num: int
    examples: list[str]


def tokenize_and_upload_shard(
    shard: ExampleShard,
    tokenizer: GPT2Tokenizer,
    data_dir: str,
    split_name: str,
    hf_repo_id: str,
    num_shards: int,
) -> CommitOperationAdd:
    with ProgressBarSlot() as slot:
        tokens = [
            token
            for tokenized_examples in tqdm(
                shard.examples,
                desc=f"Shard {shard.shard_num + 1}",  # 1-indexed for ux
                position=slot + 1,  # 0 is reserved for the main loop
                unit="example",
                leave=False,
            )
            for token in tokenizer.encode(tokenized_examples)
        ]

    parquet_file = io.BytesIO()
    pl.DataFrame({"tokens": tokens}, schema={"tokens": pl.UInt32}).write_parquet(
        parquet_file
    )
    parquet_file.seek(0)

    shard_name = f"{split_name}-{shard.shard_num:05}-of-{num_shards:05}.parquet"
    addition = CommitOperationAdd(
        path_in_repo=f"{data_dir}/{shard_name}", path_or_fileobj=parquet_file
    )

    preupload_lfs_files(hf_repo_id, additions=[addition], repo_type="dataset")

    return addition


# Main
if __name__ == "__main__":
    args = parser.parse_args(namespace=Namespace)
    process_globals = ProcessGlobals(args)
    init_globals(process_globals)

    print("Logging into HuggingFace...")
    login(
        token=args.hf_token,
        write_permission=True,
    )

    print("")
    print("Loading Tokenizer...")
    tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(
        "openai-community/gpt2",
        add_bos_token=True,
        clean_up_tokenization_spaces=False,  # https://github.com/huggingface/transformers/issues/31884
    )
    tokenizer.model_max_length = 2048  # Silence warnings

    print("Loading Dataset...")
    datasets = load_dataset(args.hf_in_repo, streaming=True)
    assert isinstance(datasets, IterableDatasetDict)

    print("Creating HuggingFace repository...")
    repo_id = create_repo(
        args.hf_out_repo,
        repo_type="dataset",
        private=args.private,
        exist_ok=args.exist_ok,
    ).repo_id

    splits = [args.split] if args.split else datasets.keys()
    for split in splits:
        print("")
        assert isinstance(split, str)
        dataset = datasets[split]
        assert isinstance(dataset, IterableDataset)

        num_shards = (num_examples(dataset) // args.examples_per_shard) + 1
        examples: Generator[str, None, None] = (row["text"] for row in dataset)
        sharded_examples = (
            ExampleShard(shard_num=i, examples=list(batch))
            for i, batch in enumerate(batched(examples, args.examples_per_shard))
        )
        handle_shard = functools.partial(
            tokenize_and_upload_shard,
            tokenizer=tokenizer,
            data_dir=args.data_dir,
            split_name=split,
            hf_repo_id=repo_id,
            num_shards=num_shards,
        )

        with mp.Pool(
            args.num_processes, initializer=init_globals, initargs=(process_globals,)
        ) as pool:
            repo_operations = list(
                tqdm(
                    pool.imap_unordered(
                        handle_shard,
                        sharded_examples,
                    ),
                    total=num_shards,
                    desc=f"Processing {split} split",
                    unit="split",
                    position=0,
                )
            )

        print(f"Uploading {len(repo_operations)} files for {split} split")
        create_commit(
            repo_id,
            operations=repo_operations,
            commit_message=f"Uploaded {len(repo_operations)} files for {split} split",
            repo_type="dataset",
        )
    print("")
    print("Tokenization and uploading complete!")
    print("")
