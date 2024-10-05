import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import GPT2Tokenizer

from tqdm import tqdm


class TinyStoriesDataset(Dataset):
    def __init__(self, context_len: int, num_stories=50000):
        super().__init__()
        self.context_len = context_len

        # Download tokenizer
        tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(
            "openai-community/gpt2",
            clean_up_tokenization_spaces=False,
            add_bos_token=True,
        )
        tokenizer.model_max_length = 2048  # Silence warnings about long string

        # Download & Process data
        tokens: list[int] = []
        ds = load_dataset(
            "roneneldan/TinyStories",
            data_files="data/train-00000-of-00004-2d5a1467fff1081b.parquet",
        )
        for story in tqdm(
            ds["train"][:num_stories]["text"],
            desc="Tokenizing Stories",
            unit=" stories",
        ):
            story = " ".join(story.split())  # Clean up whitespace
            tokens.extend(tokenizer.encode(story))

        self.tokens = torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, index: int):
        start = index * self.context_len
        end = start + self.context_len
        sample = self.tokens[start:end]
        label = self.tokens[start + 1 : end + 1]
        return sample, label

    def __len__(self):
        # len+1 so that there's space for the label
        return (self.tokens.shape[0] + 1) // self.context_len
