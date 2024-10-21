import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .. import modules, utils, data
from .train import train_gpt2


def main():
    # Seed Randomness
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # DDP Setup
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()

    # Model Params
    batch_size = 64
    micro_batch_size = 8
    num_heads = 12
    embed_dim = 768
    context_len = 1024
    vocab_size = utils.round_to_multiple(50257, 64)

    # Set up datasets
    train_ds = data.TokenizedDataset(
        1024, split="train", rank=rank, world_size=world_size
    )
    eval_ds = data.TokenizedDataset(
        1024, split="validation", rank=rank, world_size=world_size
    )

    # Set up model
    model = modules.GPT2(
        vocab_size,
        embed_dim,
        context_len,
        num_heads,
        use_flash_attention=True,
    ).to(device)
    model = torch.compile(model)
    ddp_model = DDP(model, device_ids=[device])

    train_gpt2(
        ddp_model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        num_epochs=2,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        device=device,
        enable_tf32=True,
        enable_bf16_amp=True,
        logging_interval=10,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
