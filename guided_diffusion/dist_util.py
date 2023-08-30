"""
Helpers for distributed training.
"""

import io
import os

import blobfile as bf
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).

def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    rank = int(os.environ['RANK'])
    num_gpus = th.cuda.device_count()
    th.cuda.set_device(rank % num_gpus)
    
    backend = "gloo" if not th.cuda.is_available() else "nccl"
    dist.init_process_group(backend=backend)


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    return th.load(path, **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)

