import torch
from datetime import timedelta
import os

_TENSOR_MODEL_PARALLEL_GROUP = None
_TENSOR_MODEL_PARALLEL_RANK = None
_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None

def model_parallel_is_initialized():
    if _TENSOR_MODEL_PARALLEL_GROUP is None:
        return False
    return True

def initialize_tensor_parallel(tensor_model_parallel_size: int = 1,
            distributed_backend: str = "nccl",
            distributed_timeout_minutes: float = 1.):
    """
    INPUTS
        (1) tensor_model_parallel_size, int, the number of GPUs to split
            tensors across.
    """
    
    """
    The environment variables RANK and WORLD_SIZE are made available
    because we use torchrun to run our script. See the link for more:
    https://pytorch.org/docs/stable/elastic/run.html#environment-variables
    
    Some definitions:
        RANK: rank of a worker within a worker group
        WORLD_SIZE: total number of workers in a worker group
    A worker group is a set of workers that execute the same function,
    e.g. a training function.
    """

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = int(os.getenv('RANK', '0'))
        world_size = int(os.getenv("WORLD_SIZE", '1'))
        if device_count > 0:
            device = rank % device_count
            torch.cuda.set_device(device)
    torch.distributed.init_process_group(
        backend=distributed_backend,
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=distributed_timeout_minutes)
        )
    
    if not model_parallel_is_initialized():
        # Set the tensor model-parallel communicators
        """
        The world_size is the total number of GPUs. If we have 6 GPUs and 
        tensor_model_parallel_size is 2, then we have 3 (=6/2) groups. Each
        group has a full copy of the tensors that we are trying to split
        across 2 GPUs.
        """
        num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
        global _TENSOR_MODEL_PARALLEL_GROUP
        for i in range(num_tensor_model_parallel_groups):
            ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _TENSOR_MODEL_PARALLEL_GROUP = group

def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor model parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _TENSOR_MODEL_PARALLEL_GROUP is not None
        ), 'tensor model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP

def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _TENSOR_MODEL_PARALLEL_RANK
    if _TENSOR_MODEL_PARALLEL_RANK is not None:
        return _TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())

def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())

def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size
