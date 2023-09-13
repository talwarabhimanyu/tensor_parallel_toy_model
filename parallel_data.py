import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from parallel_utils import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_src_rank
        )

class ToyDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        """
        Returning a single tensor so that I can use
        torch.distributed.broadcast which operates on a single
        tensor. An alternative is torch.distributed.broadcast_object_list
        which PyTorch notes is insecure.
        """
        return torch.from_numpy(np.hstack(
                                    (
                                    self.X[idx,:], 
                                    self.y[idx]
                                    )
                                )
                            )

def get_dataloaders(data_dir, batch_size):
    """
    DataLoaders will be built only on rank 0 of a model
    parallel group. The same batches will be broadcast 
    out to other ranks.
    """
    train_loader, valid_loader = None, None
    if get_tensor_model_parallel_rank() == 0:
        X_train = np.load(os.path.join(data_dir, "xtrain.npy"))
        y_train = np.load(os.path.join(data_dir, "ytrain.npy"))
        X_valid = np.load(os.path.join(data_dir, "xvalid.npy"))
        y_valid = np.load(os.path.join(data_dir, "yvalid.npy"))
        
        train_loader = DataLoader(dataset=ToyDataset(
                                                X_train,
                                                y_train
                                        ),
                                batch_size=batch_size,
                                shuffle=False)
        valid_loader = DataLoader(dataset=ToyDataset(
                                                X_valid,
                                                y_valid
                                        ),
                                batch_size=batch_size,
                                shuffle=False)
    return train_loader, valid_loader

def get_batch(dataloader, batch_size, input_size, datatype):
    if dataloader is not None:
        batch = next(iter(dataloader))
    else:
        batch = None

    if get_tensor_model_parallel_rank() == 0:
        flatten_data = batch.contiguous().view(-1).cuda()
    else:
        flatten_data = torch.empty(batch_size*(input_size+1),
                        device=torch.cuda.current_device(),
                        dtype=datatype
                        )
    # Broadcast
    torch.distributed.broadcast(
        flatten_data, get_tensor_model_parallel_src_rank(), group=get_tensor_model_parallel_group()
    )

    data_b = flatten_data.reshape(batch_size, input_size+1)
    inputs = data_b[:,:-1]
    targets = data_b[:,-1]
    return inputs, targets
