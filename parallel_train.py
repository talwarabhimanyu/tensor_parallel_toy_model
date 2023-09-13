import torch
import torch.nn as nn
from parallel_utils import (
        initialize_tensor_parallel,
        get_tensor_model_parallel_rank
        )
from parallel_model import ToyModelTensorParallel
from parallel_data import (
        get_dataloaders,
        get_batch
        )
import argparse

def train_tensor_parallel():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_size", type=int)
    parser.add_argument("--tensor_model_parallel_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--init_lr", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--num_iterations", type=int)
    parser.add_argument("--update_freq", type=int, default=50)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--distributed_timeout_minutes", type=float)
    parser.add_argument("--distributed_backend", type=str)
    args = parser.parse_args()

    initialize_tensor_parallel(
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            distributed_backend=args.distributed_backend,
            distributed_timeout_minutes=args.distributed_timeout_minutes
            )

    model = ToyModelTensorParallel(input_size=args.input_size,
                    hidden_size=32*args.input_size)
    model.cuda(torch.cuda.current_device())

    optimizer = torch.optim.SGD(model.parameters(),
                    lr=args.init_lr,
                    momentum=args.momentum
                )
    loss_fn = nn.MSELoss()

    train_loader, valid_loader = get_dataloaders(data_dir=args.data_dir,
                                    batch_size=args.batch_size
                                    )
    model.train()
    running_train_loss = 0.
    rank = get_tensor_model_parallel_rank()
    for i in range(args.num_iterations):
        inputs, targets = get_batch(dataloader=train_loader,
                            batch_size=args.batch_size,
                            input_size=args.input_size,
                            datatype=torch.float32
                            )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs[:,0], targets)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        if i % args.update_freq == args.update_freq-1:
            print(f"Rank {rank} Iter {i+1} Train Loss {running_train_loss:.3f}")
            running_train_loss = 0.

train_tensor_parallel()
