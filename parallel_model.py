import torch
import torch.nn as nn
from parallel_utils import (
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_world_size,
        get_tensor_model_parallel_rank
        )

class LinearWithAsyncCommunication(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
            input,
            weight,
            bias,
            async_grad_allreduce):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.async_grad_allreduce = async_grad_allreduce
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output = output + bias
        """
        We use this implementation for both ColumnParallelLinear and RowParallelLinear.
        
        (1) In case of column parallel, the non-linearity (ReLU in
        our case) which comes after this layer can independently be
        applied to each parition's output. So we don't need to reduce
        output of each partitioned linear layer.
        
        (2) In case of row parallel, we handle the reduction in the
        RowParallelLinear class itself.
        """
        return output

    @staticmethod
    def backward(ctx, 
            grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        grad_input = grad_output.matmul(weight)
        grad_output = grad_output.contiguous()
        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            """
            From PyTorch docs: this reduces the tensor data across all machines in
            a way that all get the final reduced result. Default reduction op
            is SUM (torch.distributed.ReduceOp.SUM) specified by the parameter
            'op'.
            """
            handle = torch.distributed.all_reduce(
                grad_input, group=get_tensor_model_parallel_group(), async_op=True
            )
        grad_weight = grad_output.t().matmul(input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None
        if ctx.async_grad_allreduce:
            handle.wait()
        return grad_input, grad_weight, grad_bias, None, None, None

class ColumnParallelLinear(nn.Module):
    """
    Consider the affine transformation Y = XA + b. We can split A 
    of dimension input_size x output_size along the column dimension 
    as A = [A1 A2 ... Am] where each Ai is of dim input_size x output_size//m.
    Then YA = [YA1 YA2 ... YAm].
    """
    def __init__(self, 
            input_size: int, 
            output_size: int,
            add_bias: bool = False,
            skip_weight_param_allocation: bool = False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.add_bias = add_bias
        """
        World size here is the number of chunks we'll split the
        weight matrix into, along the output dimension.
        """
        world_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = output_size//world_size

        if not skip_weight_param_allocation:
            self.weight = nn.Parameter(
                            torch.empty(
                                self.output_size_per_partition,
                                self.input_size,
                                device=torch.cuda.current_device()
                                )
                            )
            """
            Here I initialize the weight matrix with some constant
            values. The sine and arange etc. are just to ensure the
            matrix entries are not all the same (so that loss optimization
            actually works).
            """
            rank = get_tensor_model_parallel_rank()
            init_weights = torch.sin(
                            torch.arange(
                                self.output_size_per_partition*self.input_size
                                ).reshape(self.weight.shape)*(0.5**rank)
                            )*0.1
            with torch.no_grad():
                self.weight.copy_(init_weights)
        else:
            self.weight = None

        if add_bias:
            self.bias = nn.Parameter(
                            torch.empty(
                                self.output_size_per_partition,
                                device=torch.cuda.current_device()
                            )
                        )
            with torch.no_grad():
                self.bias.zero_()
        else:
            """
            The second argument being None means the 'bias' parameter
            is not included in this module's state_dict.
            """
            self.register_parameter('bias', None)
        
    def forward(self, x):
        """
        Only positional args allowed. See my comment below
        in RowParallelLinear forward.
        """
        output = LinearWithAsyncCommunication.apply(
                        x, self.weight, self.bias, True
                    )
        return output

def _reduce(input_):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())

    return input_

class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)

class RowParallelLinear(nn.Module):
    def __init__(self,
            input_size: int,
            output_size: int,
            add_bias: bool = False,
            skip_weight_param_allocation: bool = False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.add_bias = add_bias
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = input_size//world_size

        if not skip_weight_param_allocation:
            self.weight = nn.Parameter(
                            torch.empty(
                                self.output_size,
                                self.input_size_per_partition,
                                device=torch.cuda.current_device()
                                )
                            )
            """
            See my comment for this section in the ColumnParallelLinear
            class.
            """
            rank = get_tensor_model_parallel_rank()
            init_weights = torch.cos(
                            torch.arange(
                                self.input_size_per_partition*self.output_size
                                ).reshape(self.weight.shape)*(0.5**rank)
                            )*0.1
            with torch.no_grad():
                self.weight.copy_(init_weights)
        else:
            self.weight = None

        if add_bias:
            self.bias = nn.Parameter(
                            torch.empty(
                                self.output_size,
                                device=torch.cuda.current_device()
                            )
                        )
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        """
        The 'apply' function doesn't take keyword arguments, only
        position arguments. See this for more information:

        https://github.com/pytorch/pytorch/issues/16940#issuecomment-462183321
        """
        output_parallel = LinearWithAsyncCommunication.apply(
                                x, self.weight, self.bias, False
                            )
        # All-reduce across all the partitions.
        output_ = reduce_from_tensor_model_parallel_region(output_parallel)

        if self.add_bias:
            output = output_ + self.bias
        else:
            output = output_

        return output


class MLPTensorParallel(nn.Module):
    """
    A simple neural net, with a ReLU non-linearity between
    two linear layers.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = ColumnParallelLinear(input_size=input_size,
                        output_size=hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = RowParallelLinear(input_size=hidden_size,
                        output_size=input_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class ToyModelTensorParallel(nn.Module):
    def __init__(self, input_size=8, hidden_size=32):
        super().__init__()
        self.mlp = MLPTensorParallel(input_size=input_size, 
                        hidden_size=hidden_size)
        self.fc = nn.Linear(input_size, 1, bias=False)
        init_weight = torch.sin(torch.arange(
                            input_size
                            ).unsqueeze(0)
                        )*0.1
        with torch.no_grad():
            self.fc.weight.copy_(init_weight)

    def forward(self, x):
        out = self.mlp(x)
        out = self.fc(out)
        return out
