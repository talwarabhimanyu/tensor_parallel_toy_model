import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, input_size, bias=False)
        self._init_weights()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def _init_weights(self):
        """
        Initialize weights to the same values that we use for the
        tensor parallel version. Note that I assume a fixed 2 devices
        were used in the tensor parallel training run.
        """
        hidden_size, input_size = self.fc1.weight.shape
        assert hidden_size % 2 == 0, "hidden size needs to be divisible by 2"
        # fc1
        size_per_partition = hidden_size // 2
        col_parallel1 = torch.sin(
                            torch.arange(
                                size_per_partition*input_size
                                ).reshape(size_per_partition, input_size)*(0.5**0.)
                            )*0.1
        col_parallel2 = torch.sin(
                            torch.arange(
                                size_per_partition*input_size
                                ).reshape(size_per_partition, input_size)*(0.5**1.)
                            )*0.1
        weights_fc1 = torch.cat((col_parallel1.T, col_parallel2.T), dim=1).T
        
        # fc2
        row_parallel1 = torch.cos(
                            torch.arange(
                                size_per_partition*input_size
                                ).reshape(input_size, size_per_partition)*(0.5**0.)
                            )*0.1
        row_parallel2 = torch.cos(
                            torch.arange(
                                size_per_partition*input_size
                                ).reshape(input_size, size_per_partition)*(0.5**1.)
                            )*0.1
        weights_fc2 = torch.cat((row_parallel1.T, row_parallel2.T), dim=0).T


        with torch.no_grad():
            self.fc1.weight.copy_(weights_fc1)
            self.fc2.weight.copy_(weights_fc2)

class ToyModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=32):
        super().__init__()
        self.input_size = input_size
        self.mlp = MLP(input_size=input_size, 
                        hidden_size=hidden_size)
        self.fc = nn.Linear(input_size, 1, bias=False)
        """
        Init weight to match the parallel model weights.
        """
        self._init_weights()

    def forward(self, x):
        out = self.mlp(x)
        out = self.fc(out)
        return out

    def _init_weights(self):
        init_weight = torch.sin(torch.arange(
                            self.input_size,
                            ).unsqueeze(0)*1.0
                        )*0.1
        with torch.no_grad():
            self.fc.weight.copy_(init_weight)
