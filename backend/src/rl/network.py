import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def maybe_compile(model: torch.nn.Module) -> torch.nn.Module:
    if os.environ.get("USE_TORCH_COMPILE", "0") == "1" and torch.cuda.is_available():
        try:
            return torch.compile(model, mode="reduce-overhead", dynamic=True)
        except Exception:
            pass
    return model


class DeepCubeNet(nn.Module):
    def __init__(
        self,
        state_dim: int = 54,
        one_hot_depth: int = 6,
        h1_dim: int = 5000,
        resnet_dim: int = 1000,
        num_blocks: int = 4,
        out_dim: int = 1,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.one_hot_depth = one_hot_depth
        self.state_dim = state_dim
        self.num_blocks = num_blocks
        self.batch_norm = batch_norm
        self.blocks = nn.ModuleList()

        self.fc1 = nn.Linear(self.state_dim * self.one_hot_depth, h1_dim)
        self.fc2 = nn.Linear(h1_dim, resnet_dim)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(h1_dim)
            self.bn2 = nn.BatchNorm1d(resnet_dim)

        for _ in range(self.num_blocks):
            if self.batch_norm:
                block = nn.ModuleList([
                    nn.Linear(resnet_dim, resnet_dim),
                    nn.BatchNorm1d(resnet_dim),
                    nn.Linear(resnet_dim, resnet_dim),
                    nn.BatchNorm1d(resnet_dim),
                ])
            else:
                block = nn.ModuleList([
                    nn.Linear(resnet_dim, resnet_dim),
                    nn.Linear(resnet_dim, resnet_dim),
                ])
            self.blocks.append(block)

        self.fc_out = nn.Linear(resnet_dim, out_dim)

    def forward(self, states_nnet: torch.Tensor) -> torch.Tensor:
        x = F.one_hot(states_nnet.long(), self.one_hot_depth).float()
        x = x.view(-1, self.state_dim * self.one_hot_depth)

        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = F.relu(x)

        for block in self.blocks:
            residual = x
            if self.batch_norm:
                x = block[0](x)
                x = block[1](x)
                x = F.relu(x)
                x = block[2](x)
                x = block[3](x)
            else:
                x = block[0](x)
                x = F.relu(x)
                x = block[1](x)
            x = F.relu(x + residual)

        return self.fc_out(x)
