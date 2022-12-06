import torch.nn as nn

class FCNet(nn.Module):
    def __init__(self, layer_sizes) -> None:
        super(FCNet, self).__init__()

        net = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            if i == len(layer_sizes) - 2:
                net.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            else:
                net.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                net.append(nn.ReLU())

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)