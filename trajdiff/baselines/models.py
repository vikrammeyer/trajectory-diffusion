import logging

import torch.nn as nn


class FCNet(nn.Module):
    def __init__(self, layer_sizes) -> None:
        super(FCNet, self).__init__()

        net = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            if i == len(layer_sizes) - 2:
                layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                nn.init.kaiming_normal_(layer.weight)
                net.append(layer)
            else:
                layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                nn.init.kaiming_normal_(layer.weight)
                net.append(layer)
                net.append(nn.ReLU())

        self.net = nn.Sequential(*net)
        logging.info("created model: %s", self.net)

    def forward(self, x):
        return self.net(x)
