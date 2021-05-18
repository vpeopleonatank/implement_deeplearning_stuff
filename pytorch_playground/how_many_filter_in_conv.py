import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 4, out_channels = 4, kernel_size = 3)
        # filter size means: how many kernel

    def forward(self, x):
        x = self.conv1(x)
        return x
net = Net()
print(net)

params = list(net.parameters())
print(params[0])
print(params[0].size())
