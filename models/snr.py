import torch.nn as nn

class SNRBlock(nn.Module):
    def __init__(self, channels):
        super(SNRBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + res)