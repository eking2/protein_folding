import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_chan = 64, out_chan = 64, dilation = 1):

    padding = 1 + (dilation - 1)
    return nn.Conv2d(in_chan, out_chan, kernel_size = 3, padding = padding, bias = False, dilation = dilation)


class ResBlock(nn.Module):

    def __init__(self, dilation):

        super().__init__()

        self.proj_down = nn.Conv2d(in_channels = 128,
                                   out_channels = 64,
                                   kernel_size = 1,
                                   bias = False)

        self.proj_up = nn.Conv2d(in_channels = 64,
                                 out_channels = 128,
                                 kernel_size = 1)

        self.dilation = conv3x3(64, 64, dilation = dilation)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):

        # input (128, N, N) 
        identity = x

        out = self.bn1(x)
        out = F.elu_(out)
        out = self.proj_down(out)  # (64, N, N)
        out = self.bn2(out)
        out = F.elu_(out)
        out = self.dilation(out)
        out = self.bn3(out)
        out = F.elu_(out)
        out = self.proj_up(out)    # (128, N, N)

        return out + identity


class ResNet(nn.Module):

    def __init__(self, input_shape, n_dist_bins, n_blocks, dilations = [1, 2, 4, 8]):

        super().__init__()

        in_chan = input_shape[1]
        seq_len = input_shape[2]

        layers = []

        # batchnorm input
        layers.append(nn.BatchNorm2d(in_chan))

        # to 128 channels
        layers.append(nn.Conv2d(in_chan, 128, kernel_size = 1))

        # resnet blocks with dilation cycling
        for i in range(n_blocks):
            for j in dilations:
                layers.append(ResBlock(j))

        # output channel number of bins
        layers.append(nn.Conv2d(128, n_dist_bins, kernel_size = 1)) 

        self.net = nn.Sequential(*layers)

    def forward(self, x):

        return self.net(x)


def test_resblock():

    x = torch.randn(1, 128, 64, 64)
    for i in [1, 2, 4, 8]:
        block = ResBlock(dilation = i)
        out = block(x)
        assert out.shape == x.shape, f'invalid resblock dilation: {i}'
        print(i)
        print(out.shape)
        print()

def test_resnet():

    x = torch.randn(1, 50, 64, 64)
    net = ResNet(x.shape, n_dist_bins = 10, n_blocks = 2)
    out = net(x)
    print(out.shape)

#test_resblock()
test_resnet()
