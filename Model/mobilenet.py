from math import floor
import torch.nn as nn

__all__ = ['mobilenet', 'mobilenet_025', 'mobilenet_050', 'mobilenet_075']

class MobileNet(nn.Module):
    def __init__(self, channel_multiplier=1.0, min_channels=8):
        super(MobileNet, self).__init__()

        if channel_multiplier <= 0:
            raise ValueError('channel_multiplier must be >= 0')

        def conv_bn_relu(n_ifm, n_ofm, kernel_size, stride=1, padding=0, groups=1):
            return [
                nn.Conv2d(n_ifm, n_ofm, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
                nn.BatchNorm2d(n_ofm),
                nn.ReLU(inplace=True)
            ]

        def depthwise_conv(n_ifm, n_ofm, stride):
            return nn.Sequential(
                *conv_bn_relu(n_ifm, n_ifm, 3, stride=stride, padding=1, groups=n_ifm),
                *conv_bn_relu(n_ifm, n_ofm, 1, stride=1)
            )

        base_channels = [32, 64, 128, 256, 512, 1024]
        self.channels = [max(floor(n * channel_multiplier), min_channels) for n in base_channels]

        self.model = nn.Sequential(
            nn.Sequential(*conv_bn_relu(3, self.channels[0], 3, stride=2, padding=1)),
            depthwise_conv(self.channels[0], self.channels[1], 1),
            depthwise_conv(self.channels[1], self.channels[2], 2),
            depthwise_conv(self.channels[2], self.channels[2], 1),
            depthwise_conv(self.channels[2], self.channels[3], 2),
            depthwise_conv(self.channels[3], self.channels[3], 1),
            depthwise_conv(self.channels[3], self.channels[4], 2),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[5], 2),
            depthwise_conv(self.channels[5], self.channels[5], 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(self.channels[5], 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, x.size(1))
        x = self.fc(x)
        return x


def mobilenet_025():
    return MobileNet(channel_multiplier=0.25)


def mobilenet_050():
    return MobileNet(channel_multiplier=0.5)


def mobilenet_075():
    return MobileNet(channel_multiplier=0.75)


def mobilenet(pretrained=False):
    return MobileNet()
