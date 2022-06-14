"""
Custom UNet Model (End-to-end fully convolutional network (FCN))
UNet architecture contains two paths
- Encoder (Feature down sampling): used to capture the context in the image
- Decoder (Feature up sampling): used for precise localization
Input size (batch x channels x height x width)
Output size (batch x num_classes x height x width)
"""

import torch
import torchvision


class ConvBlock(torch.nn.Module):
    """
    Perform 2 convolutions operations. Each followed by a batch normalization
    and ReLU activation
    """
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_ch)
        self.act = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class UNet(torch.nn.Module):
    """
    Encoder -> Bottleneck -> Decoder -> Output
    """
    def __init__(self, in_ch=3, out_ch=1, features=(64, 128, 256, 512)):
        super(UNet, self).__init__()

        # Down sampling blocks
        self.downs = torch.nn.ModuleList()
        # Up sampling blocks
        self.ups = torch.nn.ModuleList()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Feature down sampling
        for feature in features:
            self.downs.append(ConvBlock(in_ch, feature))
            in_ch = feature

        # Feature up sampling
        for feature in reversed(features):
            self.ups.append(
                torch.nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(ConvBlock(feature*2, feature))

        self.bottleneck = ConvBlock(features[-1], features[-1]*2)
        self.final_conv = torch.nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        shortcuts = []

        for down in self.downs:
            x = down(x)
            shortcuts.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        shortcuts = shortcuts[::-1]

        # Create shortcut paths or skip connections between encoder and decoder blocks
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = shortcuts[idx//2]

            if x.shape != skip_connection.shape:
                x = torchvision.transforms.functional.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        x = self.final_conv(x)

        return torch.sigmoid(x)
