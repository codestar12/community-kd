import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import Bottleneck, Block


class Hydra(nn.Module):
    def __init__(
        self,
        ResBlock,
        layer_list,
        num_classes,
        student_layers,
        num_channels=3,
        num_heads=1,
    ):
        super(Hydra, self).__init__()

        assert len(student_layers) == num_heads - 1

        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.num_heads = num_heads
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)

        # self.in_channels is changed with side-effects in _make_layer
        copy_in = self.in_channels
        layers = []

        for i in range(num_heads):
            self.in_channels = copy_in
            if i == 0:
                layers.append(
                    self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
                )
            else:
                layers.append(
                    self._make_layer(
                        ResBlock, student_layers[i - 1][0], planes=256, stride=2
                    )
                )

        self.layer3 = nn.ModuleList(layers)

        # self.in_channels is changed with side-effects in _make_layer
        copy_in = self.in_channels
        layers = []

        for i in range(num_heads):
            self.in_channels = copy_in
            if i == 0:
                layers.append(
                    self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
                )
            else:
                layers.append(
                    self._make_layer(
                        ResBlock, student_layers[i - 1][1], planes=512, stride=2
                    )
                )
        self.layer4 = nn.ModuleList(layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.heads = nn.ModuleList(
            [nn.Linear(512 * ResBlock.expansion, num_classes) for i in range(num_heads)]
        )

    def forward_top(self, x: torch.tensor) -> torch.tensor:
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def forward_submodel(self, idx: int, x: torch.tensor) -> torch.tensor:
        x = self.layer3[idx](x)
        x = self.layer4[idx](x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.heads[idx](x)

        return x

    def forward(self, x: torch.tensor) -> torch.tensor:

        x = self.forward_top(x)

        outs = []
        for i in range(self.num_heads):
            if i == 0:
                outs.append(self.forward_submodel(i, x))
            else:
                x_c = x.clone()
                if x_c.requires_grad:
                    h = x_c.register_hook(lambda grad: grad * 0.1)
                outs.append(self.forward_submodel(i, x_c))

        x = torch.stack(outs)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    planes * ResBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(planes * ResBlock.expansion),
            )

        layers.append(
            ResBlock(
                self.in_channels, planes, i_downsample=ii_downsample, stride=stride
            )
        )
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


def Hydra18(num_classes, channels=3, student_layers=[], num_heads: int = 1):
    return Hydra(Block, [2, 2, 2, 2], num_classes, student_layers, channels, num_heads)


def Hydra34(num_classes, channels=3, student_layers=[], num_heads: int = 1):
    return Hydra(Block, [3, 4, 6, 3], num_classes, student_layers, channels, num_heads)


def Hydra50(num_classes, channels=3, student_layers=[], num_heads: int = 1):
    return Hydra(
        Bottleneck, [3, 4, 6, 3], num_classes, student_layers, channels, num_heads
    )


def Hydra41(num_classes, channels=3, student_layers=[], num_heads: int = 1):
    return Hydra(
        Bottleneck, [3, 4, 3, 3], num_classes, student_layers, channels, num_heads
    )


def Hydra35(num_classes, channels=3, student_layers=[], num_heads: int = 1):
    return Hydra(
        Bottleneck, [3, 4, 2, 2], num_classes, student_layers, channels, num_heads
    )


def Hydra29(num_classes, channels=3, student_layers=[], num_heads: int = 1):
    return Hydra(
        Bottleneck, [3, 4, 1, 1], num_classes, student_layers, channels, num_heads
    )


def Hydra101(num_classes, channels=3, student_layers=[], num_heads: int = 1):
    return Hydra(
        Bottleneck, [3, 4, 23, 3], num_classes, student_layers, channels, num_heads
    )


def Hydra152(num_classes, channels=3, student_layers=[], num_heads: int = 1):
    return Hydra(
        Bottleneck, [3, 8, 36, 3], num_classes, student_layers, channels, num_heads
    )
