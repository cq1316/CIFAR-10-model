import torch.nn as nn
import math


class CifarModel(nn.Module):
    def __init__(self):
        super(CifarModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=3,
                         stride=2),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=3,
                         stride=2),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=64,
                      out_features=2,
                      bias=True),
            nn.Softmax()
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.1)

    def forward(self, x):
        fc_in = self.features(x).view(x.size(0), -1)
        fc_out = self.fc(fc_in)
        return fc_out
