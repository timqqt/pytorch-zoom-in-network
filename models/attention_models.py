import math
import torch
from torch import nn
from torch.nn import functional as F

padding = 0



        
class Attention(torch.nn.Sequential):
    def __init__(self, no_softmax=False):
        super(Attention, self).__init__()
        # self.padding = 2
        self.no_softmax = no_softmax
        self.layers = \
            torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, kernel_size=3, padding=(1, 1)),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(8, 8, kernel_size=3, padding=(1, 1)),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(8, 1, kernel_size=3, padding=(1, 1)),
            )
        self.layers2 = \
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 8, kernel_size=3, padding=(1, 1)),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(8, 8, kernel_size=3, padding=(1, 1)),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(8, 1, kernel_size=3, padding=(1, 1)),
            )

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.layers(x)
        elif x.shape[1] == 1:
            x = self.layers2(x)
        xs = x.shape
        if self.no_softmax:
            return x.view(xs)
        else:
            x = F.softmax(x.view(x.shape[0], -1), dim=1)
            return x.view(xs)




class AttentionOnAttention(torch.nn.Sequential):
    def __init__(self):
        super(AttentionOnAttention, self).__init__()
        self.layers1 = \
            torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, kernel_size=3, padding=(1, 1)),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 8, kernel_size=3, padding=(1, 1)),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 1, kernel_size=3, padding=(1, 1)),
            )


    def forward(self, x):

        x = self.layers1(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.squeeze(1).squeeze(1).squeeze(1)
        return x





