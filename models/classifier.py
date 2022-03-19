import torch 
import torch.nn as nn 
from torch.nn import functional as F

class Classifier(torch.nn.Sequential):
    def __init__(self, num_classes=10):
        super(Classifier, self).__init__()
        self.layers = \
            torch.nn.Sequential(
            torch.nn.Linear(48, num_classes)
            )

    def forward(self, x):
        x = self.layers(x)
        prediction = F.softmax(x)
        return x, prediction