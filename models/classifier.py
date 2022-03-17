import torch 
import torch.nn as nn 

class ClassifierV2(torch.nn.Sequential):
    def __init__(self, num_classes=10):
        super(ClassifierV2, self).__init__()
        self.layers = \
            torch.nn.Sequential(
            torch.nn.Linear(48, num_classes)
            )

    def forward(self, x):
        x = self.layers(x)
        prediction = F.softmax(x)
        return x, prediction