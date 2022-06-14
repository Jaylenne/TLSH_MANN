import torch
import torch.nn as nn

class CNNController(nn.Module):

    def __init__(self, d):
        super(CNNController, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1, 128, 5),nn.ReLU(),nn.Conv2d(128, 128, 5),nn.ReLU())
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.block2 = nn.Sequential(nn.Conv2d(128, 128, 3),nn.ReLU(),nn.Conv2d(128, 128, 3),nn.ReLU())
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.fc = nn.Linear(2048, d)

    def forward(self, inputs):
        x = self.block1(inputs)
        x = self.maxpool1(x)
        x = self.block2(x)
        x = self.maxpool2(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)
    