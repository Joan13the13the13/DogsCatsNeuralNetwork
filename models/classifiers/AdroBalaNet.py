import torch.nn as nn
from torchvision import models


class AdroBalaNet(nn.Module):
    def __init__(self):
        super(AdroBalaNet, self).__init__()


        self.cnn = nn.Sequential(
                nn.Conv2d(3,32,kernel_size(5,5) , stride=(1,1), padding=(2,2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
                nn.Conv2d(32,64,kernel_size=(3,3),stride=(1,1)),
                nn.MaxPool2d(kernel_size=3,stride=2,padding=0)
        )
    
        # Define the additional layers for your specific task
        self.classifier = nn.Sequential(
            nn.Linear(in_features=227),
            nn.Flatten(1, -1),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=512, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=2, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        return x