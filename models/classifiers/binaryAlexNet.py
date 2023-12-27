import torch.nn as nn
from torchvision import models


class BinaryAlexNet(nn.Module):
    def __init__(self):
        super(BinaryAlexNet, self).__init__()
        alex = models.alexnet(weights="AlexNet_Weights.IMAGENET1K_V1")

        # Load the pre-trained model (assuming 'alex' is defined elsewhere)
        self.features = alex.features

        # Freeze the parameters of the pre-trained model
        for param in self.features.parameters():
            param.requires_grad = False

        # Define the additional layers for your specific task
        self.classifier = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=9216, out_features=4608, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4608, out_features=2048, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=2, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x