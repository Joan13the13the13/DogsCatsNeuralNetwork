import torch
import torch.nn as nn
import torchvision.models as models

class BinaryAdroBalaNet(nn.Module):
    def __init__(self, num_classes=2):
        super(BinaryAdroBalaNet, self).__init__()
        alexnet = models.alexnet(pretrained=False)

        self.features = nn.Sequential(
            *list(alexnet.features.children())[:-1]
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x