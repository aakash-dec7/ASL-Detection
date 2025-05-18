import torch.nn as nn
from types import SimpleNamespace


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 28x28
        )

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 29)  # 29 classes (A-Z + extra symbols)

    def forward(self, x, labels=None):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        x = self.dropout(self.fc1(x))
        x = self.fc2(x)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(
                label_smoothing=0.1,
            )

            loss = loss_fn(x, labels)
            return SimpleNamespace(loss=loss, logits=x)

        return x
