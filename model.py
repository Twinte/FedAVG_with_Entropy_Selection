import torch
import torch.nn as nn

# Define your neural network architecture
class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()

        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, 32, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        # First fully connected layer
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        
        # Output layer (fully connected)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)      # Apply first convolutional layer
        out = self.conv2(out)    # Apply second convolutional layer
        out = torch.flatten(out, 1)  # Flatten the output for fully connected layers
        out = self.fc1(out)      # Apply first fully connected layer
        out = self.fc(out)       # Apply output fully connected layer
        return out