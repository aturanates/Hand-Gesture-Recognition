import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)  # Adjust the dropout rate as needed


        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)  # Adjust the dropout rate as needed


        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.25)  # Adjust the dropout rate as needed
        self.fc2 = nn.Linear(128, 3)  # Adjust the output size based on your task

    def forward(self, x):
        # Input shape: (batch_size, 12, height, width)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Reshape before passing to fully connected layers
        x = x.view(-1, 64 * 8 * 8)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc2(x)

        return x


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # Repeat the RGB channels to match the 4-channel requirement
        image = Image.merge('RGB', [image.getchannel(i % 3) for i in range(4)])

        if self.transform:
            image = self.transform(image)

        return image


# Example usage:
# Assume you have a list of image file paths
image_paths = ["path/to/image1.png", "path/to/image2.png", ...]

# Define a transformation to convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create an instance of the CustomDataset
custom_dataset = CustomDataset(image_paths, transform=transform)

# Create a DataLoader to efficiently load and batch the data
batch_size = 32
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Iterate over the DataLoader to get batches of data
for batch in data_loader:
    # Your training/validation loop here
    # 'batch' will contain tensors with shape (batch_size, 4, height, width)
    pass

# Create an instance of the CustomCNN
model = CustomCNN()

# Print the model architecture
print(model)
