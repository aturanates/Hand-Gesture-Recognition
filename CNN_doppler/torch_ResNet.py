import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision

# Define ResNet model
class ResNet(nn.Module):
    def __init__(self, num_classes=10, num_channels=4):
        super(ResNet, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        # Modify the first layer to accept 4 channels
        self.resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the classifier to match the number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Create a dataset with 4 channels
class CustomDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        img = self.transform(img)
        return img, target

# Example usage
# Set your data directory and other parameters
data_dir = '/path/to/your/dataset'
num_classes = 10
num_channels = 4
batch_size = 64

# Transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create the dataset
dataset = CustomDataset(root=data_dir, transform=transform)

# Create a data loader
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Initialize the model
model = ResNet(num_classes=num_classes, num_channels=num_channels)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Save the model if needed
torch.save(model.state_dict(), 'resnet_model.pth')
