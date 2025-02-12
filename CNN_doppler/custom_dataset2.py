from torchvision import transforms
#import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from torchsummary import summary



class SpectrogramDataset(Dataset):
    def __init__(self, image_directory, transform=None):
        self.image_directory = image_directory
        self.transform = transform
        self.image_files = os.listdir(image_directory)

        self.label_mapping = {chr(ord('A') + i): i for i in range(3)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image_path = os.path.join(self.image_directory, filename)
        img = Image.open(image_path).convert("RGB")  # Convert to RGB if your images are grayscale
        label = self.label_mapping[filename[0].upper()]

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(label,dtype=torch.long)

        return img, label


# Set your desired transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Create datasets and dataloaders
train_dataset = SpectrogramDataset(image_directory="D:/Desktop/handwrite_recognition/ml_project/figures2", transform=transform)
test_dataset = SpectrogramDataset(image_directory="D:/Desktop/handwrite_recognition/ml_project/figures2_test", transform=transform)

torch.save(test_dataset,"test_dataset.pth")
torch.save(train_dataset,"train_dataset.pth")

train_loader = torch.load('train_dataset.pth')
test_loader = torch.load('test_dataset.pth')

#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes):
        super(SpectrogramCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(64 * (128 // 3) * (128 // 3), 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * (128 // 4) * (128 // 4))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SpectrogramCNN(3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model on the test set
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')
