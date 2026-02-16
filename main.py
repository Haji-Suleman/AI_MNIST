import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
)
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
num_epochs = 10
learning_rate = 0.001

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="dataset/", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="dataset/", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

accuracy = MulticlassAccuracy(num_classes=10).to(device)
precision = MulticlassPrecision(num_classes=10, average="macro").to(device)
recall = MulticlassRecall(num_classes=10, average="macro").to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    for data, targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        outputs = model(data)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}")

model.eval()
with torch.no_grad():
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        outputs = model(data)
        preds = torch.argmax(outputs, dim=1)

        accuracy.update(preds, targets)
        precision.update(preds, targets)
        recall.update(preds, targets)

print(f"Test Accuracy: {accuracy.compute():.4f}")
print(f"Precision: {precision.compute():.4f}")
print(f"Recall: {recall.compute():.4f}")