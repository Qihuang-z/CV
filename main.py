import os
import torch
import torch.nn as nn
import torch.optim as optim
from data import train_loader, test_loader
from model import ResNet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(train_loader.dataset.classes)
model = ResNet18().to(device)

model.linear = nn.Linear(512 * model.layer4[0].expansion, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

weight_dir = "./weight"
os.makedirs(weight_dir, exist_ok=True)

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, save_path="./weight/resnet18.pth"):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total * 100
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to: {save_path}")


train_model(model, train_loader, criterion, optimizer, num_epochs=10, save_path=os.path.join(weight_dir, "resnet18.pth"))

def validate_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = correct / total * 100
    print(f"Validation Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

validate_model(model, test_loader, criterion)
