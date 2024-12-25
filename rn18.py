import os
import torch
import torch.nn as nn
import torch.optim as optim
from data import train_loader, test_loader
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(train_loader.dataset.classes)
# print(len(train_loader.dataset.classes))
model = resnet18(weights=ResNet18_Weights.DEFAULT)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

weight_dir = "./resnet18_weight"
txt_path = os.path.join(weight_dir, "loss.txt")
os.makedirs(weight_dir, exist_ok=True)

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, save_path="./resnet18_weight/resnet18.pth"):
    accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0


        for batch_idx, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
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
        with open(txt_path, "a") as f:
            f.write(f"{epoch + 1}, {epoch_loss:.4f}\n")
        epoch_acc = correct / total * 100
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        if epoch_acc > accuracy:
            torch.save(model.state_dict(), save_path)
            accuracy=epoch_acc

            print(f"New best accuracy: {epoch_acc:.2f}% -> model saved to: {save_path}")
        print(f"Training finished. Best accuracy = {accuracy:.2f}%")
train_model(model, train_loader, criterion, optimizer, num_epochs=20,
            save_path=os.path.join(weight_dir, "resnet18.pth"))


print(" validation...")

def validate_model(model, test_loader, weight_path):

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = correct / total * 100
    print(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy

weight_path = './resnet18_weight/resnet18.pth'
validate_model(model, test_loader, weight_path)
