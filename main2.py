import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from data2 import get_data_loaders
from model import get_model
from tqdm import tqdm

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs, device, save_path):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total * 100
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to: {save_path}")

def evaluate_model(model, test_loader, criterion, device):
    model.to(device)
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    loss = running_loss / len(test_loader.dataset)
    accuracy = correct / total * 100
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return loss, accuracy

if __name__ == '__main__':
    root_dir = "./trashnet"
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    weight_dir = "./weights"
    os.makedirs(weight_dir, exist_ok=True)
    save_path = os.path.join(weight_dir, "efficientnet_b7.pth")

    train_loader, test_loader = get_data_loaders(root_dir, batch_size=batch_size)
    num_classes = len(train_loader.dataset.classes)
    model = get_model(num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs, device, save_path)
    evaluate_model(model, test_loader, criterion, device)