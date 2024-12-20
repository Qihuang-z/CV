import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

class TrashNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.data = []

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    file_path = os.path.join(class_dir, file_name)
                    if os.path.isfile(file_path):
                        self.data.append((file_path, class_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

root_dir = "./trashnet"
transform = transforms.Compose([
    # transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = TrashNetDataset(root_dir, transform=transform)


##------------shuffle & split

data_list = dataset.data
label_list = [label for _, label in data_list]

train_data, test_data = train_test_split(
    data_list, test_size=0.2, random_state=42, stratify=label_list
)

train_dataset = TrashNetDataset(root_dir, transform=transform)
test_dataset = TrashNetDataset(root_dir, transform=transform)
train_dataset.data = train_data
test_dataset.data = test_data

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_class_counts = {class_name: 0 for class_name in dataset.classes}
for _, label in train_dataset:
    class_name = dataset.classes[label]
    train_class_counts[class_name] += 1

test_class_counts = {class_name: 0 for class_name in dataset.classes}
for _, label in test_dataset:
    class_name = dataset.classes[label]
    test_class_counts[class_name] += 1

print("Train Class Counts:")
for class_name, count in train_class_counts.items():
    proportion = count / len(train_dataset)
    print(f"{class_name}: {count} ({proportion:.2%})")

print("\nTest Class Counts:")
for class_name, count in test_class_counts.items():
    proportion = count / len(test_dataset)
    print(f"{class_name}: {count} ({proportion:.2%})")