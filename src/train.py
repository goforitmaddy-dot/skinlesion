import os

import matplotlib.pyplot as plt
import sklearn.metrics
import torch
import torchvision
from torch import nn
from torchvision import transforms , datasets
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os

from torch.optim import adam



train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(0),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomVerticalFlip(p = 0.5),
    transforms.ToTensor(),

])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),])




train_dir = "../data/skin_dataset/train"
test_dir = "../data/skin_dataset/test"


train_data = datasets.ImageFolder(root=train_dir,
                                  transform=train_transform)
test_data  = datasets.ImageFolder(root=test_dir,
                                  transform=test_transform)

train_loader = DataLoader(train_data,
                          batch_size=32,
                          shuffle=True,
                          num_workers=0)
test_loader  = DataLoader(test_data,
                          batch_size=32,
                          shuffle=False,
                          num_workers=0)





def walk_through_dir(dir_path):
    for dirpath , dirnames , filenames in os.walk(dir_path):
        print(f"there are {len(dirnames)} and directories and {len(filenames)}images in '{dir_path}'")


#walk_through_dir( "../data/skin_dataset/train/Healthy"

#print(len(data))
#print(data.class_to_idx)


#print(data.samples[0:5])


class SkinLesionV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(hidden_units),           # helps training stabilize
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 → 112  (was 3,3 which is unusual)
        )

        # After MaxPool2d(2,2): spatial size = 224/2 = 112
        # Flattened size = hidden_units * 112 * 112
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 112 * 112, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, output_shape),
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)
        return x


model = SkinLesionV0(3, 32, 14)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch = 12


def train_epoch(model , loader , optimizer , loss_fn):
    model.train()
    for images , labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        preds = model(images)
        loss = loss_fn(preds , labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (preds.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total























