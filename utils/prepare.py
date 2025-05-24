from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder

def prepare_dataset():
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = ImageFolder(root="data", transform=transform)
    print("Dataset prepared: ", dataset)
    print("---------------------------------------------------------------\n")
    return dataset

def split_dataset(dataset):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("Train and validation datasets prepared: ", train_loader, val_loader)
    print("---------------------------------------------------------------\n")
    return train_loader, val_loader