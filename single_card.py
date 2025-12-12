import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import json


def load_data(batch_size, path='./'):
    train_path = os.path.join(path, 'train')
    val_path = os.path.join(path, 'val')

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalizer
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalizer
    ])

    train_data = datasets.ImageFolder(root=train_path, transform=train_transform)
    valid_data = datasets.ImageFolder(root=val_path, transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16
    )
    val_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16
    )

    print(f"Train dataset size: {len(train_data)}")
    print(f"Validation dataset size: {len(valid_data)}")

    return train_loader, val_loader

# Training and validation loop
def run_epoch(model, dataloader, criterion, optimizer=None, device='cpu', phase='train'):
    if phase == 'train':
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_loss = []
   
    loop = tqdm(dataloader, desc=f"{phase.capitalize()} Phase", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        if phase == 'train':
            optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        all_loss.append(loss.item())
        loop.set_postfix(loss=loss.item(), accuracy=100. * correct / total)

    epoch_loss = running_loss / total
    epoch_accuracy = 100. * correct / total
    return epoch_loss, epoch_accuracy, all_loss


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 256
    train_loader, val_loader = load_data(batch_size, path='./')

    num_classes = 1000
    model = models.resnet50(weights=True)
    
    local_weights_path = "./resnet50-0676ba61.pth"
    state_dict = torch.load(local_weights_path)
    model.load_state_dict(state_dict)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 3
    best_accuracy = 0.0

    save_path = './saved_models'
    os.makedirs(save_path, exist_ok=True)

    all_train_loss = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('-' * 20)

        train_loss, train_accuracy, train_epoch_loss = run_epoch(model, train_loader, criterion, optimizer, device, phase='train')
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        val_loss, val_accuracy, _ = run_epoch(model, val_loader, criterion, device=device, phase='validate')
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        all_train_loss.extend(train_epoch_loss)
        
        torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch + 1}.pth'))
        
        # Save the best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))

    # Testing
    test_loss, test_accuracy, _ = run_epoch(model, val_loader, criterion, device=device, phase='test')
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    
    with open('./single_card_loss.json', 'w') as f:
        json.dump(all_train_loss, f)