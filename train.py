import torch
import torch.nn as nn
from model import CatDogCNN
from dataset import get_loaders
from torchvision import models


def train(data_dir, epochs=10):
    #setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CatDogCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    train_loader, val_loader = get_loaders(data_dir)

    for epoch in range(epochs):
        #training
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()   #clear old grads
            output = model(images)  #forward pass
            loss = criterion(output, labels)
            loss.backward()         #backprop
            optimizer.step()        #weight updates

            train_loss += loss.item()

        #validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)    # index of highest logit
                correct += torch.sum(preds == labels).sum().item()
                total += labels.size(0)

        print(f"Epoch {epoch+1}/{epochs+1}.. ")
        print(f"Train loss: {train_loss/len(train_loader):.4f}.. ")
        print(f"Train acc: {correct/len(train_loader):.4f}.. ")
    torch.save(model.state_dict(), "./catdog_1_model.pth")
    print("model saved")

def train_optimized(data_dir, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights="IMAGENET1k_V1")
    model.fc = nn.Linear(model.fc.in_features, 2)

    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p : p.requires_grad, model.parameters()), lr=0.001
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode= "min", patience=3, factor=0.1
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    train_loader, val_loader = get_loaders(data_dir)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type = device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        model.eval()
        correct, total, val_loss = 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                with torch.autocast(device_type = device.type == "cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs+1}.. ")
        print(f"Train loss: {train_loss/len(train_loader):.4f}.. ")
        print(f"Train acc: {correct/len(train_loader):.4f}.. ")
        print(f"val loss: {val_loss/len(val_loader):.4f}.. ")
        print(f"LR: {optimizer.param_groups[0]['lr']:.4f}.. ")

    torch.save(model.state_dict(), "./catdog_1_model.pth")
    print("model saved")



if __name__ == "main":
    train(data_dir="./Users/rajatsharma/Downloads/kaggle/datasets/abhinavnayak/catsvdogs")