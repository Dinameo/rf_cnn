import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config.config import *

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):

        # chuyển dữ liệu sang device (CPU hoặc GPU)
        images = images.to(device)
        labels = labels.to(device)

        # reset gradients
        optimizer.zero_grad(set_to_none=True)

        # forward pass
        outputs = model(images)

        # tính loss
        loss = criterion(outputs, labels)
        # backpropagation
        loss.backward()
        # cập nhật weights
        optimizer.step()
        # cộng dồn loss
        running_loss += loss.item()
        # tính số lượng dự đoán đúng
        predicted = outputs.argmax(dim=1)
        # cập nhật tổng số mẫu
        total += labels.size(0)
        # cập nhật số mẫu dự đoán đúng
        correct += predicted.eq(labels).sum().item()
    acc = correct / total
    return running_loss / len(loader), acc
def validation(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader):
            # chuyển dữ liệu sang device (CPU hoặc GPU)
            images = images.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(images)
            # tính loss
            loss = criterion(outputs, labels)

            # cộng dồn loss
            running_loss += loss.item()
            # tính số lượng dự đoán đúng
            predicted = outputs.argmax(dim=1)
            # cập nhật tổng số mẫu
            total += labels.size(0)
            # cập nhật số mẫu dự đoán đúng
            correct += predicted.eq(labels).sum().item()
    acc = correct / total
    return running_loss / len(loader), acc
def save_checkpoint(epoch, model, optimizer, train_losses, val_losses, train_accs, val_accs, path):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs
    }

    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path, device):

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint["epoch"]

    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]
    train_accs = checkpoint["train_accs"]
    val_accs = checkpoint["val_accs"]

    return start_epoch, train_losses, val_losses, train_accs, val_accs