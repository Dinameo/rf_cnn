import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.cnn_model import RFNet
from config.config import *
import os
import random
<<<<<<< HEAD
from utils.visualize import plot_confusion_matrix, plot_distribution_of_dataset
=======
from utils.visualize import plot_confusion_matrix, plot_distribution_of_dataset, create_visualization_figure
import matplotlib.pyplot as plt
>>>>>>> 66f2a012b (initial commit)

def predict(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device).unsqueeze(0)   # thêm batch dimension
        # dự đoán xác suất và nhãn
        output = model(image)
        # chuyển đổi output thành xác suất và nhãn dự đoán
        prob = torch.softmax(output, dim=1)
        # lấy nhãn dự đoán có xác suất cao nhất
        pred = torch.argmax(prob, dim=1)
    return pred.item(), prob
dat_path = "data/images/"
<<<<<<< HEAD
checkpoint_path = os.path.join(CHECKPOINT_DIR, "ver9", "last_checkpoint.pth")
=======
checkpoint_path = os.path.join(CHECKPOINT_DIR, "ver3", "last_checkpoint.pth")
>>>>>>> 66f2a012b (initial commit)
n_samples = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
model = RFNet(num_classes=NUM_CLASSES).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
dataset = datasets.ImageFolder(
    root=dat_path,
    transform=transform
)

indices = random.sample(range(len(dataset)), n_samples)
samples = [dataset[i] for i in indices]


correct_total = 0
y_pred = []
y_true = []



for image, label in samples:

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1)
    if pred.item() == label:
        correct_total += 1
    # print(f"True: {CLASS_NAMES[label]:<10} Pred: {CLASS_NAMES[pred.item()]:<10}")
    y_true.append(label)
    y_pred.append(pred.item())
print("Accuracy: {:.2f}%".format(correct_total / n_samples * 100))

<<<<<<< HEAD
plot_confusion_matrix(y_true, y_pred)
=======
fig, ax = create_visualization_figure(row=1, col=1, figsize=(8, 6))

plot_confusion_matrix(y_true, y_pred, ax)
plt.show()
>>>>>>> 66f2a012b (initial commit)
