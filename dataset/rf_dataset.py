import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


def get_dataloaders(path, batch_size, image_size, num_workers, train_split=0.8, seed=42):
    # cố định seed
    generator = torch.Generator().manual_seed(seed)

    transform = transforms.Compose([
        # Đảm bảo ảnh luôn 1 kênh màu
        transforms.Grayscale(num_output_channels=1),
        # Chuyền PIL Image sang Tensor scale về [0, 1]
        transforms.ToTensor(),
        # Biến [0, 1] thành [-1, 1]
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = ImageFolder(
        root=path,
        transform=transform
    )

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, val_loader