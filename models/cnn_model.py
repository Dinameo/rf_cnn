import torch.nn as nn
from config.config import *

class RFNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        """
        # lớp convolution tách đặc trưng
        # self.features = nn.Sequential(
        #     nn.Conv2d(
        #         1,                  # số kênh đầu vào
        #         16,                 # số kênh đầu ra
        #         kernel_size=3,      # kích thước kernel
        #         padding=1
        #     ),
        #     # output shape: (16, 224, 224)
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     # output shape: (16, 112, 112)

        #     nn.Conv2d(
        #         16, 
        #         32, 
        #         kernel_size=3, 
        #         padding=1
        #     ),
        #     # output shape: (32, 112, 112)
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     # output shape: (32, 56, 56)

        #     nn.Conv2d(
        #         32, 
        #         64, 
        #         kernel_size=3, 
        #         padding=1
        #     ),
        #     # output shape: (64, 56, 56)
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     # output shape: (64, 28, 28)

        #     nn.Conv2d(
        #         64, 
        #         128, 
        #         kernel_size=3, 
        #         padding=1
        #     ),
        #     # output shape: (128, 28, 28)
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        #     # output shape: (128, 14, 14)
        # )
    
        # # lớp fully connected phân loại
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     # output shape: (128*14*14)
        #     nn.Linear(128*14*14, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(256, num_classes)
        # )
        """


        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),     # 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),     # 56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),     # 28x28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1))  # output: (256,1,1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x