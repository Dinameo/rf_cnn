import os
from PIL import Image
from config.config import *
import kagglehub
import shutil
import os


def download_dataset(dat_path, kaggel_path):
    if not os.path.exists(dat_path):
        print("- Dataset chưa tồn tại, bắt đầu tải về...")
        path = kagglehub.dataset_download(kaggel_path)
        shutil.move(path, dat_path)
        print(f"- Dataset lưu tại {dat_path}")
    else:
        print("- Dataset đã tồn tại, không cần tải lại.")

def check_dataset(dat_path):
    classes = os.listdir(dat_path)
    total_images = 0
    print(f"- Số lượng lớp đầu ra: {len(classes)}")
    for i, cls in enumerate(classes):
        image_path = os.path.join(dat_path, cls)
        images_name = os.listdir(image_path)
        total_images += len(images_name)
        print(f"\t{i}: \"{cls}\"\t\t-> {len(images_name)} ảnh")

    print(f"- Tổng số ảnh trong dataset: {total_images}")

    img_test_name = os.listdir(os.path.join(dat_path, classes[0]))[0]
    img_test_path = os.path.join(dat_path, classes[0], img_test_name)
    img_test = Image.open(img_test_path)

    print(f"- Kích thước ảnh mẫu: ", img_test.size)