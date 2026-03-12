# from dataset.rf_dataset import get_dataloaders
# from models.cnn_model import RFNet
# from train.train import train_one_epoch, validation, save_checkpoint, load_checkpoint
# from utils.check_dataset import check_dataset, download_dataset
# from config.config import *
# import torch
# import os

# def check_device():
#     is_cuda = torch.cuda.is_available()
#     if is_cuda:
#         print(f"- Sử dụng GPU: {torch.cuda.get_device_name(0)}")
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#         print("- Sử dụng CPU")
#     return device

# def get_version(path):
#     if not os.path.exists(path):
#         os.makedirs(path)
#     versions = [d for d in os.listdir(path) if d.startswith("ver")]
#     ver = len(versions) + 1
#     return f"ver{ver}"


# def main(resume=False, checkpoint_path=None):
#     # prepare
#     if checkpoint_path is None:
#       checkpoint_path = os.path.join(CHECKPOINT_DIR, get_version(CHECKPOINT_DIR))
#       os.mkdir(checkpoint_path)
#       print(f"- Lượt huấn luyện mới lưu tại: {checkpoint_path}")
#     download_dataset(DATASET_PATH, KAGGLE_PATH)
#     check_dataset(DATASET_PATH)
#     device = check_device()
#     if device.type == "cuda":
#         torch.backends.cudnn.benchmark = True


#     # dataset
#     train_loader, val_loader = get_dataloaders(DATASET_PATH, BATCH_SIZE, device, NUM_WORKERS, TRAIN_SPLIT, SEED)
    
#     # model
#     model = RFNet(num_classes=NUM_CLASSES)
#     model.to(device)

#     # loss với label smoothing
#     criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

#     # optimizer
#     optimizer = torch.optim.Adam(
#         model.parameters(), 
#         lr=LEARNING_RATE,
#         weight_decay=WEIGHT_DECAY
#     )

#     # learning rate scheduler
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode="min",
#         factor=0.3,
#         patience=2 # giảm lr nếu val loss không cải thiện sau 2 epochs
#     )

#     best_acc = 0.0

#     start_epoch = 0
#     if resume:
#         start_epoch = load_checkpoint(
#             model,
#             optimizer,
#             f"{checkpoint_path}/last_checkpoint.pth",
#             device
#         )
#         print(f"- Đã load checkpoint từ {checkpoint_path}: epoch {start_epoch+1}")
#     for epoch in range(start_epoch, EPOCHS):
#         print(f"\nEpoch {epoch+1}/{EPOCHS}")
#         train_loss, train_acc = train_one_epoch(
#             model,
#             train_loader,
#             criterion,
#             optimizer,
#             device
#         )
#         val_loss, val_acc = validation(
#             model,
#             val_loader,
#             criterion,
#             device
#         )
#         scheduler.step(val_loss)
#         print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
#         print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
#         print("LR: ", optimizer.param_groups[0]["lr"])

#         save_checkpoint(
#             model,
#             optimizer,
#             epoch + 1,
#             f"{checkpoint_path}/last_checkpoint.pth"
#         )
#         print("- Đã lưu check_point")
#         if (epoch + 1) % SAVE_EVERY == 0:

#             save_checkpoint(
#                 model,
#                 optimizer,
#                 epoch + 1,
#                 f"{checkpoint_path}/checkpoint_epoch_{epoch+1}.pth"
#             )

#         # save best model
#         if val_acc > best_acc:
#             best_acc = val_acc
#             torch.save(model.state_dict(), f"{checkpoint_path}/{MODEL_NAME}")
#             print(f"- Model saved with val acc: {best_acc:.4f}")

# if __name__ == "__main__":
#     main(resume=True, checkpoint_path="checkpoints/ver9")
