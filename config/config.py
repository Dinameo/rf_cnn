# đường dẫn đến dataset
DATASET_PATH = "data/images"

# Kaggle dataset path
KAGGLE_PATH = "nhanle2001/rf-signal-modulation-classification-dataset"

# kích thước batch
BATCH_SIZE = 32

# Số lượng lớp đầu ra
NUM_CLASSES = 12

# tên các lớp
CLASS_NAMES = {
    0: "Barker",
    1: "BPSK",
    2: "16-QAM",
    3: "StepFM",
    4: "B-FM",
    5: "PAM4",
    6: "LFM",
    7: "QPSK",
    8: "CPFSK",
    9: "DSB-AM",
    10: "GFSK",
    11: "Rect",
}

# Kích thước ảnh đầu vào
IMAGE_SIZE = 224

# số lượng epoch
EPOCHS = 20

# learning rate
LEARNING_RATE = 0.001


# đường dẫn lưu checkpoints
CHECKPOINT_DIR = "checkpoints"

# lưu check point sau mỗi 5 epochs
SAVE_EVERY = 5

# tên file model
MODEL_NAME = "rf_cnn_model.pth"

# random seed
SEED = 42

# tỉ lệ chia tập train/val
TRAIN_SPLIT = 0.8


# số worker cho DataLoader
NUM_WORKERS = 4