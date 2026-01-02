from torch.utils.data import DataLoader
from emotion_faces_dataset import YoloEmotionDataset

train_dataset = YoloEmotionDataset(
    images_dir="/mnt/homeGPU/pablomarpa/data/archive/YOLO_format/train/images",
    labels_dir="/mnt/homeGPU/pablomarpa/data/archive/YOLO_format/train/labels",
    img_size=64
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for imgs, labels in train_loader:
    print(imgs.shape, labels.shape)
    break

