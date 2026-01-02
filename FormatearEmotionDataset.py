import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class YoloEmotionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=64, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_files = sorted([
            f for f in os.listdir(images_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)  # [-1, 1]
        ])
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_filename = self.img_files[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        label_path = os.path.join(
            self.labels_dir, os.path.splitext(img_filename)[0] + '.txt'
        )
        
        # Leer imagen
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        # Leer etiquetas YOLO (puede haber varias, usamos la primera cara)
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            # Imagen sin anotaciones: devolver tensor vacío
            return torch.zeros(3, self.img_size, self.img_size), 0
        
        # Tomamos la primera anotación (asumiendo una cara)
        cls, x, y, bw, bh = map(float, lines[0].split())
        x, y, bw, bh = x * w, y * h, bw * w, bh * h
        
        # Calcular bounding box en píxeles
        x1 = int(x - bw / 2)
        y1 = int(y - bh / 2)
        x2 = int(x + bw / 2)
        y2 = int(y + bh / 2)
        
        # Recorte con límites válidos
        face_crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        
        if face_crop.size == 0:
            face_crop = img  # fallback
        
        face_img = Image.fromarray(face_crop)
        face_tensor = self.transform(face_img)
        
        label = int(cls)
        
        return face_tensor, label

