# ✅ [최종 통합 버전] baseline 코드 기반 개선 모델 (ver2 split + 하이퍼파라미터 수정 + 모델 개선)

# ============================
# 1. Import Libraries
# ============================
import os
import time
import tarfile
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch import nn
from torch.optim import Adam
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import timm
import cv2
from PIL import Image


# ============================
# 2. Image Dataset
# ============================
class ImageDataset(Dataset):
    def __init__(self, csv, path, transform=None):
        self.df = pd.read_csv(csv).values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, target


# ============================
# 3. Hyper-parameters
# ============================
device = torch.device("cpu")
data_path = "datasets_folder"

# 모델 및 학습 설정
model_name = "tf_efficientnet_b3"  # ✅ 모델 개선
img_size = 224                    # ✅ 이미지 크기 향상
LR = 2e-4                         # ✅ 학습률 조정
EPOCHS = 25                       # ✅ 에폭 수 증가
BATCH_SIZE = 64
num_workers = 0


# ============================
# 4. Transform
# ============================
trn_transform = A.Compose([
    A.Resize(img_size, img_size),
    A.HorizontalFlip(p=0.5),            # ✅ 추가 augmentation
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

tst_transform = A.Compose([
    A.Resize(img_size, img_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# ============================
# 5. Dataset & DataLoader
# ============================
# 🟢 ver2 (validation 포함)
train_csv = r"C:\Users\재형띠\Desktop\코딩친구들\컴퓨터비전 프로젝트\vaildation_set_test\ver2_rare_class_train.csv"
valid_csv = r"C:\Users\재형띠\Desktop\코딩친구들\컴퓨터비전 프로젝트\vaildation_set_test\ver2_rare_class_valid.csv"
img_dir = r"C:\Users\재형띠\Desktop\코딩친구들\컴퓨터비전 프로젝트\datasets_folder\data\train"

# Dataset
trn_dataset = ImageDataset(train_csv, img_dir, transform=trn_transform)
val_dataset = ImageDataset(valid_csv, img_dir, transform=tst_transform)
tst_dataset = ImageDataset(
    r"C:\Users\재형띠\Desktop\코딩친구들\컴퓨터비전 프로젝트\datasets_folder\data\sample_submission.csv",
    r"C:\Users\재형띠\Desktop\코딩친구들\컴퓨터비전 프로젝트\datasets_folder\data\test",
    transform=tst_transform
)

# Dataloader
trn_loader = DataLoader(trn_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
tst_loader = DataLoader(tst_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
print(len(trn_dataset), len(tst_dataset), len(val_loader))



# ============================
# 6. Train Function
# ============================
def train_one_epoch(loader, model, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    targets_list = []
    preds_list = []

    pbar = tqdm(loader)
    for image, targets in pbar:
        image = image.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(image)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds_list.extend(preds.argmax(1).detach().cpu().numpy())
        targets_list.extend(targets.cpu().numpy())

        pbar.set_description(f"Loss: {loss.item():.4f}")

    train_loss = total_loss / len(loader)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')

    return {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_f1': train_f1,
    }

# ============================
# 7. vali Function
# ============================
def eval_one_epoch(loader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    targets_list = []
    preds_list = []

    with torch.no_grad():
        for image, targets in loader:
            image = image.to(device)
            targets = targets.to(device)

            preds = model(image)
            loss = loss_fn(preds, targets)

            total_loss += loss.item()
            preds_list.extend(preds.argmax(1).cpu().numpy())
            targets_list.extend(targets.cpu().numpy())

    val_loss = total_loss / len(loader)
    val_acc = accuracy_score(targets_list, preds_list)
    val_f1 = f1_score(targets_list, preds_list, average='macro')

    return {
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_f1': val_f1,
    }

# ============================
# 8. Train Model
# ============================
model = timm.create_model(model_name, pretrained=True, num_classes=17).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    train_metrics = train_one_epoch(trn_loader, model, optimizer, loss_fn, device)
    val_metrics = eval_one_epoch(val_loader, model, loss_fn, device)

    print(f"[Epoch {epoch}]")
    print(f"Train | Loss: {train_metrics['train_loss']:.4f}, Acc: {train_metrics['train_acc']:.4f}, F1: {train_metrics['train_f1']:.4f}")
    print(f"Valid | Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_acc']:.4f}, F1: {val_metrics['val_f1']:.4f}")


# ============================
# 9. Inference & Save
# ============================
preds_list = []
model.eval()

for image, _ in tqdm(tst_loader):
    image = image.to(device)
    with torch.no_grad():
        preds = model(image)
        preds_list.extend(preds.argmax(1).detach().cpu().numpy())

pred_df = pd.DataFrame(tst_dataset.df, columns=['ID', 'target'])
pred_df['target'] = preds_list

sample_submission = pd.read_csv(r"C:\Users\재형띠\Desktop\코딩친구들\컴퓨터비전 프로젝트\datasets_folder\data\sample_submission.csv")
assert (sample_submission['ID'] == pred_df['ID']).all()
pred_df.to_csv("pred_ver2_efficientnet.csv", index=False)

print("✅ Prediction Saved!")
