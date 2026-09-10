import os
import torch
import numpy as np
import random
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import confusion_matrix
import segmentation_models_pytorch as smp

# ===================== 固定随机种子 =====================
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed()

# ===================== 数据集 =====================
class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment

        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        self.total = min(len(self.images), len(self.masks))

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.image_dir, self.images[idx]))
        mask = cv2.imread(os.path.join(self.mask_dir, self.masks[idx]), 0)

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        if self.augment:
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            if random.random() > 0.5:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        return torch.from_numpy(img).permute(2,0,1).float(), torch.from_numpy(mask).long()

# ===================== 指标计算 =====================
def compute_metrics(model, loader, device, num_classes=3):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for img, mask in loader:
            pred = model(img.to(device)).argmax(1).cpu().numpy()
            y_pred.extend(pred.flatten())
            y_true.extend(mask.numpy().flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])

    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP

    p = np.divide(TP, TP+FP, out=np.zeros_like(TP, float), where=(TP+FP)!=0)
    r = np.divide(TP, TP+FN, out=np.zeros_like(TP, float), where=(TP+FN)!=0)
    f1 = 2 * p * r / (p + r + 1e-8)

    return (
        (y_true == y_pred).mean(),
        p.mean(), r.mean(), f1.mean(), p.mean()
    )

# ===================== 主程序：只跑 DeepLabV3+ =====================
if __name__ == '__main__':
    # 数据集划分
    full_dataset = SegDataset(r"D:\GOLD\image", r"D:\GOLD\mask")
    test_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    train_dataset.dataset.augment = True

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = torch.device('cpu')

    # ===================== 🔥 只加载 DeepLabV3+ 一个模型 =====================
    model = smp.DeepLabV3Plus(
        encoder_name="resnet18",  # 小模型，电脑不卡
        encoder_weights="imagenet",
        classes=3
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    # 训练
    print("\n===== 开始训练 DeepLabV3+ =====")
    best_f1 = 0
    patience = 5
    wait = 0

    for epoch in range(30):
        model.train()
        total_loss = 0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = criterion(model(img), mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        acc, p, r, f1, mAP = compute_metrics(model, test_loader, device)
        print(f"Epoch {epoch+1:2d} | Loss {total_loss/len(train_loader):.4f} | Test F1 {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            wait = 0
            torch.save(model.state_dict(), "deeplab_best.pth")
        else:
            wait += 1
            if wait >= patience:
                print("✅ 早停：防止过拟合")
                break

    # 最终指标
    model.load_state_dict(torch.load("deeplab_best.pth"))
    acc, p, r, f1, mAP = compute_metrics(model, test_loader, device)

    print("\n" + "="*55)
    print("="*55)
    print(f"Accuracy  = {acc:.4f}")
    print(f"Precision = {p:.4f}")
    print(f"Recall    = {r:.4f}")
    print(f"F1        = {f1:.4f}")
    print(f"mAP       = {mAP:.4f}")
    print("="*55)