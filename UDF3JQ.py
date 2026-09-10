import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cv2

from GOLDUnet import UNet
from GOLDtrain_unet import SegDataset


# ====== 训练函数 ======
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, masks in loader:
        if imgs.size(0) < 2:
            continue
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        if isinstance(outputs, dict):
            pred = outputs['out'] if 'out' in outputs else outputs['tensor']
        else:
            pred = outputs
        loss = criterion(pred, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader) if len(loader) > 0 else 0


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, masks in loader:
            if imgs.size(0) < 1:
                continue
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            if isinstance(outputs, dict):
                pred = outputs['out'] if 'out' in outputs else outputs['tensor']
            else:
                pred = outputs
            loss = criterion(pred, masks)
            total_loss += loss.item()
    return total_loss / len(loader) if len(loader) > 0 else 0


# ====== 计算指标（核心修改：添加加权准确率） ======
def compute_metrics(model, loader, device, num_classes=3):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            if imgs.size(0) < 1:
                continue
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            if isinstance(outputs, dict):
                pred_logits = outputs['out'] if 'out' in outputs else outputs['tensor']
            else:
                pred_logits = outputs
            preds = torch.argmax(pred_logits, dim=1).cpu().numpy()
            targets = masks.cpu().numpy()
            all_preds.extend(preds.flatten())
            all_targets.extend(targets.flatten())

    # 1. 生成混淆矩阵
    conf_mat = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    # 2. 原始整体像素准确率（等权重）
    original_acc = np.diag(conf_mat).sum() / conf_mat.sum() if conf_mat.sum() > 0 else 0

    # 3. 加权准确率计算（统一权重：背景0.1，EPB 0.45，EIA 0.45）
    class_weights = [0.1, 0.45, 0.45]  # 三个模型共用此权重
    # 计算每类的自身准确率（类别内正确像素占比）
    class_accs = []
    for i in range(num_classes):
        cls_total = conf_mat[i, :].sum()  # 该类总像素数
        cls_correct = conf_mat[i, i]      # 该类正确分类像素数
        class_acc = cls_correct / cls_total if cls_total > 0 else 0
        class_accs.append(class_acc)
    # 加权准确率 = 各类别准确率 × 对应权重 的总和
    weighted_acc = np.sum(np.array(class_accs) * np.array(class_weights))

    # 4. 保留原有IoU和Dice计算
    ious, dices = [], []
    for i in range(num_classes):
        tp = conf_mat[i, i]
        fp = conf_mat[:, i].sum() - tp
        fn = conf_mat[i, :].sum() - tp
        union = tp + fp + fn
        iou = tp / union if union > 0 else 0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        ious.append(iou)
        dices.append(dice)

    # 返回值新增：加权准确率、各类别自身准确率
    return original_acc, weighted_acc, class_accs, ious, dices


# ====== 解码函数（离散颜色） ======
def decode_mask(mask):
    colors = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0)}  # 背景黑，EPB红，EIA绿
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in colors.items():
        color_mask[mask == cls] = color
    return color_mask


# ====== 单模型可视化 ======
def visualize_predictions(model, loader, device, save_dir, model_name):
    os.makedirs(os.path.join(save_dir, model_name, "predict"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, model_name, "vis"), exist_ok=True)

    with torch.no_grad():
        for idx, (img, mask) in enumerate(loader):
            if img.size(0) < 1:
                continue
            img_tensor = img.to(device)
            output = model(img_tensor)
            if isinstance(output, dict):
                pred_logits = output['out'] if 'out' in output else output['tensor']
            else:
                pred_logits = output
            pred = torch.argmax(pred_logits, dim=1).squeeze().cpu().numpy()
            target = mask.squeeze().numpy()

            cv2.imwrite(os.path.join(save_dir, model_name, "predict", f"pred_{idx:03d}.png"),
                        pred.astype(np.uint8) * 100)

            img_np = img.squeeze().permute(1, 2, 0).numpy()
            gt_mask = decode_mask(target)
            pred_mask = decode_mask(pred)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_np)
            axes[0].set_title("Original")
            axes[1].imshow(gt_mask)
            axes[1].set_title("Ground Truth")
            axes[2].imshow(pred_mask)
            axes[2].set_title("Prediction")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, model_name, "vis", f"vis_{idx:03d}.png"))
            plt.close()


# ====== 三模型对比可视化 ======
def compare_models(models, loader, device, save_dir):
    os.makedirs(os.path.join(save_dir, "comparison"), exist_ok=True)
    with torch.no_grad():
        for idx, (img, mask) in enumerate(loader):
            if img.size(0) < 1:
                continue
            img_tensor = img.to(device)
            target = mask.squeeze().numpy()
            preds = {}
            for name, model in models.items():
                output = model(img_tensor)
                if isinstance(output, dict):
                    pred_logits = output['out'] if 'out' in output else output['tensor']
                else:
                    pred_logits = output
                pred = torch.argmax(pred_logits, dim=1).squeeze().cpu().numpy()
                preds[name] = decode_mask(pred)

            img_np = img.squeeze().permute(1, 2, 0).numpy()
            gt_mask = decode_mask(target)
            fig, axes = plt.subplots(1, len(models) + 2, figsize=(16, 4))
            axes[0].imshow(img_np)
            axes[0].set_title("Original")
            axes[1].imshow(gt_mask)
            axes[1].set_title("Ground Truth")
            for i, (name, pred_mask) in enumerate(preds.items(), start=2):
                axes[i].imshow(pred_mask)
                axes[i].set_title(name)
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "comparison", f"compare_{idx:03d}.png"))
            plt.close()


# ====== 主程序（核心修改：接收并保存加权准确率） ======
if __name__ == "__main__":
    image_dir = r"./data/image"
    mask_dir = r"./data/mask"
    save_dir = r"./data/"
    os.makedirs(save_dir, exist_ok=True)

    dataset = SegDataset(image_dir, mask_dir)
    total_len = len(dataset)

    # 数据集划分逻辑不变
    train_len = int(0.7 * total_len)
    train_len = train_len if train_len % 2 == 0 else train_len - 1
    val_len = int(0.15 * total_len)
    val_len = val_len if val_len % 2 == 0 else val_len - 1
    test_len = total_len - train_len - val_len
    test_len = max(1, test_len)
    remaining = total_len - train_len - val_len - test_len
    train_len += remaining

    train_set, val_set, test_set = random_split(
        dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    # 定义三个模型（不变）
    models = {
        "UNet": UNet(n_classes=3).to(device),
        "DeepLabV3+": smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=3
        ).to(device),
        "FPN": smp.FPN(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=3
        ).to(device)
    }

    epochs = 50
    results = []
    history_dict = {}

    for name, model in models.items():
        print(f"\n===== 训练 {name} 模型 =====")
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            train_loss = train(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate(model, val_loader, criterion, device)
            print(f"[{name}] Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss and val_loss > 0:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{name.lower()}_best.pth")

        history_dict[name] = history

        # 加载最优模型测试（核心修改：接收加权准确率）
        if os.path.exists(f"{name.lower()}_best.pth"):
            model.load_state_dict(torch.load(f"{name.lower()}_best.pth", map_location=device))
            test_loss = evaluate(model, test_loader, criterion, device)
            # 调用修改后的compute_metrics，获取加权准确率
            original_acc, weighted_acc, class_accs, ious, dices = compute_metrics(model, test_loader, device)

            # 打印该模型的加权准确率结果（便于实时查看）
            print(f"\n{name} 模型测试结果：")
            print(f"原始准确率：{original_acc:.4f} | 加权准确率（BG=0.1）：{weighted_acc:.4f}")
            print(f"EPB准确率：{class_accs[1]:.4f} | EIA准确率：{class_accs[2]:.4f}")

            # 结果字典新增加权准确率字段
            results.append({
                "Model": name,
                "Best_Val_Loss": best_val_loss,
                "Test_Loss": test_loss,
                "Pixel_Acc": original_acc,       # 原有：原始准确率
                "Weighted_Acc": weighted_acc,    # 新增：加权准确率
                "IoU_Background": ious[0], "IoU_EPB": ious[1], "IoU_EIA": ious[2],
                "Dice_Background": dices[0], "Dice_EPB": dices[1], "Dice_EIA": dices[2],
                "ClassAcc_EPB": class_accs[1], "ClassAcc_EIA": class_accs[2]  # 可选：保存EPB/EIA单独准确率
            })

            visualize_predictions(model, test_loader, device, save_dir, name)

    # 后续可视化和保存逻辑不变，但会自动包含加权准确率
    compare_models(models, test_loader, device, save_dir)
    plot_training_curves(history_dict, os.path.join(save_dir, "training_curves.png"))

    if results:
        df = pd.DataFrame(results)
        df.to_csv("model_comparison_metrics.csv", index=False)  # CSV包含加权准确率
        plot_metrics_bar(df, save_dir)  # 生成加权准确率柱状图

    print("\n✅ 所有模型训练完成，结果已包含加权准确率（背景0.1，EPB/EIA 0.45）")
