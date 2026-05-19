#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
U-Net 图像分割训练脚本
数据集: Stanford Background Dataset
- 8 个语义类别: 0=sky, 1=tree, 2=road, 3=grass, 4=water, 5=building, 6=mountain, 7=foreground
- -1 表示 unknown/未标注，不参与损失计算
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import random
import warnings
warnings.filterwarnings('ignore')

# 可选: 使用 wandb 记录实验
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed, skipping logging")

# ==================== 设置随机种子 ====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# ==================== 数据集类 ====================
class StanfordBackgroundDataset(Dataset):
    """
    Stanford Background Dataset
    语义类别: 0=sky, 1=tree, 2=road, 3=grass, 4=water, 5=building, 6=mountain, 7=foreground
    标签值 -1 表示 unknown/未标注，训练时忽略
    """
    def __init__(self, images_dir, labels_dir, target_size=(256, 256)):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.target_size = target_size
        self.num_classes = 8
        self.class_names = ['sky', 'tree', 'road', 'grass', 'water', 'building', 'mountain', 'foreground']
        self.ignore_index = -1  # unknown 标签
        
        # 获取所有图像文件
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        print(f"Found {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def read_label(self, label_path):
        """
        读取标签文件，保留 -1 作为 unknown
        """
        with open(label_path, 'r') as f:
            content = f.read().strip()
        
        # 按空白字符分割
        numbers = content.split()
        label_values = []
        
        for num_str in numbers:
            try:
                val = int(num_str)
                label_values.append(val)
            except ValueError:
                label_values.append(self.ignore_index)
        
        total_pixels = len(label_values)
        
        # 推断图像尺寸 (约 320x240)
        if total_pixels == 76800:      # 320x240
            height, width = 240, 320
        else:
            # 尝试找到最接近的矩形形状
            height = int(np.sqrt(total_pixels))
            while total_pixels % height != 0 and height > 0:
                height -= 1
            width = total_pixels // height if height > 0 else 320
        
        label_map = np.array(label_values, dtype=np.int64).reshape((height, width))
        return label_map
    
    def __getitem__(self, idx):
        # 读取图像
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取标签
        stem = img_path.stem
        label_path = self.labels_dir / f"{stem}.regions.txt"
        if not label_path.exists():
            raise RuntimeError(f"Label file not found: {label_path}")
        label = self.read_label(label_path)
        
        # 调整尺寸
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        # 使用最近邻插值保持标签值不变
        label = cv2.resize(label, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # 归一化图像
        image = image.astype(np.float32) / 255.0
        
        # 转换为 Tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        label = torch.from_numpy(label).long()
        
        return image, label


# ==================== U-Net 模型 ====================
class DoubleConv(nn.Module):
    """双卷积块 (Conv -> BN -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """下采样模块: MaxPool + DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样模块: Upsample + Skip Connection + DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 处理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Skip Connection: 特征拼接
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    """输出卷积层"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    完整的 U-Net 架构
    输入: (B, 3, H, W)
    输出: (B, n_classes, H, W)
    """
    def __init__(self, n_channels=3, n_classes=8, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        # 编码器
        self.inc = DoubleConv(n_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        # 瓶颈层
        self.down4 = Down(features[3], features[3] * 2)
        
        # 解码器
        self.up1 = Up(features[3] * 2, features[3])
        self.up2 = Up(features[3], features[2])
        self.up3 = Up(features[2], features[1])
        self.up4 = Up(features[1], features[0])
        
        # 输出层
        self.outc = OutConv(features[0], n_classes)
    
    def forward(self, x):
        # 编码路径
        x1 = self.inc(x)           # 64
        x2 = self.down1(x1)        # 128
        x3 = self.down2(x2)        # 256
        x4 = self.down3(x3)        # 512
        x5 = self.down4(x4)        # 1024
        
        # 解码路径 (带 Skip Connection)
        x = self.up1(x5, x4)       # 512
        x = self.up2(x, x3)        # 256
        x = self.up3(x, x2)        # 128
        x = self.up4(x, x1)        # 64
        
        # 输出
        logits = self.outc(x)
        
        return logits


# ==================== 损失函数 ====================
# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6, ignore_index=-1):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth
#         self.ignore_index = ignore_index
    
#     def forward(self, pred, target):
#         # pred: (B, C, H, W), target: (B, H, W)
#         B, C, H, W = pred.shape
        
#         # 创建 mask（-1 的位置需要忽略）
#         mask = (target != self.ignore_index).float().unsqueeze(1)  # (B, 1, H, W)
        
#         # Softmax
#         pred_softmax = F.softmax(pred, dim=1)
        
#         # 将 -1 临时替换为 0（用于 one_hot）
#         target_safe = target.clamp(min=0)  # -1 -> 0
#         target_one_hot = F.one_hot(target_safe, num_classes=C)  # (B, H, W, C)
#         target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
#         # 将 ignore_index 位置设为 0
#         target_one_hot = target_one_hot * mask
        
#         # 应用 mask 到预测
#         pred_masked = pred_softmax * mask
        
#         # 计算 Dice
#         intersection = (pred_masked * target_one_hot).sum(dim=(0, 2, 3))
#         union = pred_masked.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
        
#         # 避免除零
#         dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
#         # 只对有效类别求平均
#         valid = (union > 0)
#         if valid.sum() > 0:
#             dice = dice[valid].mean()
#         else:
#             dice = torch.tensor(1.0, device=pred.device)
        
#         return 1 - dice

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6, ignore_index=-1):
#         super(DiceLoss, self).__init__()
#         self.dice = smp.losses.DiceLoss(mode='multiclass', 
#                                          classes=8, 
#                                          smooth=smooth, 
#                                          ignore_index=ignore_index)
#     def forward(self, pred, target):
#         return self.dice(pred, target)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=-1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        B, C, H, W = pred.shape
        
        # 关键改进1：使用 log_softmax + exp
        pred = pred.log_softmax(dim=1).exp()
        
        # 处理 target
        mask = (target != self.ignore_index)
        target_safe = target.clone()
        target_safe[~mask] = 0
        target_one_hot = F.one_hot(target_safe, num_classes=C).permute(0, 3, 1, 2).float()
        
        # 展平
        pred_flat = pred.view(B, C, -1)
        target_flat = target_one_hot.view(B, C, -1)
        
        # 关键改进2：正确应用 mask
        mask_flat = mask.view(B, 1, -1)
        pred_flat = pred_flat * mask_flat
        target_flat = target_flat * mask_flat
        
        # 计算 Dice
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 关键改进3：只对有效类别平均
        valid = (union > 0)
        if valid.any():
            dice = dice[valid].mean()
        else:
            dice = torch.tensor(1.0, device=pred.device)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    """
    CrossEntropyLoss + DiceLoss 组合
    两者都忽略 ignore_index
    """
    def __init__(self, ce_weight=1.0, dice_weight=0.1, ignore_index=-1):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice


# ==================== 评估指标 ====================
def compute_miou(pred, target, num_classes=8, ignore_index=-1):
    """
    计算平均交并比 (Mean IoU)
    忽略 ignore_index 的像素
    """
    pred = torch.argmax(pred, dim=1)
    
    # 创建 mask，排除 ignore_index
    mask = (target != ignore_index)
    pred = pred[mask]
    target = target[mask]
    
    ious = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        if union == 0:
            iou = float('nan')
        else:
            iou = (intersection / union).item()
        ious.append(iou)
    
    # 忽略无效类
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(valid_ious) if valid_ious else 0.0


def evaluate(model, dataloader, device, num_classes=8, ignore_index=-1):
    """
    评估模型
    """
    model.eval()
    total_loss = 0.0
    total_miou = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            miou = compute_miou(outputs, labels, num_classes, ignore_index)
            total_miou += miou
    
    return total_loss / len(dataloader), total_miou / len(dataloader)


# ==================== 训练函数 ====================
def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    训练一个 epoch
    """
    model.train()
    total_loss = 0.0
    
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                epochs, device, exp_name, num_classes=8, ignore_index=-1):
    """
    完整训练流程
    """
    if WANDB_AVAILABLE:
        wandb.init(project="unet_segmentation", name=exp_name, config={
            "model": "U-Net",
            "dataset": "Stanford Background",
            "num_classes": num_classes,
            "ignore_index": ignore_index,
            "epochs": epochs,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "loss_function": exp_name
        })
    
    best_miou = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_miou': []
    }
    
    for epoch in range(epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)
        
        # 验证
        val_loss, val_miou = evaluate(model, val_loader, device, num_classes, ignore_index)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_miou)
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'loss_name': exp_name
            }, f"best_model_{exp_name}.pt")
        
        if WANDB_AVAILABLE:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mIoU": val_miou,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        print(f"[{exp_name}] Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, "
              f"mIoU={val_miou:.4f}")
    
    if WANDB_AVAILABLE:
        wandb.finish()
    
    return history, best_miou


# ==================== 主程序 ====================
def main():
    # ==================== 配置参数 ====================
    data_dir = "/home/shijc/U-net/iccv09Data"
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    batch_size = 4          # 批大小
    epochs = 20             # 训练轮数
    learning_rate = 1e-4   # 学习率
    image_size = (256, 256) # 图像尺寸
    num_workers = 2         # 数据加载线程数
    test_size = 0.05         # 验证集比例
    ignore_index = -1       # unknown 标签
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ==================== 加载数据集 ====================
    print("\n" + "="*60)
    print("Loading Stanford Background Dataset...")
    print("="*60)
    
    dataset = StanfordBackgroundDataset(images_dir, labels_dir, target_size=image_size)
    
    # 划分训练集和验证集
    train_idx, val_idx = train_test_split(
        range(len(dataset)), 
        test_size=test_size, 
        random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # 检查一个 batch 的数据
    print("\n" + "="*60)
    print("Checking one batch...")
    print("="*60)
    for images, labels in train_loader:
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Label unique values: {torch.unique(labels)}")
        print(f"Number of -1 (unknown): {(labels == -1).sum().item()}")
        break
    
    # ==================== 定义损失函数配置 ====================
    loss_configs = [
        ("CE", nn.CrossEntropyLoss(ignore_index=ignore_index)),
        ("Dice", DiceLoss(ignore_index=ignore_index)),
        ("CE_Dice", CombinedLoss(ignore_index=ignore_index))
    ]
    
    results = {}
    
    # ==================== 训练不同配置 ====================
    for loss_name, criterion in loss_configs:
        print("\n" + "="*60)
        print(f"Training with {loss_name} Loss")
        print("="*60)
        
        # 重新初始化模型（确保从头开始）
        model = UNet(n_channels=3, n_classes=dataset.num_classes).to(device)
        
        # 优化器和调度器
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 训练
        history, best_miou = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            device=device,
            exp_name=loss_name,
            num_classes=dataset.num_classes,
            ignore_index=ignore_index
        )
        
        results[loss_name] = {
            'history': history,
            'best_miou': best_miou
        }
        
        print(f"\n{loss_name} - Best mIoU: {best_miou:.4f}")
    
    # ==================== 绘制结果对比图 ====================
    print("\n" + "="*60)
    print("Plotting Results...")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    colors = {"CE": "blue", "Dice": "green", "CE_Dice": "red"}
    
    # 损失曲线
    for loss_name in results:
        axes[0].plot(
            results[loss_name]['history']['train_loss'], 
            color=colors[loss_name], 
            label=f"{loss_name} Train"
        )
        axes[0].plot(
            results[loss_name]['history']['val_loss'], 
            color=colors[loss_name], 
            linestyle='--',
            label=f"{loss_name} Val"
        )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True)
    
    # mIoU 曲线
    for loss_name in results:
        axes[1].plot(
            results[loss_name]['history']['val_miou'], 
            color=colors[loss_name], 
            label=loss_name
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].set_title("Validation mIoU Curves")
    axes[1].legend()
    axes[1].grid(True)
    
    # 最佳 mIoU 柱状图
    names = list(results.keys())
    best_mious = [results[n]['best_miou'] for n in names]
    bars = axes[2].bar(names, best_mious, color=[colors[n] for n in names])
    axes[2].set_ylabel("Best mIoU")
    axes[2].set_title("Best Validation mIoU Comparison")
    for bar, val in zip(bars, best_mious):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                     f"{val:.4f}", ha='center', va='bottom')
    axes[2].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig("comparison_results.png", dpi=150)
    plt.show()
    
    # ==================== 打印最终结果 ====================
    print("\n" + "="*60)
    print("Final Results Summary")
    print("="*60)
    for loss_name in results:
        print(f"  {loss_name:10s}: Best mIoU = {results[loss_name]['best_miou']:.4f}")
    
    # 保存结果到 JSON
    import json
    with open("experiment_results.json", "w") as f:
        json.dump({
            name: {
                "best_miou": results[name]["best_miou"],
                "final_train_loss": results[name]["history"]["train_loss"][-1],
                "final_val_loss": results[name]["history"]["val_loss"][-1]
            }
            for name in results
        }, f, indent=2)
    
    print("\n✅ Training completed!")
    print("Results saved to: comparison_results.png, experiment_results.json")
    print("Best models saved as: best_model_CE.pt, best_model_Dice.pt, best_model_CE_Dice.pt")


if __name__ == "__main__":
    main()