import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
from tqdm import tqdm
import argparse

# 导入自定义模块
from clip import ModifiedCLIP
from dateset import create_datasets_and_loaders
from loss import MultiPosConLossMM

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    ModifiedCLIP模型训练器
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        device: 训练设备
        save_dir: 模型保存目录
        log_interval: 日志打印间隔
    """
    
    def __init__(
        self,
        model: ModifiedCLIP,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        save_dir: str = "./checkpoints",
        log_interval: int = 100
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.multi_pos_con_loss_mm = MultiPosConLossMM(temperature=0.1)
        self.device = device
        self.save_dir = Path(save_dir)
        self.log_interval = log_interval
        
        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史记录
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # 最佳模型追踪
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        logger.info(f"训练器初始化完成，模型保存目录: {self.save_dir}")
        
    def compute_loss(self,
                 logits_per_image: torch.Tensor,
                 labels: torch.Tensor,            # 图像标签 [B]
                 image_features: torch.Tensor,    # 图像特征 [B, 768]
                 text_features: torch.Tensor,     # 文本特征 [2, 768]
                 logit_scale: float        # 缩放因子
                 ) -> torch.Tensor:
        """
        用 MultiPosConLossMM 计算图像与文本的多正例对比损失。
        假设 text_features[0] 标签为 0，text_features[1] 标签为 1。
        """

        device = labels.device
        text_labels = torch.tensor([0, 1], dtype=torch.long, device=device)  # 明确标签

        outputs = {
            'image_emb': image_features,         # 用于 image-text 对比
            'text_emb': text_features,           # [2, 768]
            'image_feats': image_features,       # 用于 image-only contrastive loss
            'image_labels': labels,              # 图像标签 [B]
            'text_labels': text_labels,          # 文本标签 [2]
            'logit_scale': logit_scale           # 缩放因子
        }

        loss_dict = self.multi_pos_con_loss_mm(outputs)
        
        orthogonal_loss = torch.dot(text_features[0], text_features[1])** 2
        bce_loss = F.cross_entropy(logits_per_image, labels)
        return loss_dict['loss'] + orthogonal_loss + bce_loss

    
    def compute_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """计算准确率"""
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        return correct / total
    
    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(self.train_loader)
        
        # 创建进度条
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/20 [Train]')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # 将数据移到设备
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            image_features, text_features, logits_per_image, logits_per_text, logit_scale = self.model(images)
            
            # 计算损失
            loss = self.compute_loss(
                logits_per_image=logits_per_image,  # 图像logits
                labels=labels,                      # 图像标签 [B]
                image_features=image_features,      # [B, 768]
                text_features=text_features,        # [2, 768]
                logit_scale=logit_scale  # 常见做法是传入 exp(logit_scale)
            )

            # 反向传播
            loss.backward()
            
            # 梯度裁剪（可选）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 计算准确率
            accuracy = self.compute_accuracy(logits_per_image, labels)
            
            # 累积统计
            total_loss += loss.item()
            total_accuracy += accuracy
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # 定期打印日志
            if batch_idx % self.log_interval == 0:
                logger.info(
                    f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, '
                    f'Loss: {loss.item():.4f}, Acc: {accuracy:.4f}, '
                    f'LR: {self.optimizer.param_groups[0]["lr"]:.2e}'
                )
        
        # 计算平均值
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(self.val_loader)
        
        # 创建进度条
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/20 [Val]')
        
        with torch.no_grad():
            for images, labels in pbar:
                # 将数据移到设备
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                image_features, text_features, logits_per_image, logits_per_text, logit_scale = self.model(images)
            
                # 计算损失和准确率
                loss = self.compute_loss(
                    logits_per_image=logits_per_image,  # 图像logits
                    labels=labels,                      # 图像标签 [B]
                    image_features=image_features,      # [B, 768]
                    text_features=text_features,        # [2, 768]
                    logit_scale=logit_scale  # 常见做法是传入 exp(logit_scale)
                )
                accuracy = self.compute_accuracy(logits_per_image, labels)
                
                # 累积统计
                total_loss += loss.item()
                total_accuracy += accuracy
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{accuracy:.4f}'
                })
        
        # 计算平均值
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """保存模型检查点（只保存可训练的参数）"""
        # 获取模型中所有 requires_grad=True 的参数
        trainable_state_dict = {
            k: v for k, v in self.model.state_dict().items()
            if k in dict(self.model.named_parameters()) and self.model.get_parameter(k).requires_grad
        }

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': trainable_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'train_history': self.train_history,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }

        # 保存当前epoch的检查点
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch:02d}.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f'保存检查点: {checkpoint_path}')

        
    def save_training_history(self) -> None:
        """保存训练历史"""
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        logger.info(f'保存训练历史: {history_path}')
    
    def train(self, num_epochs: int = 20) -> None:
        """完整的训练流程"""
        logger.info(f"开始训练，共 {num_epochs} 个epoch")
        logger.info(f"训练集大小: {len(self.train_loader.dataset)}")
        logger.info(f"验证集大小: {len(self.val_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            # 训练一个epoch
            train_loss, train_acc = self.train_one_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate(epoch)
            
            # 更新学习率
            self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录训练历史
            self.train_history['epoch'].append(epoch)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['learning_rate'].append(current_lr)
            
            # 检查是否是最佳模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            # 保存检查点
            self.save_checkpoint(epoch, is_best)
            
            # 计算epoch时间
            epoch_time = time.time() - epoch_start_time
            
            # 打印epoch总结
            logger.info(
                f'Epoch {epoch:2d}/{num_epochs} 完成 - '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                f'LR: {current_lr:.2e}, Time: {epoch_time:.1f}s'
            )
            
            if is_best:
                logger.info(f'🎉 新的最佳验证准确率: {val_acc:.4f}')
        
        # 训练完成
        total_time = time.time() - start_time
        logger.info(f'训练完成！总时间: {total_time:.1f}s')
        logger.info(f'最佳验证准确率: {self.best_val_acc:.4f} (Epoch {self.best_epoch})')
        
        # 保存训练历史
        self.save_training_history()


def create_model_and_optimizer(args) -> Tuple[ModifiedCLIP, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """创建模型、优化器和学习率调度器"""
    # 创建模型
    model = ModifiedCLIP(
        backbone=args.backbone,
        prompt_length=args.prompt_length,
        text_adapt_until=args.text_adapt_until,
        enable_fsm=args.enable_fsm,
        classnames=["photo", "deepfake photo"]
    )
    
    # 将模型移到设备
    model = model.to(args.device)
    
    # 打印模型参数状态
    print(model.get_trainable_parameters())
    
    # 创建优化器 - 只优化可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate)
    
    # 创建学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=1e-5
    )
    
    logger.info(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"可训练参数数量: {sum(p.numel() for p in trainable_params):,}")
    
    return model, optimizer, lr_scheduler


def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='ModifiedCLIP 训练脚本')
    parser.add_argument('--train_json', type=str, default="./data/progan_train.json", help='训练集元数据JSON文件')
    parser.add_argument('--val_json', type=str, default="./data/progan_val.json", help='验证集元数据JSON文件')
    parser.add_argument('--samples_per_class_train', type=int, default=1000, help='训练集每个类别的样本数')
    parser.add_argument('--samples_per_class_val', type=int, default=500, help='验证集每个类别的样本数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--img_size', type=int, default=224, help='图像尺寸')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器worker数量')
    parser.add_argument('--backbone', type=str, default='ViT-L/14', help='CLIP骨干网络')
    parser.add_argument('--prompt_length', type=int, default=16, help='提示词长度')
    parser.add_argument('--text_adapt_until', type=int, default=3, help='文本适配器层数')
    parser.add_argument('--enable_fsm', type=bool, default=True, help='启用特征分离模块')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--device', type=str, default='cuda:2', help='训练设备')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    logger.info(f"使用设备: {device}")
    
    # 创建数据集和数据加载器
    logger.info("创建数据集...")
    train_loader, val_loader = create_datasets_and_loaders(
        train_json=args.train_json, 
        val_json=args.val_json,     
        samples_per_class_train=args.samples_per_class_train, 
        samples_per_class_val=args.samples_per_class_val,     
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    
    # 创建模型、优化器和调度器
    logger.info("创建模型...")
    model, optimizer, lr_scheduler = create_model_and_optimizer(args)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        save_dir=args.save_dir
    )
    
    # 开始训练
    trainer.train(num_epochs=args.num_epochs)


if __name__ == "__main__":
    main()


# 使用示例脚本
"""
python train.py \
    --train_json ./data/train_meta.json \
    --val_json ./data/val_meta.json \
    --samples_per_class_train 1000 \
    --samples_per_class_val 500 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 20 \
    --enable_fsm \
    --save_dir ./checkpoints \
    --device cuda:2
"""