import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import json
import random
import logging
from PIL import Image, ImageFile
from pathlib import Path
from typing import Literal, List, Dict, Tuple, Optional, Union
import numpy as np

# 允许加载截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealFakeDataset(Dataset):
    """
    真假图像数据集，支持每个类别采样相同数量的样本
    
    Args:
        meta_json: 元数据JSON文件路径
        samples_per_class: 每个类别采样的样本数量，-1表示使用该类别的所有样本
        mode: 数据集模式，'train'或'val'
        img_size: 图像尺寸
        max_retry: 加载失败时的最大重试次数
        augmentation_strength: 数据增强强度
    """
    
    REAL_LABEL = 0
    FAKE_LABEL = 1
    LABEL_NAMES = {REAL_LABEL: "real", FAKE_LABEL: "fake"}
    
    def __init__(
        self,
        meta_json: Union[str, Path],
        samples_per_class: int = 1000,
        mode: Literal["train", "val", "test"] = "train",
        img_size: int = 224,
        max_retry: int = 3,
        augmentation_strength: Literal["light", "medium", "strong"] = "medium"
    ):
        self.meta_json = Path(meta_json)
        self.samples_per_class = samples_per_class
        self.mode = mode
        self.img_size = img_size
        self.max_retry = max_retry
        self.augmentation_strength = augmentation_strength
        
        # 验证输入参数
        self._validate_inputs()
        
        # 加载和处理元数据
        self.samples = self._load_and_process_metadata()
        
        # 构建变换
        self.transform = self._build_transforms()
        
        # 统计信息
        self._log_dataset_stats()
    
    def _validate_inputs(self) -> None:
        """验证输入参数"""
        if not self.meta_json.exists():
            raise FileNotFoundError(f"元数据文件不存在: {self.meta_json}")
        
        if self.img_size <= 0:
            raise ValueError(f"图像尺寸必须为正数: {self.img_size}")
        
        if self.max_retry < 0:
            raise ValueError(f"最大重试次数必须非负: {self.max_retry}")
        
        if self.samples_per_class == 0:
            raise ValueError("每个类别的样本数不能为0")
    
    def _load_and_process_metadata(self) -> List[Tuple[str, int]]:
        """加载和处理元数据，返回(路径, 标签)的列表"""
        try:
            with open(self.meta_json, 'r', encoding='utf-8') as f:
                meta_info = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"无法解析元数据文件 {self.meta_json}: {e}")
        
        if not isinstance(meta_info, dict):
            raise ValueError("元数据格式错误，应为字典格式")
        
        all_samples = []
        real_samples_collected = []
        fake_samples_collected = []
        
        # 收集所有类别的样本
        for cls_name, cls_data in meta_info.items():
            if not isinstance(cls_data, dict):
                logger.warning(f"跳过无效类别数据: {cls_name}")
                continue
            
            # 收集真实样本
            real_samples = cls_data.get('real', [])
            if real_samples:
                valid_real = self._validate_file_paths(real_samples, cls_name, "real")
                real_samples_collected.extend(valid_real)
            
            # 收集虚假样本
            fake_samples = cls_data.get('fake', [])
            if fake_samples:
                valid_fake = self._validate_file_paths(fake_samples, cls_name, "fake")
                fake_samples_collected.extend(valid_fake)
        
        # 对每个类别进行采样
        if self.samples_per_class == -1:
            # 使用所有样本
            sampled_real = real_samples_collected
            sampled_fake = fake_samples_collected
        else:
            # 采样指定数量
            sampled_real = self._sample_from_list(real_samples_collected, self.samples_per_class)
            sampled_fake = self._sample_from_list(fake_samples_collected, self.samples_per_class)
        
        # 构建最终样本列表
        for path in sampled_real:
            all_samples.append((path, self.REAL_LABEL))
        
        for path in sampled_fake:
            all_samples.append((path, self.FAKE_LABEL))
        
        if not all_samples:
            raise ValueError("没有找到有效的样本文件")
        
        # 打乱样本顺序
        random.shuffle(all_samples)
        
        return all_samples
    
    def _sample_from_list(self, items: List[str], target_count: int) -> List[str]:
        """从列表中采样指定数量的项目"""
        if len(items) <= target_count:
            return items.copy()
        else:
            # 随机采样
            return random.sample(items, target_count)
    
    def _validate_file_paths(self, file_paths: List[str], cls_name: str, label_type: str) -> List[str]:
        """验证文件路径是否存在"""
        valid_paths = []
        invalid_count = 0
        
        for path in file_paths:
            if Path(path).exists():
                valid_paths.append(path)
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            logger.warning(f"类别 {cls_name} 的 {label_type} 样本中有 {invalid_count} 个文件不存在")
        
        return valid_paths
    
    def _build_transforms(self) -> transforms.Compose:
        """构建数据变换pipeline"""
        # CLIP标准化参数
        normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        
        base_transforms = [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            normalize
        ]
        
        if self.mode == "train":
            augmentation_transforms = self._get_augmentation_transforms()
            transforms_list = augmentation_transforms + base_transforms
        else:
            # 验证/测试模式使用更温和的变换
            resize_transforms = [
                transforms.Resize(
                    size=(self.img_size, self.img_size),
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
            ]
            transforms_list = resize_transforms + base_transforms
            
        return transforms.Compose(transforms_list)
    
    def _get_augmentation_transforms(self) -> List:
        """根据增强强度获取数据增强变换"""
        base_aug = [
            transforms.RandomResizedCrop(
                size=(self.img_size, self.img_size),
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
        ]
        
        if self.augmentation_strength == "light":
            return base_aug + [
                transforms.RandomHorizontalFlip(p=0.3),
            ]
        elif self.augmentation_strength == "medium":
            return base_aug + [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ]
        elif self.augmentation_strength == "strong":
            return base_aug + [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=5),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ]
        else:
            raise ValueError(f"不支持的增强强度: {self.augmentation_strength}")
    
    def _log_dataset_stats(self) -> None:
        """记录数据集统计信息"""
        class_counts = self.get_class_distribution()
        logger.info(f"数据集统计 - 模式: {self.mode}")
        logger.info(f"真假样本采样数量: {self.samples_per_class if self.samples_per_class != -1 else '全部'}")
        logger.info(f"真实样本: {class_counts['real']}")
        logger.info(f"虚假样本: {class_counts['fake']}")
        logger.info(f"总样本数: {len(self)}")
        logger.info(f"图像尺寸: {self.img_size}x{self.img_size}")
        logger.info(f"增强强度: {self.augmentation_strength}")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """获取单个样本"""
        sample_path, label = self.samples[index]
        
        for attempt in range(self.max_retry + 1):
            try:
                img = self._load_image(sample_path)
                img_tensor = self.transform(img)
                return img_tensor, label
                
            except Exception as e:
                if attempt == self.max_retry:
                    logger.error(f"加载样本失败，跳过该样本 (索引: {index}, 路径: {sample_path}): {e}")
                    # 直接抛出异常，不返回fallback
                    raise RuntimeError(f"无法加载样本 {sample_path}") from e
                else:
                    logger.warning(f"加载样本失败，重试 {attempt + 1}/{self.max_retry}: {e}")
    
    def _load_image(self, image_path: str) -> Image.Image:
        """安全加载图像"""
        try:
            img = Image.open(image_path).convert("RGB")
            # 验证图像是否有效
            img.verify()
            # 重新打开图像用于变换（verify会关闭文件）
            img = Image.open(image_path).convert("RGB")
            return img
        except Exception as e:
            raise IOError(f"无法加载图像 {image_path}: {e}")
    
    def get_class_distribution(self) -> Dict[str, int]:
        """获取类别分布"""
        real_count = sum(1 for _, label in self.samples if label == self.REAL_LABEL)
        fake_count = sum(1 for _, label in self.samples if label == self.FAKE_LABEL)
        
        return {
            self.LABEL_NAMES[self.REAL_LABEL]: real_count,
            self.LABEL_NAMES[self.FAKE_LABEL]: fake_count
        }
    
    def get_sample_paths_by_class(self) -> Dict[str, List[str]]:
        """获取每个类别的样本路径"""
        real_paths = [path for path, label in self.samples if label == self.REAL_LABEL]
        fake_paths = [path for path, label in self.samples if label == self.FAKE_LABEL]
        
        return {
            self.LABEL_NAMES[self.REAL_LABEL]: real_paths,
            self.LABEL_NAMES[self.FAKE_LABEL]: fake_paths
        }
    
    def set_augmentation_strength(self, strength: Literal["light", "medium", "strong"]) -> None:
        """动态设置增强强度"""
        if self.mode == "train":
            self.augmentation_strength = strength
            self.transform = self._build_transforms()
            logger.info(f"增强强度已更新为: {strength}")
        else:
            logger.warning("非训练模式下无法更改增强强度")


def create_dataloader(
    dataset: RealFakeDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True
) -> DataLoader:
    """创建优化的数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,  # 保持worker进程
        prefetch_factor=2 if num_workers > 0 else None,  # 预取因子
    )


def create_datasets_and_loaders(
    train_json: str,
    val_json: str,
    samples_per_class_train: int = 1000,
    samples_per_class_val: int = 500,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据集及加载器"""
    
    # 创建训练数据集
    train_dataset = RealFakeDataset(
        meta_json=train_json,
        samples_per_class=samples_per_class_train,
        mode="train",
        img_size=img_size,
        augmentation_strength="medium"
    )
    
    # 创建验证数据集
    val_dataset = RealFakeDataset(
        meta_json=val_json,
        samples_per_class=samples_per_class_val,
        mode="val",
        img_size=img_size
    )
    
    # 创建数据加载器
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2)  # 验证时使用较少worker
    )
    
    return train_loader, val_loader


# 使用示例
if __name__ == "__main__":
    # 创建数据集 - 每个类别采样1000个样本
    train_dataset = RealFakeDataset(
        meta_json="./data/progan_train.json",
        samples_per_class=1000,  # 每个类别1000个样本
        mode="train",
        img_size=224,
        augmentation_strength="medium"
    )
    
    # 创建验证数据集 - 每个类别采样500个样本
    val_dataset = RealFakeDataset(
        meta_json="./data/progan_train.json",
        samples_per_class=500,   # 每个类别500个样本
        mode="val",
        img_size=224
    )
    
    # 或者使用便捷函数
    train_loader, val_loader = create_datasets_and_loaders(
        train_json="./data/progan_train.json",
        val_json="./data/progan_train.json",
        samples_per_class_train=1000,  # 训练集每个类别1000个样本
        samples_per_class_val=500,     # 验证集每个类别500个样本
        batch_size=32,
        img_size=224,
        num_workers=4
    )
    
    # 打印数据集信息
    print("训练集大小:", len(train_loader.dataset))
    print("验证集大小:", len(val_loader.dataset))
    print("训练集类别分布:", train_loader.dataset.get_class_distribution())
    print("验证集类别分布:", val_loader.dataset.get_class_distribution())