import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

# 导入自定义模块
from clip import ModifiedCLIP
from dateset import RealFakeDataset

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_test_datasets(data_dir: str, img_size: int = 224) -> List[Tuple[str, RealFakeDataset]]:
    """
    创建测试数据集
    
    Args:
        data_dir: 数据目录路径
        img_size: 图像尺寸
    
    Returns:
        测试数据集列表
    """
    dataset_files = [
        'progan.json', 'stylegan.json', 'stylegan2.json', 'biggan.json',
        'cyclegan.json', 'stargan.json', 'gaugan.json', 'deepfake.json',
        'guided.json', 'ldm_200.json', 'ldm_200_cfg.json', 'ldm_100.json',
        'glide_100_10.json', 'glide_100_27.json', 'glide_50_27.json', 'dalle2.json'
    ]
    #guided Also known as ADM.
    test_datasets = []
    data_path = Path(data_dir)
    
    for dataset_file in dataset_files:
        json_path = data_path / dataset_file
        if json_path.exists():
            try:
                dataset = RealFakeDataset(
                    meta_json=json_path,
                    samples_per_class=-1,  # 使用所有样本
                    mode="val",
                    img_size=img_size
                )
                test_datasets.append((dataset_file, dataset))
                logger.info(f"成功加载测试数据集: {dataset_file}, 样本数: {len(dataset)}")
            except Exception as e:
                logger.warning(f"跳过数据集 {dataset_file}: {e}")
        else:
            logger.warning(f"数据集文件不存在: {json_path}")
    
    return test_datasets

class Tester:
    def __init__(
        self,
        model: ModifiedCLIP,
        test_datasets: List[Tuple[str, RealFakeDataset]],
        device: torch.device,
        save_dir: str = "./test_results"
    ):
        self.model = model
        self.test_datasets = test_datasets
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 测试结果记录
        self.test_results = {}
        
    def compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """计算准确率和AP指标"""
        # 转换为numpy数组
        logits_np = logits.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # 计算预测概率（对于二分类，取正类概率）
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        
        # 计算预测标签
        predictions = np.argmax(logits_np, axis=1)
        
        # 准确率
        accuracy = np.mean(predictions == labels_np)
        
        # AP值（Average Precision）
        try:
            ap = average_precision_score(labels_np, probs)
        except:
            ap = 0.0
        
        return {
            'accuracy': accuracy,
            'ap': ap
        }
    
    def test_dataset(self, dataset: RealFakeDataset, dataset_name: str) -> Dict[str, float]:
        """测试单个数据集"""
        self.model.eval()
        
        # 创建数据加载器
        test_loader = DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f'Testing {dataset_name}'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                image_features, text_features, logits_per_image, logits_per_text, logit_scale = self.model(images)
                
                all_logits.append(logits_per_image)
                all_labels.append(labels)
        
        # 计算指标
        if all_logits:
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            metrics = self.compute_metrics(all_logits, all_labels)
        else:
            metrics = {'accuracy': 0.0, 'ap': 0.0}
        
        return metrics
    
    def test_all_datasets(self) -> Dict[str, Dict[str, float]]:
        """测试所有数据集"""
        logger.info("开始测试所有数据集...")
        
        all_results = {}
        all_metrics = []
        
        for dataset_name, dataset in self.test_datasets:
            try:
                metrics = self.test_dataset(dataset, dataset_name)
                all_results[dataset_name] = metrics
                all_metrics.append(metrics)
                
                logger.info(
                    f"测试 {dataset_name}: "
                    f"Acc={metrics['accuracy']:.4f}, "
                    f"AP={metrics['ap']:.4f}"
                )
            except Exception as e:
                logger.error(f"测试数据集 {dataset_name} 时出错: {e}")
                all_results[dataset_name] = {
                    'accuracy': 0.0, 
                    'ap': 0.0
                }
        
        # 计算平均指标
        avg_metrics = {
            'avg_accuracy': np.mean([r['accuracy'] for r in all_metrics]),
            'avg_ap': np.mean([r['ap'] for r in all_metrics])
        }
        
        logger.info(
            f"平均测试结果: "
            f"Avg_Acc={avg_metrics['avg_accuracy']:.4f}, "
            f"Avg_AP={avg_metrics['avg_ap']:.4f}"
        )
        
        # 保存测试结果
        results = {
            'per_dataset': all_results,
            'average': avg_metrics
        }
        
        results_path = self.save_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f'保存测试结果: {results_path}')
        
        return all_results, avg_metrics

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='ModifiedCLIP 测试脚本')
    parser.add_argument('--test_data_dir', type=str, default="./data", help='测试数据集目录')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--img_size', type=int, default=224, help='图像尺寸')
    parser.add_argument('--backbone', type=str, default='ViT-L/14', help='CLIP骨干网络')
    parser.add_argument('--prompt_length', type=int, default=16, help='提示词长度')
    parser.add_argument('--class_names', type=str, nargs='+', default=["real", "fake"], help='类别名称')
    parser.add_argument('--text_adapt_until', type=int, default=0, help='文本适配器层数')
    parser.add_argument('--enable_fsm', type=bool, default=True, help='启用特征分离模块')
    parser.add_argument('--device', type=str, default='cuda:2', help='测试设备')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    logger.info(f"使用设备: {device}")
    
    # 创建模型
    model = ModifiedCLIP(
        backbone=args.backbone,
        prompt_length=args.prompt_length,
        text_adapt_until=args.text_adapt_until,
        enable_fsm=args.enable_fsm,
        classnames=args.class_names
    )
    
    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # 获取当前模型的可训练参数
    trainable_state_dict = {
        k: v for k, v in model.state_dict().items()
        if k in dict(model.named_parameters()) and model.get_parameter(k).requires_grad
    }
    
    # 更新可训练参数
    trainable_state_dict.update(checkpoint['model_state_dict'])
    
    # 加载更新后的参数
    model.load_state_dict(trainable_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    logger.info(f"成功加载模型检查点: {args.checkpoint}")
    
    # 创建测试数据集
    logger.info("创建测试数据集...")
    test_datasets = create_test_datasets(
        data_dir=args.test_data_dir,
        img_size=args.img_size
    )
    
    if not test_datasets:
        logger.error("没有找到有效的测试数据集")
        return
    
    # 从checkpoint路径创建保存目录
    checkpoint_path = Path(args.checkpoint)
    save_dir = checkpoint_path.parent / ('test_results' + "_" + args.checkpoint.split('/')[-1].split('.')[0])
    
    # 创建测试器
    tester = Tester(
        model=model,
        test_datasets=test_datasets,
        device=device,
        save_dir=save_dir
    )
    
    # 开始测试
    tester.test_all_datasets()

if __name__ == "__main__":
    main()

# 使用示例脚本
"""
python test.py \
    --test_data_dir ./data \
    --checkpoint ./checkpoints/best_model.pth \
    --enable_fsm \
    --device cuda:2
""" 