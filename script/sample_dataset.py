from datasets import load_from_disk
import os
import numpy as np
from tqdm import tqdm
import shutil

def sample_dataset(source_dir, target_dir, sample_ratio=0.2, random_seed=42):
    """
    从原始数据集中随机采样指定比例的数据
    
    Args:
        source_dir: 原始数据集路径
        target_dir: 采样后数据集保存路径
        sample_ratio: 采样比例
        random_seed: 随机种子
    """
    print(f"正在从 {source_dir} 加载数据集...")
    dataset = load_from_disk(source_dir)
    
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 计算采样大小
    total_size = len(dataset)
    sample_size = int(total_size * sample_ratio)
    
    print(f"\n数据集大小: {total_size}")
    print(f"采样大小: {sample_size} ({sample_ratio*100}%)")
    
    # 随机采样
    indices = np.random.choice(total_size, sample_size, replace=False)
    sampled_dataset = dataset.select(indices)
    
    # 确保目标目录存在
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    
    # 如果目标目录已存在，先删除
    if os.path.exists(target_dir):
        print(f"\n目标目录 {target_dir} 已存在，正在删除...")
        shutil.rmtree(target_dir)
    
    print(f"\n正在保存采样数据集到 {target_dir}...")
    sampled_dataset.save_to_disk(target_dir)
    
    print(f"\n采样完成！")
    print(f"原始数据集大小: {total_size}")
    print(f"采样后数据集大小: {len(sampled_dataset)}")
    print(f"采样数据集已保存到: {target_dir}")
    
    return len(sampled_dataset)

def sample_all_datasets(source_base_dir, target_base_dir, sample_ratio=0.2, random_seed=42):
    """
    同时对训练集和测试集进行采样
    
    Args:
        source_base_dir: 原始数据集的基础目录
        target_base_dir: 目标数据集的基础目录
        sample_ratio: 采样比例
        random_seed: 随机种子
    """
    # 确保目标基础目录存在
    os.makedirs(target_base_dir, exist_ok=True)
    
    # 设置数据集路径
    datasets = {
        'train': 'train_dataset',
        'test': 'test_dataset'
    }
    
    print(f"开始数据采样 (采样比例: {sample_ratio*100}%)")
    print("=" * 50)
    
    results = {}
    for name, dirname in datasets.items():
        print(f"\n处理{name}数据集:")
        source_dir = os.path.join(source_base_dir, dirname)
        target_dir = os.path.join(target_base_dir, dirname)
        
        if not os.path.exists(source_dir):
            print(f"警告: {source_dir} 不存在，跳过处理")
            continue
            
        results[name] = sample_dataset(
            source_dir=source_dir,
            target_dir=target_dir,
            sample_ratio=sample_ratio,
            random_seed=random_seed
        )
    
    print("\n" + "=" * 50)
    print("采样总结:")
    for name, size in results.items():
        print(f"{name}数据集采样后大小: {size}")
    print("=" * 50)

if __name__ == "__main__":
    ratio=0.1
    # 设置路径
    source_base_dir = "/hy-tmp/data/origin"
    target_base_dir = f"/hy-tmp/data/sample_{ratio}"
    
    # 执行采样
    sample_all_datasets(
        source_base_dir=source_base_dir,
        target_base_dir=target_base_dir,
        sample_ratio=ratio
    ) 