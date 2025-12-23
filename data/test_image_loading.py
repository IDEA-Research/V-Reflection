#!/usr/bin/env python3
"""
测试所有子数据集的图片路径映射和加载是否正常
"""

import json
import os
import sys
from collections import defaultdict

# 添加项目路径到sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.dataset.data_utils import map_image_path
except ImportError as e:
    print(f"导入错误: {e}")
    print("尝试直接定义map_image_path函数...")
    
    # 如果导入失败，直接定义函数
    def map_image_path(image_path, image_folder, dataset_name=None):
        """Map image path from JSON format to actual file system path."""
        # If already an absolute path and exists, return as is
        if os.path.isabs(image_path) and os.path.exists(image_path):
            return image_path
        
        # If it's a URL, return as is
        if image_path.startswith("http"):
            return image_path
        
        # Dataset path mappings based on Visual_cot structure
        dataset_path_mappings = {
            'cub': 'cub/CUB_200_2011/images',
            'docvqa': 'docvqa',
            'flickr30k': 'flickr30k/flickr30k-images',
            'gqa': 'gqa/images',
            'infographicsvqa': 'infographicsvqa',
            'openimages': 'openimages',
            'textcap': 'textvqa/train_images',
            'textvqa': 'textvqa/train_images',
            'v7w': 'visual7w/images',
            'vsr': 'vsr/images',
            'sroie': 'sroie',
            'dude': 'dude',
        }
        
        # Remove viscot/ prefix if present
        if image_path.startswith('viscot/'):
            image_path = image_path[7:]  # Remove 'viscot/'
        
        # Extract dataset name from path if not provided
        if dataset_name is None:
            path_parts = image_path.split('/', 1)
            if len(path_parts) > 1:
                dataset_name = path_parts[0]
                remaining_path = path_parts[1]
            else:
                remaining_path = image_path
        else:
            # Remove dataset name from path if it's at the start
            if image_path.startswith(f'{dataset_name}/'):
                remaining_path = image_path[len(dataset_name)+1:]
            else:
                remaining_path = image_path
        
        # Get mapped dataset path
        mapped_dataset_path = dataset_path_mappings.get(dataset_name, dataset_name)
        
        # Special handling for cub dataset - preserve subdirectory structure
        if dataset_name == 'cub':
            full_path = os.path.join(image_folder, mapped_dataset_path, remaining_path)
        elif dataset_name == 'openimages':
            # For openimages, files are in subdirectories like train_0, train_1, etc.
            filename = os.path.basename(remaining_path)
            base_path = os.path.join(image_folder, mapped_dataset_path)
            
            # First try direct path
            full_path = os.path.join(base_path, filename)
            if os.path.exists(full_path):
                return full_path
            
            # Search in subdirectories
            if os.path.exists(base_path):
                for subdir in os.listdir(base_path):
                    subdir_path = os.path.join(base_path, subdir)
                    if os.path.isdir(subdir_path):
                        candidate_path = os.path.join(subdir_path, filename)
                        if os.path.exists(candidate_path):
                            return candidate_path
            
            # Return the first candidate if not found (for error reporting)
            return full_path
        elif dataset_name in ['dude', 'sroie']:
            # For dude and sroie, check multiple possible locations
            filename = os.path.basename(remaining_path)
            
            # Try viscot subdirectory first (common location)
            viscot_path = os.path.join(image_folder, 'viscot', dataset_name, filename)
            if os.path.exists(viscot_path):
                return viscot_path
            
            # Try direct dataset path
            direct_path = os.path.join(image_folder, mapped_dataset_path, filename)
            if os.path.exists(direct_path):
                return direct_path
            
            # For dude, also check DUDE_train-val-test_binaries/images/train/
            if dataset_name == 'dude':
                dude_train_path = os.path.join(image_folder, 'dude', 'DUDE_train-val-test_binaries', 'images', 'train', filename)
                if os.path.exists(dude_train_path):
                    return dude_train_path
            
            # Return the viscot path as default (most likely location)
            return viscot_path
        else:
            # For other datasets, use only the filename
            filename = os.path.basename(remaining_path)
            full_path = os.path.join(image_folder, mapped_dataset_path, filename)
        
        return full_path

# 数据集路径映射（与data_utils.py中保持一致）
DATASET_PATH_MAPPINGS = {
    'cub': 'cub/CUB_200_2011/images',
    'docvqa': 'docvqa',
    'flickr30k': 'flickr30k/flickr30k-images',
    'gqa': 'gqa/images',
    'infographicsvqa': 'infographicsvqa',
    'openimages': 'openimages',
    'textcap': 'textvqa/train_images',
    'textvqa': 'textvqa/train_images',
    'v7w': 'visual7w/images',
    'vsr': 'vsr/images',
    'sroie': 'viscot/sroie',
    'dude': 'viscot/dude',
}

IMAGE_FOLDER = "/comp_robot/zhoujiazhou/Datasets/Visual_cot/images"

def test_image_path_mapping(json_file, max_samples_per_dataset=5):
    """
    测试JSON文件中的图片路径映射
    
    Args:
        json_file: JSON文件路径
        max_samples_per_dataset: 每个数据集测试的最大样本数
    """
    print(f"\n{'='*80}")
    print(f"测试文件: {json_file}")
    print(f"{'='*80}")
    
    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print(f"错误: JSON文件格式不正确，期望列表格式")
        return {}
    
    print(f"总记录数: {len(data)}")
    
    # 按数据集分组
    dataset_samples = defaultdict(list)
    for item in data:
        if isinstance(item, dict):
            dataset_name = item.get('dataset', '')
            image_path = item.get('image', '')
            if dataset_name and image_path:
                if len(dataset_samples[dataset_name]) < max_samples_per_dataset:
                    dataset_samples[dataset_name].append({
                        'dataset': dataset_name,
                        'image': image_path
                    })
    
    print(f"发现 {len(dataset_samples)} 个数据集: {sorted(dataset_samples.keys())}\n")
    
    # 测试每个数据集
    results = {}
    for dataset_name in sorted(dataset_samples.keys()):
        print(f"\n数据集: {dataset_name}")
        print("-" * 80)
        
        samples = dataset_samples[dataset_name]
        dataset_results = {
            'total_samples': len(samples),
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        for idx, sample in enumerate(samples, 1):
            image_path_json = sample['image']
            
            # 如果是列表，取第一个
            if isinstance(image_path_json, list):
                image_path_json = image_path_json[0] if len(image_path_json) > 0 else ""
            
            if not image_path_json:
                continue
            
            # 映射路径
            try:
                mapped_path = map_image_path(
                    image_path_json, 
                    IMAGE_FOLDER, 
                    dataset_name
                )
                
                # 检查文件是否存在
                file_exists = os.path.exists(mapped_path)
                
                if file_exists:
                    dataset_results['successful'] += 1
                    status = "✓"
                else:
                    dataset_results['failed'] += 1
                    status = "✗"
                    dataset_results['errors'].append({
                        'sample_idx': idx,
                        'json_path': image_path_json,
                        'mapped_path': mapped_path,
                        'error': 'File not found'
                    })
                
                # 打印前3个样本的详细信息
                if idx <= 3:
                    print(f"  样本 {idx}:")
                    print(f"    JSON路径: {image_path_json}")
                    print(f"    映射路径: {mapped_path}")
                    print(f"    文件存在: {status}")
                    if not file_exists:
                        print(f"    ⚠️  警告: 文件不存在")
                
            except Exception as e:
                dataset_results['failed'] += 1
                status = "✗"
                error_msg = f"Error: {str(e)}"
                dataset_results['errors'].append({
                    'sample_idx': idx,
                    'json_path': image_path_json,
                    'mapped_path': 'N/A',
                    'error': error_msg
                })
                print(f"  样本 {idx}: {status} - {error_msg}")
        
        # 打印汇总
        success_rate = (dataset_results['successful'] / dataset_results['total_samples'] * 100) if dataset_results['total_samples'] > 0 else 0
        print(f"\n  汇总:")
        print(f"    总样本数: {dataset_results['total_samples']}")
        print(f"    成功: {dataset_results['successful']} ({success_rate:.1f}%)")
        print(f"    失败: {dataset_results['failed']}")
        
        results[dataset_name] = dataset_results
    
    return results


def print_summary(all_results):
    """打印所有数据集的汇总信息"""
    print(f"\n\n{'='*80}")
    print("总体测试汇总")
    print(f"{'='*80}\n")
    
    total_datasets = len(all_results)
    total_samples = sum(r['total_samples'] for r in all_results.values())
    total_successful = sum(r['successful'] for r in all_results.values())
    total_failed = sum(r['failed'] for r in all_results.values())
    
    print(f"数据集总数: {total_datasets}")
    print(f"测试样本总数: {total_samples}")
    print(f"成功加载: {total_successful} ({total_successful/total_samples*100:.1f}%)" if total_samples > 0 else "成功加载: 0")
    print(f"失败: {total_failed} ({total_failed/total_samples*100:.1f}%)" if total_samples > 0 else "失败: 0")
    
    print(f"\n各数据集详细结果:")
    print(f"{'数据集名称':<20} {'样本数':<10} {'成功':<10} {'失败':<10} {'成功率':<10}")
    print("-" * 80)
    
    for dataset_name in sorted(all_results.keys()):
        r = all_results[dataset_name]
        success_rate = (r['successful'] / r['total_samples'] * 100) if r['total_samples'] > 0 else 0
        print(f"{dataset_name:<20} {r['total_samples']:<10} {r['successful']:<10} {r['failed']:<10} {success_rate:>6.1f}%")
    
    # 打印失败案例
    print(f"\n\n失败案例详情:")
    print("=" * 80)
    has_failures = False
    for dataset_name in sorted(all_results.keys()):
        r = all_results[dataset_name]
        if r['errors']:
            has_failures = True
            print(f"\n数据集: {dataset_name}")
            for error in r['errors'][:5]:  # 只显示前5个错误
                print(f"  样本 {error['sample_idx']}:")
                print(f"    JSON路径: {error['json_path']}")
                print(f"    映射路径: {error['mapped_path']}")
                print(f"    错误: {error['error']}")
            if len(r['errors']) > 5:
                print(f"  ... 还有 {len(r['errors']) - 5} 个错误未显示")
    
    if not has_failures:
        print("✓ 所有测试样本都成功加载！")


if __name__ == "__main__":
    print("=" * 80)
    print("图片路径映射和加载测试")
    print("=" * 80)
    
    # 测试两个JSON文件
    json_files = [
        "data/viscot_sroie_dude_lvr_formatted.json",
        "data/viscot_363k_lvr_formatted.json"
    ]
    
    all_results = {}
    
    for json_file in json_files:
        if os.path.exists(json_file):
            results = test_image_path_mapping(json_file, max_samples_per_dataset=10)
            # 合并结果（如果同一个数据集出现在多个文件中）
            for dataset_name, result in results.items():
                if dataset_name in all_results:
                    # 合并统计
                    all_results[dataset_name]['total_samples'] += result['total_samples']
                    all_results[dataset_name]['successful'] += result['successful']
                    all_results[dataset_name]['failed'] += result['failed']
                    all_results[dataset_name]['errors'].extend(result['errors'])
                else:
                    all_results[dataset_name] = result
        else:
            print(f"\n警告: 文件不存在: {json_file}")
    
    # 打印汇总
    print_summary(all_results)
    
    print(f"\n{'='*80}")
    print("测试完成")
    print(f"{'='*80}")
