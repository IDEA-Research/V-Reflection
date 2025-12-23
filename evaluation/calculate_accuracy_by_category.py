#!/usr/bin/env python3
"""
计算 steps004.json 文件中按 category 分类的 accuracy
"""

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


def extract_answer(prediction: str) -> str:
    """
    从 prediction 字符串中提取答案
    使用与 evaluation.py 中 accuracy_reward 函数相同的逻辑
    """
    given_answer = prediction.split('<answer>')[-1]
    given_answer = given_answer.split('</answer')[0].strip()
    if " " in given_answer:
        given_answer = given_answer.split(" ")[0]
    if len(given_answer) > 1:
        given_answer = given_answer[0]
    return given_answer


def calculate_accuracy_by_category(data: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    按 category 计算 accuracy
    
    Args:
        data: JSON 数据列表
        
    Returns:
        字典，包含每个 category 的统计信息
        {
            'category_name': {
                'total': int,
                'correct': int,
                'accuracy': float
            }
        }
    """
    # 按 category 分组统计
    category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    # 遍历所有数据
    for item in data:
        # 跳过已经是统计结果的项
        if 'accuracy_by_category' in item or 'overall_accuracy' in item:
            continue
            
        category = item.get('category', 'Unknown')
        prediction = item.get('prediction', [])
        label = item.get('label', '')
        
        # prediction 是一个列表，取第一个元素
        if isinstance(prediction, list) and len(prediction) > 0:
            prediction_str = prediction[0]
        elif isinstance(prediction, str):
            prediction_str = prediction
        else:
            prediction_str = ''
        
        # 提取答案
        extracted_answer = extract_answer(prediction_str)
        
        # 统计
        category_stats[category]['total'] += 1
        if extracted_answer == label:
            category_stats[category]['correct'] += 1
    
    # 计算 accuracy
    results = {}
    for category, stats in category_stats.items():
        total = stats['total']
        correct = stats['correct']
        accuracy = (correct / total * 100) if total > 0 else 0.0
        results[category] = {
            'total': total,
            'correct': correct,
            'accuracy': accuracy
        }
    
    return results


def print_results(results: Dict[str, Dict[str, float]], json_file_path: str):
    """
    打印结果
    """
    print("=" * 80)
    print(f"Results for: {json_file_path}")
    print("=" * 80)
    print(f"{'Category':<30} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 80)
    
    # 按 category 名称排序
    sorted_categories = sorted(results.keys())
    
    total_correct = 0
    total_samples = 0
    
    for category in sorted_categories:
        stats = results[category]
        print(f"{category:<30} {stats['correct']:<10} {stats['total']:<10} {stats['accuracy']:.2f}%")
        total_correct += stats['correct']
        total_samples += stats['total']
    
    # 打印总体统计
    print("-" * 80)
    overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
    print(f"{'Overall':<30} {total_correct:<10} {total_samples:<10} {overall_accuracy:.2f}%")
    print("=" * 80)


def main():
    """
    主函数
    支持命令行参数指定 JSON 文件路径，或使用默认路径
    """
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
    else:
        # 默认使用当前打开的文件路径
        json_file_path = "/comp_robot/zhoujiazhou/projects/Active-Coconut/evaluation/results/blink/decoding_by_steps/_comp_robot_zhoujiazhou_projects_Active-Coconut_stage1_checkpoints_7b_checkpoint-2500/steps004.json"
    
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"Error: File not found: {json_file_path}")
        sys.exit(1)
    
    # 读取 JSON 文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 计算 accuracy
    results = calculate_accuracy_by_category(data)
    
    # 计算总体 accuracy
    total_correct = sum(stats['correct'] for stats in results.values())
    total_samples = sum(stats['total'] for stats in results.values())
    overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
    
    # 打印结果
    print_results(results, json_file_path)
    
    # 创建统计结果对象
    accuracy_summary = {
        'accuracy_by_category': results,
        'overall_accuracy': overall_accuracy,
        'overall_correct': total_correct,
        'overall_total': total_samples
    }
    
    # 将结果添加到原始数据的最前面
    # 如果第一个元素已经是统计结果，先移除它
    if len(data) > 0 and ('accuracy_by_category' in data[0] or 'overall_accuracy' in data[0]):
        data = data[1:]
    
    # 将统计结果插入到最前面
    updated_data = [accuracy_summary] + data
    
    # 保存回原文件
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults added to the beginning of: {json_file_path}")


if __name__ == "__main__":
    main()

