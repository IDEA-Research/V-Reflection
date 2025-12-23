#!/usr/bin/env python3
"""
读取不同实验批次的checkpoints文件夹下的steps004.json，
计算accuracy并绘制accuracy随训练step变化的图表
"""

import json
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple
import glob


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


def calculate_accuracy(data: List[Dict]) -> float:
    """
    计算总体accuracy
    
    Args:
        data: JSON 数据列表
        
    Returns:
        总体accuracy (百分比)
    """
    # 跳过已经是统计结果的项
    filtered_data = []
    for item in data:
        if 'accuracy_by_category' in item or 'overall_accuracy' in item:
            continue
        filtered_data.append(item)
    
    if len(filtered_data) == 0:
        return 0.0
    
    total = 0
    correct = 0
    
    for item in filtered_data:
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
        total += 1
        if extracted_answer == label:
            correct += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0.0
    return accuracy


def extract_step_from_folder(folder_name: str) -> int:
    """
    从文件夹名称中提取checkpoint step
    例如: ...checkpoint-300 -> 300
    """
    match = re.search(r'checkpoint-(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return 0


def find_checkpoint_folders(base_dir: str, experiment_pattern: str) -> List[Tuple[int, str]]:
    """
    查找指定实验批次的所有checkpoint文件夹
    
    Args:
        base_dir: 基础目录路径
        experiment_pattern: 实验批次名称模式
        
    Returns:
        [(step, folder_path), ...] 按step排序的列表
    """
    pattern = os.path.join(base_dir, f"*{experiment_pattern}*")
    folders = glob.glob(pattern)
    
    results = []
    for folder in folders:
        step = extract_step_from_folder(folder)
        if step > 0:
            json_file = os.path.join(folder, 'steps004.json')
            if os.path.exists(json_file):
                results.append((step, json_file))
    
    # 按step排序
    results.sort(key=lambda x: x[0])
    return results


def process_experiment(base_dir: str, experiment_pattern: str, experiment_name: str) -> Tuple[List[int], List[float]]:
    """
    处理一个实验批次，计算所有checkpoints的accuracy
    
    Args:
        base_dir: 基础目录路径
        experiment_pattern: 实验批次名称模式
        experiment_name: 实验批次显示名称
        
    Returns:
        (steps, accuracies) 两个列表
    """
    print(f"\n处理实验批次: {experiment_name}")
    print(f"模式: {experiment_pattern}")
    
    checkpoint_files = find_checkpoint_folders(base_dir, experiment_pattern)
    
    if len(checkpoint_files) == 0:
        print(f"警告: 未找到匹配的checkpoint文件夹")
        return [], []
    
    steps = []
    accuracies = []
    
    for step, json_file in checkpoint_files:
        print(f"  处理 checkpoint-{step}: {json_file}")
        
        try:
            # 读取JSON文件
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 计算accuracy
            accuracy = calculate_accuracy(data)
            steps.append(step)
            accuracies.append(accuracy)
            
            print(f"    Step {step}: Accuracy = {accuracy:.2f}%")
            
        except Exception as e:
            print(f"    错误: 处理文件 {json_file} 时出错: {e}")
            continue
    
    return steps, accuracies


def plot_accuracy(steps: List[int], accuracies: List[float], experiment_name: str, output_file: str, baseline_accuracy: float = None):
    """
    绘制accuracy随step变化的图表
    
    Args:
        steps: 训练step列表
        accuracies: accuracy列表
        experiment_name: 实验名称
        output_file: 输出文件路径
        baseline_accuracy: baseline的accuracy值（可选）
    """
    plt.figure(figsize=(12, 7))
    
    # 绘制实验数据
    plt.plot(steps, accuracies, marker='o', linewidth=2, markersize=8, label=experiment_name, color='blue')
    
    # 添加数值标签
    for step, acc in zip(steps, accuracies):
        plt.annotate(f'{acc:.2f}%', (step, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # 绘制baseline线
    if baseline_accuracy is not None:
        plt.axhline(y=baseline_accuracy, color='red', linestyle='--', linewidth=2, 
                   label=f'Baseline (Qwen2.5-VL-7B): {baseline_accuracy:.2f}%')
        # 在右侧添加baseline标签
        max_step = max(steps) if steps else 2100
        plt.text(max_step, baseline_accuracy, f' {baseline_accuracy:.2f}%', 
                verticalalignment='center', fontsize=9, color='red')
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Accuracy vs Training Step: {experiment_name}', fontsize=14, fontweight='bold')
    
    # 设置x轴刻度：以300为起点，每300标一次
    if steps:
        min_step = min(steps)
        max_step = max(steps)
        # 从300开始，每300一个刻度
        start_step = 300
        # 确保包含最大step，向上取整到最近的300
        end_step = ((max_step // 300) + 1) * 300
        x_ticks = list(range(start_step, end_step + 1, 300))
        plt.xticks(x_ticks, fontsize=10)
        # 设置x轴范围，稍微扩展一点以便显示所有点
        plt.xlim(min(300, min_step) - 50, max_step + 100)
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  图表已保存到: {output_file}")
    plt.close()


def load_baseline_accuracy(base_dir: str) -> float:
    """
    加载baseline的accuracy
    
    Args:
        base_dir: 基础目录路径
        
    Returns:
        baseline的accuracy值
    """
    baseline_file = os.path.join(base_dir, 'Qwen_Qwen2.5-VL-7B-Instruct', 'steps004.json')
    
    if not os.path.exists(baseline_file):
        print(f"警告: Baseline文件不存在: {baseline_file}")
        return None
    
    try:
        print(f"\n读取Baseline数据: {baseline_file}")
        with open(baseline_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        baseline_accuracy = calculate_accuracy(data)
        print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")
        return baseline_accuracy
    except Exception as e:
        print(f"错误: 读取baseline文件时出错: {e}")
        return None


def main():
    """
    主函数
    """
    # 基础目录
    base_dir = "/comp_robot/zhoujiazhou/projects/Active-Coconut/evaluation/results/blink/decoding_by_steps"
    
    # 三个实验批次
    experiments = [
        {
            'pattern': 'Stage1_steps1600_b4_mseLVR0.1',
            'name': 'Stage1_steps1600_b4_mseLVR0.1'
        },
        {
            'pattern': 'Stage1_steps2500_b1_mseLVR0.1',
            'name': 'Stage1_steps2500_b1_mseLVR0.1'
        },
        {
            'pattern': 'Stage1_steps2500_b4_mseLVR0.01',
            'name': 'Stage1_steps2500_b4_mseLVR0.01'
        }
    ]
    
    # 输出目录
    output_dir = "/comp_robot/zhoujiazhou/projects/Active-Coconut/evaluation/results/blink/accuracy_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载baseline accuracy
    baseline_accuracy = load_baseline_accuracy(base_dir)
    
    # 处理每个实验批次
    for exp in experiments:
        steps, accuracies = process_experiment(base_dir, exp['pattern'], exp['name'])
        
        if len(steps) > 0 and len(accuracies) > 0:
            # 绘制图表（包含baseline）
            output_file = os.path.join(output_dir, f"{exp['name']}_accuracy.png")
            plot_accuracy(steps, accuracies, exp['name'], output_file, baseline_accuracy)
            
            # 打印汇总信息
            print(f"\n汇总 - {exp['name']}:")
            print(f"  总checkpoints数: {len(steps)}")
            print(f"  Step范围: {min(steps)} - {max(steps)}")
            print(f"  Accuracy范围: {min(accuracies):.2f}% - {max(accuracies):.2f}%")
            print(f"  最终Accuracy: {accuracies[-1]:.2f}%")
            if baseline_accuracy is not None:
                print(f"  Baseline Accuracy: {baseline_accuracy:.2f}%")
                print(f"  相对Baseline提升: {accuracies[-1] - baseline_accuracy:.2f}%")
        else:
            print(f"\n警告: {exp['name']} 没有有效数据")
    
    print(f"\n所有图表已保存到: {output_dir}")


if __name__ == "__main__":
    main()

