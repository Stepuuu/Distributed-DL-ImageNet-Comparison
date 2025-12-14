#!/usr/bin/env python3
"""
修正吞吐量计算错误的脚本
旧代码只统计了单个GPU的吞吐量，需要乘以GPU数量(4)
"""
import json
import os

WORLD_SIZE = 4  # GPU数量

def fix_json_file(filepath):
    """修正JSON文件中的吞吐量数据"""
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return False
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # 备份原文件
    backup_path = filepath + '.backup'
    with open(backup_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"✅ 已备份原文件到: {backup_path}")
    
    # 修正吞吐量字段
    fixed_fields = []
    
    # 修正 epoch 级别的吞吐量
    if 'epochs' in data:
        for epoch in data['epochs']:
            if 'train_throughput' in epoch:
                old_val = epoch['train_throughput']
                epoch['train_throughput'] = old_val * WORLD_SIZE
                fixed_fields.append(f"Epoch {epoch['epoch']} train_throughput: {old_val:.1f} -> {epoch['train_throughput']:.1f}")
            
            if 'val_throughput' in epoch:
                old_val = epoch['val_throughput']
                epoch['val_throughput'] = old_val * WORLD_SIZE
                fixed_fields.append(f"Epoch {epoch['epoch']} val_throughput: {old_val:.1f} -> {epoch['val_throughput']:.1f}")
    
    # 修正 summary 中的平均吞吐量（关键！analyze_results.py会读取这个）
    if 'summary' in data:
        summary = data['summary']
        
        if 'avg_train_throughput' in summary:
            old_val = summary['avg_train_throughput']
            summary['avg_train_throughput'] = old_val * WORLD_SIZE
            fixed_fields.append(f"summary.avg_train_throughput: {old_val:.1f} -> {summary['avg_train_throughput']:.1f}")
        
        if 'avg_val_throughput' in summary:
            old_val = summary['avg_val_throughput']
            summary['avg_val_throughput'] = old_val * WORLD_SIZE
            fixed_fields.append(f"summary.avg_val_throughput: {old_val:.1f} -> {summary['avg_val_throughput']:.1f}")
    
    # 兼容旧字段名（如果存在）
    if 'average_train_throughput' in data:
        old_val = data['average_train_throughput']
        data['average_train_throughput'] = old_val * WORLD_SIZE
        fixed_fields.append(f"average_train_throughput: {old_val:.1f} -> {data['average_train_throughput']:.1f}")
    
    if 'average_val_throughput' in data:
        old_val = data['average_val_throughput']
        data['average_val_throughput'] = old_val * WORLD_SIZE
        fixed_fields.append(f"average_val_throughput: {old_val:.1f} -> {data['average_val_throughput']:.1f}")
    
    # 保存修正后的文件
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"✅ 已修正 {len(fixed_fields)} 个字段")
    for field in fixed_fields[:5]:  # 只显示前5个
        print(f"   {field}")
    if len(fixed_fields) > 5:
        print(f"   ... 还有 {len(fixed_fields) - 5} 个字段")
    
    return True

def verify_fix(filepath):
    """验证修正后的数据"""
    if not os.path.exists(filepath):
        return
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"\n验证: {filepath}")
    print("-" * 60)
    
    # 检查 summary 中的关键字段
    if 'summary' in data:
        summary = data['summary']
        throughput = summary.get('avg_train_throughput', 0)
        print(f"✓ summary.avg_train_throughput = {throughput:.2f} img/s")
        
        # 合理性检查（修正后应该在 400-5000 img/s 范围）
        if 400 < throughput < 5000:
            print(f"  ✓ 数值合理（预期范围: 400-5000 img/s）")
        else:
            print(f"  ⚠️  数值异常！请检查是否修正正确")
    
    # 检查第一个 epoch 的吞吐量
    if 'epochs' in data and len(data['epochs']) > 0:
        epoch1 = data['epochs'][0]
        train_tp = epoch1.get('train_throughput', 0)
        print(f"✓ Epoch 1 train_throughput = {train_tp:.2f} img/s")

if __name__ == '__main__':
    print("=" * 60)
    print("修正吞吐量计算错误")
    print("=" * 60)
    print(f"GPU数量: {WORLD_SIZE}")
    print(f"修正方法: 吞吐量 × {WORLD_SIZE}\n")
    
    result_files = [
        'results/results_baseline_ddp.json',
        'results/results_all_reduce.json',
        'results/results_ps.json'
    ]
    
    # 第一步：修正所有文件
    success_count = 0
    for filepath in result_files:
        print(f"\n处理文件: {filepath}")
        print("-" * 60)
        if fix_json_file(filepath):
            success_count += 1
            print(f"✅ {filepath} 修正完成")
        else:
            print(f"⚠️  {filepath} 跳过")
    
    # 第二步：验证修正结果
    if success_count > 0:
        print("\n" + "=" * 60)
        print("验证修正结果")
        print("=" * 60)
        
        for filepath in result_files:
            verify_fix(filepath)
    
    print("\n" + "=" * 60)
    print(f"处理完成！成功修正 {success_count} 个文件")
    print("原始文件已备份为 *.backup")
    print("\n下一步: 运行 analyze_results.py 生成图表")
    print("  python scripts/analysis/analyze_results.py")
    print("=" * 60)
