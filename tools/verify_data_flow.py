#!/usr/bin/env python3
"""
数据流验证脚本
验证从训练脚本 -> fix_throughput.py -> analyze_results.py 的完整数据流
"""
import json
import os

def check_json_structure(filepath):
    """检查JSON文件结构是否符合预期"""
    print(f"\n检查文件: {filepath}")
    print("-" * 60)
    
    if not os.path.exists(filepath):
        print("❌ 文件不存在")
        return False
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # 检查必需字段
    required_top_fields = ['method', 'epochs', 'summary']
    required_summary_fields = ['avg_train_throughput', 'avg_train_time_per_epoch']
    required_epoch_fields = ['epoch', 'train_throughput', 'train_time']
    
    # 检查顶层字段
    missing_fields = []
    for field in required_top_fields:
        if field not in data:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"❌ 缺少顶层字段: {', '.join(missing_fields)}")
        return False
    else:
        print(f"✓ 顶层字段完整: {', '.join(required_top_fields)}")
    
    # 检查 summary 字段
    summary = data['summary']
    missing_summary = []
    for field in required_summary_fields:
        if field not in summary:
            missing_summary.append(field)
    
    if missing_summary:
        print(f"❌ summary 缺少字段: {', '.join(missing_summary)}")
        return False
    else:
        print(f"✓ summary 字段完整")
        print(f"  - avg_train_throughput: {summary['avg_train_throughput']:.2f} img/s")
        print(f"  - avg_train_time_per_epoch: {summary['avg_train_time_per_epoch']:.2f} s")
    
    # 检查 epochs 数组
    if not data['epochs'] or len(data['epochs']) == 0:
        print("❌ epochs 数组为空")
        return False
    
    print(f"✓ epochs 数组包含 {len(data['epochs'])} 个epoch")
    
    # 检查第一个 epoch 的字段
    epoch1 = data['epochs'][0]
    missing_epoch = []
    for field in required_epoch_fields:
        if field not in epoch1:
            missing_epoch.append(field)
    
    if missing_epoch:
        print(f"❌ epoch 缺少字段: {', '.join(missing_epoch)}")
        return False
    else:
        print(f"✓ epoch 字段完整")
        print(f"  - train_throughput: {epoch1['train_throughput']:.2f} img/s")
    
    # 数值合理性检查
    throughput = summary['avg_train_throughput']
    
    # 判断是否已修正（修正前 < 400, 修正后 > 400）
    if throughput < 400:
        print(f"⚠️  吞吐量偏低 ({throughput:.2f} img/s)")
        print(f"    可能原因:")
        print(f"    1. 尚未运行 fix_throughput.py 修正")
        print(f"    2. 训练过程存在性能问题")
        return "需要修正"
    elif throughput > 5000:
        print(f"⚠️  吞吐量过高 ({throughput:.2f} img/s)")
        print(f"    可能已重复修正，请检查")
        return "可能重复修正"
    else:
        print(f"✓ 吞吐量数值合理 ({throughput:.2f} img/s)")
        return True

def main():
    print("=" * 80)
    print("数据流完整性验证")
    print("=" * 80)
    print("\n此脚本验证以下数据流:")
    print("  训练脚本 -> JSON文件 -> fix_throughput.py -> analyze_results.py")
    print()
    
    result_files = [
        ('results/results_baseline_ddp.json', 'Baseline DDP'),
        ('results/results_all_reduce.json', 'Manual All-Reduce'),
        ('results/results_ps.json', 'Parameter Server'),
    ]
    
    results = {}
    
    for filepath, name in result_files:
        result = check_json_structure(filepath)
        results[name] = result
    
    # 总结
    print("\n" + "=" * 80)
    print("验证总结")
    print("=" * 80)
    
    all_ok = True
    needs_fix = []
    
    for name, result in results.items():
        if result is True:
            print(f"✅ {name}: 数据完整且已修正")
        elif result == "需要修正":
            print(f"⚠️  {name}: 需要运行 fix_throughput.py")
            needs_fix.append(name)
            all_ok = False
        elif result == "可能重复修正":
            print(f"⚠️  {name}: 可能重复修正，请检查")
            all_ok = False
        else:
            print(f"❌ {name}: 数据不完整或有错误")
            all_ok = False
    
    if all_ok:
        print("\n🎉 所有数据文件验证通过！")
        print("\n下一步:")
        print("  python scripts/analysis/analyze_results.py")
    elif needs_fix:
        print(f"\n⚠️  {len(needs_fix)} 个文件需要修正吞吐量")
        print("\n下一步:")
        print("  python fix_throughput.py")
        print("  python scripts/analysis/analyze_results.py")
    else:
        print("\n❌ 存在数据问题，请检查训练脚本输出")
    
    print("=" * 80)

if __name__ == '__main__':
    main()
