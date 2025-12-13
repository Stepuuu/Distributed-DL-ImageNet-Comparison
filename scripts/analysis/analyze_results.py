#!/usr/bin/env python3
"""
性能分析和可视化脚本
用于对比 Parameter Server, All-Reduce, 和 Baseline DDP 的性能
"""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class PerformanceAnalyzer:
    def __init__(self):
        self.results = {}
        self.methods = {
            "./results/results_baseline_ddp.json": "Baseline DDP",
            "./results/results_all_reduce.json": "Manual All-Reduce",
            "./results/results_ps.json": "Parameter Server",
        }
        self.colors = {
            "Baseline DDP": "#2E86AB",
            "Manual All-Reduce": "#A23B72",
            "Parameter Server": "#F18F01",
        }

    def load_results(self):
        """加载所有结果文件"""
        print("=" * 80)
        print("加载实验结果...")
        print("=" * 80)

        for filename, method_name in self.methods.items():
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    self.results[method_name] = json.load(f)
                print(f"✓ 已加载: {filename} -> {method_name}")
            else:
                print(f"✗ 未找到: {filename}")

        if not self.results:
            print("\n错误: 没有找到任何结果文件!")
            print("请先运行训练脚本生成结果文件。")
            return False

        print(f"\n成功加载 {len(self.results)} 个结果文件\n")
        return True

    def generate_comparison_table(self):
        """生成性能对比表格"""
        print("=" * 80)
        print("性能对比表格")
        print("=" * 80)

        # 表头
        header = (
            f"{'方法':<25} {'平均吞吐量':<15} {'平均训练时间':<15} {'最佳准确率':<12}"
        )
        print(header)
        print("-" * 80)

        table_data = []

        for method_name, data in self.results.items():
            summary = data.get("summary", {})

            throughput = summary.get("avg_train_throughput", 0)
            train_time = summary.get("avg_train_time_per_epoch", 0)

            # 获取最佳准确率
            if "best_val_accuracy" in summary:
                accuracy = summary["best_val_accuracy"]
            elif "final_train_accuracy" in summary:
                accuracy = summary["final_train_accuracy"]
            else:
                accuracy = 0

            row = f"{method_name:<25} {throughput:<15.2f} {train_time:<15.2f} {accuracy:<12.2f}%"
            print(row)

            table_data.append(
                {
                    "method": method_name,
                    "throughput": throughput,
                    "train_time": train_time,
                    "accuracy": accuracy,
                }
            )

        print("=" * 80)

        # 计算加速比
        if len(table_data) >= 2:
            print("\n加速比分析 (相对于 Baseline DDP):")
            print("-" * 80)

            baseline = None
            for item in table_data:
                if "Baseline" in item["method"]:
                    baseline = item
                    break

            if baseline:
                for item in table_data:
                    if item["method"] != baseline["method"]:
                        speedup = (
                            item["throughput"] / baseline["throughput"]
                            if baseline["throughput"] > 0
                            else 0
                        )
                        time_ratio = (
                            baseline["train_time"] / item["train_time"]
                            if item["train_time"] > 0
                            else 0
                        )
                        print(
                            f"{item['method']:<25} 吞吐量加速比: {speedup:.2f}x, 时间比率: {time_ratio:.2f}x"
                        )

        print("\n")
        return table_data

    def plot_throughput_comparison(self, table_data, output_dir="./results/plots"):
        """绘制吞吐量对比图"""
        os.makedirs(output_dir, exist_ok=True)

        methods = [item["method"] for item in table_data]
        throughputs = [item["throughput"] for item in table_data]
        colors = [self.colors.get(m, "#333333") for m in methods]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, throughputs, color=colors, alpha=0.8, edgecolor="black")

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        plt.xlabel("Training Method", fontsize=12, fontweight="bold")
        plt.ylabel("Throughput (images/sec)", fontsize=12, fontweight="bold")
        plt.title(
            "Throughput Comparison Across Methods", fontsize=14, fontweight="bold"
        )
        plt.xticks(rotation=15, ha="right")
        plt.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()

        output_path = os.path.join(output_dir, "throughput_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ 吞吐量对比图已保存: {output_path}")
        plt.close()

    def plot_training_time_comparison(self, table_data, output_dir="./results/plots"):
        """绘制训练时间对比图"""
        os.makedirs(output_dir, exist_ok=True)

        methods = [item["method"] for item in table_data]
        train_times = [item["train_time"] for item in table_data]
        colors = [self.colors.get(m, "#333333") for m in methods]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, train_times, color=colors, alpha=0.8, edgecolor="black")

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}s",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        plt.xlabel("Training Method", fontsize=12, fontweight="bold")
        plt.ylabel(
            "Average Training Time per Epoch (seconds)", fontsize=12, fontweight="bold"
        )
        plt.title(
            "Training Time Comparison Across Methods", fontsize=14, fontweight="bold"
        )
        plt.xticks(rotation=15, ha="right")
        plt.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()

        output_path = os.path.join(output_dir, "training_time_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ 训练时间对比图已保存: {output_path}")
        plt.close()

    def plot_loss_curves(self, output_dir="./plots"):
        """绘制Loss曲线对比"""
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12, 6))

        for method_name, data in self.results.items():
            epochs_data = data.get("epochs", [])
            if epochs_data:
                epochs = [e["epoch"] for e in epochs_data]
                losses = [e.get("train_loss", 0) for e in epochs_data]
                color = self.colors.get(method_name, "#333333")
                plt.plot(
                    epochs,
                    losses,
                    marker="o",
                    label=method_name,
                    color=color,
                    linewidth=2,
                    markersize=8,
                )

        plt.xlabel("Epoch", fontsize=12, fontweight="bold")
        plt.ylabel("Training Loss", fontsize=12, fontweight="bold")
        plt.title(
            "Training Loss Convergence Comparison", fontsize=14, fontweight="bold"
        )
        plt.legend(fontsize=11, loc="best")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()

        output_path = os.path.join(output_dir, "loss_curves.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Loss曲线图已保存: {output_path}")
        plt.close()

    def plot_accuracy_curves(self, output_dir="./results/plots"):
        """绘制准确率曲线对比图"""
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12, 6))

        for method_name, data in self.results.items():
            epochs_data = data.get("epochs", [])
            if epochs_data:
                epochs = [e["epoch"] for e in epochs_data]
                # 优先使用验证准确率，如果没有则使用训练准确率
                if "val_accuracy" in epochs_data[0]:
                    accuracies = [e.get("val_accuracy", 0) for e in epochs_data]
                    label = f"{method_name} (Val)"
                else:
                    accuracies = [e.get("train_accuracy", 0) for e in epochs_data]
                    label = f"{method_name} (Train)"

                color = self.colors.get(method_name, "#333333")
                plt.plot(
                    epochs,
                    accuracies,
                    marker="s",
                    label=label,
                    color=color,
                    linewidth=2,
                    markersize=8,
                )

        plt.xlabel("Epoch", fontsize=12, fontweight="bold")
        plt.ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
        plt.title("Accuracy Comparison Across Epochs", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11, loc="best")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()

        output_path = os.path.join(output_dir, "accuracy_curves.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ 准确率曲线图已保存: {output_path}")
        plt.close()

    def plot_communication_overhead(self, output_dir="./results/plots"):
        """绘制通信开销对比图 (仅PS和All-Reduce)"""
        os.makedirs(output_dir, exist_ok=True)

        comm_data = {}

        # 收集PS的通信时间
        if "Parameter Server" in self.results:
            ps_data = self.results["Parameter Server"]
            summary = ps_data.get("summary", {})
            pull_time = summary.get("avg_comm_time_pull", 0) * 1000  # 转换为ms
            push_time = summary.get("avg_comm_time_push", 0) * 1000
            comm_data["Parameter Server"] = {
                "pull": pull_time,
                "push": push_time,
                "total": pull_time + push_time,
            }

        # 收集All-Reduce的通信时间
        if "Manual All-Reduce" in self.results:
            ar_data = self.results["Manual All-Reduce"]
            epochs_data = ar_data.get("epochs", [])
            if epochs_data:
                avg_comm = (
                    np.mean([e.get("avg_comm_time", 0) for e in epochs_data]) * 1000
                )
                comm_data["Manual All-Reduce"] = {
                    "all_reduce": avg_comm,
                    "total": avg_comm,
                }

        if not comm_data:
            print("⚠ 没有足够的通信时间数据用于绘图")
            return

        # 绘制堆叠柱状图
        fig, ax = plt.subplots(figsize=(10, 6))

        methods = list(comm_data.keys())
        x_pos = np.arange(len(methods))
        width = 0.5

        # PS: 分为pull和push
        if "Parameter Server" in comm_data:
            idx = methods.index("Parameter Server")
            ps = comm_data["Parameter Server"]
            ax.bar(
                x_pos[idx],
                ps["pull"],
                width,
                label="PS Pull (Param Download)",
                color="#3498db",
                edgecolor="black",
            )
            ax.bar(
                x_pos[idx],
                ps["push"],
                width,
                bottom=ps["pull"],
                label="PS Push (Gradient Upload)",
                color="#e74c3c",
                edgecolor="black",
            )

        # All-Reduce: 单独一项
        if "Manual All-Reduce" in comm_data:
            idx = methods.index("Manual All-Reduce")
            ar = comm_data["Manual All-Reduce"]
            ax.bar(
                x_pos[idx],
                ar["all_reduce"],
                width,
                label="All-Reduce Sync",
                color="#9b59b6",
                edgecolor="black",
            )

        ax.set_xlabel("Method", fontsize=12, fontweight="bold")
        ax.set_ylabel("Communication Time (ms)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Average Communication Overhead per Batch", fontsize=14, fontweight="bold"
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        ax.legend(fontsize=10, loc="best")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        plt.tight_layout()
        output_path = os.path.join(output_dir, "communication_overhead.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ 通信开销对比图已保存: {output_path}")
        plt.close()

    def generate_report(self, output_file="./results/performance_report.txt"):
        """生成文本报告"""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("分布式训练性能对比报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            for method_name, data in self.results.items():
                f.write(f"\n{'=' * 80}\n")
                f.write(f"方法: {method_name}\n")
                f.write(f"{'=' * 80}\n")

                f.write("\n配置信息:\n")
                f.write(f"  - World Size: {data.get('world_size', 'N/A')}\n")
                f.write(f"  - Batch Size: {data.get('batch_size', 'N/A')}\n")
                f.write(f"  - 训练开始时间: {data.get('start_time', 'N/A')}\n")
                f.write(f"  - 训练结束时间: {data.get('end_time', 'N/A')}\n")

                summary = data.get("summary", {})
                f.write("\n性能摘要:\n")
                f.write(
                    f"  - 平均吞吐量: {summary.get('avg_train_throughput', 0):.2f} img/s\n"
                )
                f.write(
                    f"  - 平均训练时间/epoch: {summary.get('avg_train_time_per_epoch', 0):.2f} s\n"
                )

                if "best_val_accuracy" in summary:
                    f.write(
                        f"  - 最佳验证准确率: {summary['best_val_accuracy']:.2f}%\n"
                    )
                if "final_train_accuracy" in summary:
                    f.write(
                        f"  - 最终训练准确率: {summary['final_train_accuracy']:.2f}%\n"
                    )

                # PS特有指标
                if "avg_comm_time_pull" in summary:
                    f.write(
                        f"  - 平均参数拉取时间: {summary['avg_comm_time_pull'] * 1000:.2f} ms\n"
                    )
                    f.write(
                        f"  - 平均梯度推送时间: {summary['avg_comm_time_push'] * 1000:.2f} ms\n"
                    )

                # Epoch详情
                epochs_data = data.get("epochs", [])
                if epochs_data:
                    f.write("\nEpoch详情:\n")
                    for e in epochs_data:
                        f.write(
                            f"  Epoch {e['epoch']}: Loss={e.get('train_loss', 0):.4f}, "
                            f"Acc={e.get('train_accuracy', 0):.2f}%, "
                            f"Throughput={e.get('train_throughput', 0):.2f} img/s\n"
                        )

        print(f"✓ 详细报告已保存: {output_file}")

    def run_full_analysis(self):
        """运行完整分析流程"""
        if not self.load_results():
            return

        # 生成对比表格
        table_data = self.generate_comparison_table()

        # 绘制所有图表
        print("\n生成可视化图表...")
        print("-" * 80)
        self.plot_throughput_comparison(table_data)
        self.plot_training_time_comparison(table_data)
        self.plot_loss_curves()
        self.plot_accuracy_curves()
        self.plot_communication_overhead()

        # 生成文本报告
        print("\n生成详细报告...")
        print("-" * 80)
        self.generate_report()

        print("\n" + "=" * 80)
        print("分析完成! 所有结果已保存到:")
        print("  - 图表目录: ./plots/")
        print("  - 文本报告: performance_report.txt")
        print("=" * 80)


if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    analyzer.run_full_analysis()
