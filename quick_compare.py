"""
빠른 비교 실험 - Baseline vs Phase 1 Improved
"""
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging

# 로컬 모듈
from config import get_baseline_config, get_improved_config
from main_v2 import run_experiment

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compare_configurations():
    """Baseline과 Improved 설정 비교"""

    logger.info("=" * 80)
    logger.info("Quick Comparison: Baseline vs Phase 1 Improved")
    logger.info("=" * 80)

    configurations = [
        ('baseline', get_baseline_config()),
        ('improved', get_improved_config())
    ]

    results = []

    for name, config in configurations:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running: {name.upper()}")
        logger.info(f"{'='*80}")
        logger.info(f"  Window Size: {config.window_size}")
        logger.info(f"  Hidden States: {config.n_states}")
        logger.info(f"  Threshold Percentile: {config.threshold_percentile}%")

        try:
            metrics, attack_rates = run_experiment(config)

            result = {
                'Configuration': name,
                'Window Size': config.window_size,
                'Hidden States': config.n_states,
                'Threshold %': config.threshold_percentile,
                'FPR (%)': metrics['FPR'] * 100,
                'TPR (%)': metrics['TPR'] * 100,
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1-Score': metrics['F1'],
                'Accuracy': metrics['Accuracy']
            }

            # 공격 유형별 탐지율 추가
            for attack_type, rate in attack_rates.items():
                result[f'DR_{attack_type} (%)'] = rate * 100

            results.append(result)

        except Exception as e:
            logger.error(f"Failed to run {name}: {str(e)}")

    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(results)

    # 콘솔 출력
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 80)
    print("\n" + results_df.to_string(index=False))

    # 개선도 계산
    if len(results_df) == 2:
        logger.info("\n" + "=" * 80)
        logger.info("IMPROVEMENTS (Improved vs Baseline)")
        logger.info("=" * 80)

        baseline = results_df.iloc[0]
        improved = results_df.iloc[1]

        improvements = {
            'FPR': improved['FPR (%)'] - baseline['FPR (%)'],
            'TPR': improved['TPR (%)'] - baseline['TPR (%)'],
            'F1-Score': improved['F1-Score'] - baseline['F1-Score'],
            'Accuracy': improved['Accuracy'] - baseline['Accuracy']
        }

        for metric, change in improvements.items():
            sign = '+' if change >= 0 else ''
            logger.info(f"  {metric:15s}: {sign}{change:+.2f}")

        # 공격 유형별 개선도
        logger.info("\n  Attack Detection Rate Improvements:")
        for col in results_df.columns:
            if col.startswith('DR_'):
                attack_type = col.replace('DR_', '').replace(' (%)', '')
                change = improved[col] - baseline[col]
                sign = '+' if change >= 0 else ''
                logger.info(f"    {attack_type:20s}: {sign}{change:+.2f}%")

    # CSV로 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/comparison_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"\nResults saved to: {results_file}")

    # 시각화
    plot_comparison(results_df, timestamp)

    logger.info("\n" + "=" * 80)
    logger.info("Comparison Completed!")
    logger.info("=" * 80)

    return results_df


def plot_comparison(results_df: pd.DataFrame, timestamp: str):
    """비교 결과 시각화"""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Baseline vs Phase 1 Improved Comparison', fontsize=16)

    configs = results_df['Configuration'].values
    n_configs = len(configs)
    x = np.arange(n_configs)
    width = 0.35

    # 1. FPR vs TPR
    ax = axes[0, 0]
    ax.bar(x - width/2, results_df['FPR (%)'], width, label='FPR', color='red', alpha=0.7)
    ax.bar(x + width/2, results_df['TPR (%)'], width, label='TPR', color='green', alpha=0.7)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('False Positive Rate vs True Positive Rate', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=5, color='red', linestyle='--', alpha=0.3, label='FPR Target')

    # 값 표시
    for i, (fpr, tpr) in enumerate(zip(results_df['FPR (%)'], results_df['TPR (%)'])):
        ax.text(i - width/2, fpr + 1, f'{fpr:.1f}%', ha='center', fontsize=10)
        ax.text(i + width/2, tpr + 1, f'{tpr:.1f}%', ha='center', fontsize=10)

    # 2. Precision, Recall, F1-Score
    ax = axes[0, 1]
    metrics = ['Precision', 'Recall', 'F1-Score']
    for i, metric in enumerate(metrics):
        values = results_df[metric] * 100
        ax.bar(x + (i - 1) * width/1.5, values, width/1.5, label=metric, alpha=0.7)

    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Precision, Recall, F1-Score', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)

    # 3. 공격 유형별 탐지율
    ax = axes[1, 0]
    attack_types = ['Adduser', 'Hydra_FTP', 'Hydra_SSH',
                   'Java_Meterpreter', 'Meterpreter', 'Web_Shell']

    x_attacks = np.arange(len(attack_types))
    for i, config in enumerate(configs):
        rates = []
        for attack in attack_types:
            col_name = f'DR_{attack} (%)'
            if col_name in results_df.columns:
                rate = results_df[results_df['Configuration'] == config][col_name].values[0]
                rates.append(rate)
            else:
                rates.append(0)

        ax.bar(x_attacks + i * width - width/2, rates, width, label=config, alpha=0.7)

    ax.set_ylabel('Detection Rate (%)', fontsize=12)
    ax.set_title('Detection Rate by Attack Type', fontsize=14)
    ax.set_xticks(x_attacks)
    ax.set_xticklabels(attack_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.3, label='Target')

    # 4. 전체 메트릭 요약
    ax = axes[1, 1]
    metric_names = ['FPR (%)', 'TPR (%)', 'F1-Score', 'Accuracy']

    # Baseline
    baseline_values = []
    for metric in metric_names:
        if metric in ['F1-Score', 'Accuracy']:
            baseline_values.append(results_df.iloc[0][metric] * 100)
        else:
            baseline_values.append(results_df.iloc[0][metric])

    # Improved
    improved_values = []
    for metric in metric_names:
        if metric in ['F1-Score', 'Accuracy']:
            improved_values.append(results_df.iloc[1][metric] * 100)
        else:
            improved_values.append(results_df.iloc[1][metric])

    x_metrics = np.arange(len(metric_names))
    ax.bar(x_metrics - width/2, baseline_values, width, label='Baseline', alpha=0.7)
    ax.bar(x_metrics + width/2, improved_values, width, label='Improved', alpha=0.7)

    # 값 표시
    for i, (b, imp) in enumerate(zip(baseline_values, improved_values)):
        ax.text(i - width/2, b + 1, f'{b:.1f}', ha='center', fontsize=10)
        ax.text(i + width/2, imp + 1, f'{imp:.1f}', ha='center', fontsize=10)

    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Overall Metrics Comparison', fontsize=14)
    ax.set_xticks(x_metrics)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    plot_path = f"results/comparison_plot_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Comparison plot saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    compare_configurations()
