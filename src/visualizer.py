"""
결과 시각화 모듈
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultVisualizer:
    """결과 시각화"""

    @staticmethod
    def plot_likelihood_distribution(normal_scores: np.ndarray,
                                     attack_scores: np.ndarray,
                                     threshold: float,
                                     save_path: str = None):
        """
        정상/공격의 log-likelihood 분포 시각화

        Args:
            normal_scores: 정상 데이터의 log-likelihood
            attack_scores: 공격 데이터의 log-likelihood
            threshold: 설정된 threshold
            save_path: 저장 경로
        """
        plt.figure(figsize=(12, 6))

        # 히스토그램
        plt.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
        plt.hist(attack_scores, bins=50, alpha=0.6, label='Attack', color='red', density=True)

        # Threshold 선
        plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Threshold ({threshold:.2f})')

        plt.xlabel('Log-Likelihood', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Log-Likelihood Distribution (Normal vs Attack)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Likelihood distribution plot saved to {save_path}")

        plt.close()

    @staticmethod
    def plot_confusion_matrix(metrics: Dict[str, float],
                             save_path: str = None):
        """
        Confusion Matrix 시각화

        Args:
            metrics: 평가 메트릭
            save_path: 저장 경로
        """
        cm = np.array([[metrics['TN'], metrics['FP']],
                      [metrics['FN'], metrics['TP']]])

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap='Blues')

        # 축 레이블
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Attack'])
        ax.set_yticklabels(['Normal', 'Attack'])

        # 텍스트 추가
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, int(cm[i, j]),
                             ha="center", va="center", color="black", fontsize=20)

        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14)

        plt.colorbar(im, ax=ax)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")

        plt.close()

    @staticmethod
    def plot_attack_detection_rates(attack_rates: Dict[str, float],
                                    save_path: str = None):
        """
        공격 유형별 Detection Rate 시각화

        Args:
            attack_rates: {attack_type: detection_rate}
            save_path: 저장 경로
        """
        attack_types = list(attack_rates.keys())
        rates = [attack_rates[at] * 100 for at in attack_types]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(attack_types)), rates, color='steelblue', alpha=0.7)

        # 90% 기준선
        plt.axhline(y=90, color='green', linestyle='--', linewidth=1.5, label='90% Target')

        # 각 막대 위에 값 표시
        for i, (bar, rate) in enumerate(zip(bars, rates)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)

        plt.xticks(range(len(attack_types)), attack_types, rotation=45, ha='right')
        plt.ylabel('Detection Rate (%)', fontsize=12)
        plt.title('Detection Rate by Attack Type', fontsize=14)
        plt.ylim(0, 105)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attack detection rates plot saved to {save_path}")

        plt.close()

    @staticmethod
    def plot_metrics_summary(metrics: Dict[str, float],
                           save_path: str = None):
        """
        주요 메트릭 요약 시각화

        Args:
            metrics: 평가 메트릭
            save_path: 저장 경로
        """
        metric_names = ['FPR', 'TPR', 'Precision', 'Recall', 'F1', 'Accuracy']
        values = [metrics[name] * 100 for name in metric_names]

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['red' if name == 'FPR' else 'green' for name in metric_names]
        bars = ax.bar(metric_names, values, color=colors, alpha=0.7)

        # 목표선 (FPR < 5%, 나머지 > 85%)
        ax.axhline(y=5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='FPR Target (5%)')
        ax.axhline(y=85, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Performance Target (85%)')

        # 값 표시
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{value:.2f}%', ha='center', va='bottom', fontsize=10)

        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Performance Metrics Summary', fontsize=14)
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics summary plot saved to {save_path}")

        plt.close()

    @staticmethod
    def create_all_visualizations(normal_scores: np.ndarray,
                                 attack_scores: np.ndarray,
                                 threshold: float,
                                 metrics: Dict[str, float],
                                 attack_rates: Dict[str, float],
                                 output_dir: str = "results"):
        """
        모든 시각화 생성 및 저장

        Args:
            normal_scores: 정상 데이터 log-likelihood
            attack_scores: 공격 데이터 log-likelihood
            threshold: Threshold 값
            metrics: 평가 메트릭
            attack_rates: 공격 유형별 Detection Rate
            output_dir: 출력 디렉토리
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 1. Likelihood Distribution
        ResultVisualizer.plot_likelihood_distribution(
            normal_scores, attack_scores, threshold,
            save_path=output_path / "likelihood_distribution.png"
        )

        # 2. Confusion Matrix
        ResultVisualizer.plot_confusion_matrix(
            metrics,
            save_path=output_path / "confusion_matrix.png"
        )

        # 3. Attack Detection Rates
        ResultVisualizer.plot_attack_detection_rates(
            attack_rates,
            save_path=output_path / "attack_detection_rates.png"
        )

        # 4. Metrics Summary
        ResultVisualizer.plot_metrics_summary(
            metrics,
            save_path=output_path / "metrics_summary.png"
        )

        logger.info(f"All visualizations saved to {output_dir}/")
