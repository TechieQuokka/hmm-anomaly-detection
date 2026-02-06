"""
HMM 파라미터 그리드 서치
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Tuple
import json

# 로컬 모듈
sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_loader import ADFADataLoader
from hmm_model import AnomalyDetectorHMM
from evaluator import AnomalyDetectionEvaluator
from config import ExperimentConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GridSearchHMM:
    """HMM 파라미터 그리드 서치"""

    def __init__(self, data_dir: str = 'adfa-ld/ADFA-LD'):
        """
        Args:
            data_dir: 데이터 디렉토리
        """
        self.data_dir = data_dir
        self.results = []

    def run_grid_search(self, param_grid: Dict[str, List]) -> pd.DataFrame:
        """
        그리드 서치 실행

        Args:
            param_grid: 파라미터 그리드
                예: {
                    'window_size': [300, 500, 700],
                    'n_states': [10, 15, 20],
                    'threshold_percentile': [5.0, 10.0, 15.0, 20.0]
                }

        Returns:
            결과 DataFrame
        """
        logger.info("=" * 80)
        logger.info("Starting Grid Search")
        logger.info("=" * 80)
        logger.info(f"\nParameter Grid:")
        for param, values in param_grid.items():
            logger.info(f"  {param}: {values}")

        # 총 실험 횟수 계산
        total_experiments = 1
        for values in param_grid.values():
            total_experiments *= len(values)
        logger.info(f"\nTotal experiments: {total_experiments}")

        # 그리드 생성
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        # 실험 실행
        experiment_idx = 0
        for param_combination in product(*param_values):
            experiment_idx += 1
            params = dict(zip(param_names, param_combination))

            logger.info("\n" + "-" * 80)
            logger.info(f"Experiment {experiment_idx}/{total_experiments}")
            logger.info(f"Parameters: {params}")
            logger.info("-" * 80)

            try:
                # 실험 실행
                metrics = self._run_single_experiment(params)

                # 결과 저장
                result = {**params, **metrics}
                self.results.append(result)

                logger.info(f"Results: FPR={metrics['FPR']:.4f}, "
                          f"TPR={metrics['TPR']:.4f}, F1={metrics['F1']:.4f}")

            except Exception as e:
                logger.error(f"Experiment failed: {str(e)}")
                result = {**params, 'error': str(e)}
                self.results.append(result)

        # 결과를 DataFrame으로 변환
        results_df = pd.DataFrame(self.results)

        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/grid_search_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        logger.info(f"\n\nGrid search results saved to: {results_file}")

        return results_df

    def _run_single_experiment(self, params: Dict) -> Dict[str, float]:
        """
        단일 실험 실행

        Args:
            params: 파라미터 딕셔너리

        Returns:
            메트릭 딕셔너리
        """
        # 설정 생성
        config = ExperimentConfig(
            data_dir=self.data_dir,
            window_size=params.get('window_size', 500),
            n_states=params.get('n_states', 10),
            threshold_percentile=params.get('threshold_percentile', 5.0),
            train_ratio=params.get('train_ratio', 0.6),
            val_ratio=params.get('val_ratio', 0.2),
            test_ratio=params.get('test_ratio', 0.2),
            random_seed=params.get('random_seed', 42)
        )

        # 데이터 로더
        data_loader = ADFADataLoader(
            data_dir=config.data_dir,
            window_size=config.window_size
        )

        # 데이터 로드
        normal_sequences = data_loader.load_normal_data(data_loader.train_dir)
        attack_data = data_loader.load_attack_data()

        # 데이터 분할
        train_seqs, val_seqs, test_normal_seqs = data_loader.split_data(
            normal_sequences,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            random_seed=config.random_seed
        )

        # 시스템 호출 매핑
        syscall_mapping = data_loader.get_syscall_mapping(train_seqs)
        n_observations = len(syscall_mapping)

        train_seqs = data_loader.apply_mapping(train_seqs, syscall_mapping)
        val_seqs = data_loader.apply_mapping(val_seqs, syscall_mapping)
        test_normal_seqs = data_loader.apply_mapping(test_normal_seqs, syscall_mapping)

        for attack_type in attack_data.keys():
            attack_data[attack_type] = data_loader.apply_mapping(
                attack_data[attack_type], syscall_mapping
            )

        # HMM 학습
        detector = AnomalyDetectorHMM(
            n_states=config.n_states,
            n_observations=n_observations,
            random_state=config.random_seed
        )
        detector.fit(train_seqs)

        # Threshold 설정
        detector.set_threshold_percentile(val_seqs, percentile=config.threshold_percentile)

        # 평가
        test_normal_preds, _ = detector.predict_with_scores(test_normal_seqs)

        all_attack_preds = []
        attack_predictions = {}

        for attack_type, sequences in attack_data.items():
            preds, _ = detector.predict_with_scores(sequences)
            attack_predictions[attack_type] = preds
            all_attack_preds.extend(preds)

        # 메트릭 계산
        y_true = np.concatenate([
            np.zeros(len(test_normal_preds)),
            np.ones(len(all_attack_preds))
        ])

        y_pred = np.concatenate([
            test_normal_preds,
            np.array(all_attack_preds)  # 🐛 버그 수정: 리스트를 numpy array로 변환
        ])

        evaluator = AnomalyDetectionEvaluator()
        metrics = evaluator.compute_metrics(y_true, y_pred)
        attack_detection_rates = evaluator.evaluate_attack_types(attack_data, attack_predictions)

        # 공격 유형별 탐지율 추가
        for attack_type, rate in attack_detection_rates.items():
            metrics[f'DR_{attack_type}'] = rate

        return metrics

    def get_best_params(self, results_df: pd.DataFrame,
                        metric: str = 'F1',
                        fpr_constraint: float = 0.05) -> Tuple[Dict, float]:
        """
        최적 파라미터 찾기

        Args:
            results_df: 그리드 서치 결과 DataFrame
            metric: 최적화할 메트릭 ('F1', 'TPR', 'Accuracy')
            fpr_constraint: FPR 제약 조건 (default: 5%)

        Returns:
            (최적 파라미터, 메트릭 값)
        """
        # FPR 제약 조건 적용
        valid_results = results_df[results_df['FPR'] <= fpr_constraint]

        if len(valid_results) == 0:
            logger.warning(f"No results satisfy FPR <= {fpr_constraint}. "
                          f"Using all results.")
            valid_results = results_df

        # 최적 메트릭 찾기
        best_idx = valid_results[metric].idxmax()
        best_row = valid_results.loc[best_idx]

        # 파라미터 추출
        param_cols = ['window_size', 'n_states', 'threshold_percentile']
        best_params = {col: best_row[col] for col in param_cols if col in best_row}
        best_metric = best_row[metric]

        return best_params, best_metric

    def plot_results(self, results_df: pd.DataFrame, save_path: str = None):
        """
        그리드 서치 결과 시각화

        Args:
            results_df: 결과 DataFrame
            save_path: 저장 경로
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Grid Search Results', fontsize=16)

        # 1. FPR vs TPR (각 파라미터별)
        ax = axes[0, 0]
        for n_states in results_df['n_states'].unique():
            subset = results_df[results_df['n_states'] == n_states]
            ax.scatter(subset['FPR'], subset['TPR'],
                      label=f'n_states={n_states}', alpha=0.7, s=100)
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
        ax.set_title('FPR vs TPR by Hidden States', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0.05, color='red', linestyle='--', alpha=0.5, label='FPR=5%')

        # 2. F1-Score vs Window Size
        ax = axes[0, 1]
        for threshold in sorted(results_df['threshold_percentile'].unique()):
            subset = results_df[results_df['threshold_percentile'] == threshold]
            grouped = subset.groupby('window_size')['F1'].mean()
            ax.plot(grouped.index, grouped.values, marker='o',
                   label=f'threshold={threshold}%', linewidth=2)
        ax.set_xlabel('Window Size', fontsize=12)
        ax.set_ylabel('F1-Score', fontsize=12)
        ax.set_title('F1-Score vs Window Size', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Heatmap: n_states vs threshold_percentile (F1-Score)
        ax = axes[1, 0]
        pivot_data = results_df.pivot_table(
            values='F1',
            index='n_states',
            columns='threshold_percentile',
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
        ax.set_title('F1-Score Heatmap\n(n_states vs threshold_percentile)', fontsize=14)
        ax.set_xlabel('Threshold Percentile', fontsize=12)
        ax.set_ylabel('Hidden States', fontsize=12)

        # 4. 메트릭 비교 (Best configuration)
        ax = axes[1, 1]
        best_params, _ = self.get_best_params(results_df, metric='F1')
        best_result = results_df[
            (results_df['window_size'] == best_params['window_size']) &
            (results_df['n_states'] == best_params['n_states']) &
            (results_df['threshold_percentile'] == best_params['threshold_percentile'])
        ].iloc[0]

        metrics = ['FPR', 'TPR', 'Precision', 'Recall', 'F1', 'Accuracy']
        values = [best_result[m] * 100 for m in metrics]
        colors = ['red' if m == 'FPR' else 'green' for m in metrics]

        bars = ax.bar(metrics, values, color=colors, alpha=0.7)
        ax.axhline(y=5, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=85, color='green', linestyle='--', linewidth=1, alpha=0.5)

        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{value:.1f}%', ha='center', va='bottom', fontsize=10)

        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title(f'Best Configuration Metrics\n{best_params}', fontsize=14)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Results plot saved to: {save_path}")

        plt.close()


def main():
    """메인 함수"""
    logger.info("=" * 80)
    logger.info("HMM Parameter Grid Search")
    logger.info("=" * 80)

    # 그리드 서치 설정
    param_grid = {
        'window_size': [300, 500, 700],
        'n_states': [10, 15, 20],
        'threshold_percentile': [5.0, 10.0, 15.0, 20.0]
    }

    # 그리드 서치 실행
    searcher = GridSearchHMM(data_dir='adfa-ld/ADFA-LD')
    results_df = searcher.run_grid_search(param_grid)

    # 결과 분석
    logger.info("\n" + "=" * 80)
    logger.info("Grid Search Summary")
    logger.info("=" * 80)

    # 통계 요약
    logger.info("\nMetrics Summary:")
    logger.info(results_df[['FPR', 'TPR', 'F1', 'Accuracy']].describe())

    # 최적 파라미터 (F1 기준)
    best_params_f1, best_f1 = searcher.get_best_params(results_df, metric='F1')
    logger.info(f"\nBest Parameters (F1-Score):")
    logger.info(f"  {best_params_f1}")
    logger.info(f"  F1-Score: {best_f1:.4f}")

    # 최적 파라미터 (TPR 기준)
    best_params_tpr, best_tpr = searcher.get_best_params(results_df, metric='TPR')
    logger.info(f"\nBest Parameters (TPR/Detection Rate):")
    logger.info(f"  {best_params_tpr}")
    logger.info(f"  TPR: {best_tpr:.4f}")

    # 시각화
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"results/grid_search_plot_{timestamp}.png"
    searcher.plot_results(results_df, save_path=plot_path)

    logger.info("\n" + "=" * 80)
    logger.info("Grid Search Completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
