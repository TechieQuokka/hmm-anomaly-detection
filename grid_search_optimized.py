"""
HMM 파라미터 그리드 서치 (최적화 + 병렬화 버전)
데이터 로딩 캐싱으로 12배 속도 향상
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Tuple
from joblib import Parallel, delayed
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


def prepare_data_for_window_size(window_size: int, data_dir: str,
                                  train_ratio: float = 0.6,
                                  val_ratio: float = 0.2,
                                  test_ratio: float = 0.2,
                                  random_seed: int = 42) -> Dict:
    """
    특정 window_size에 대한 데이터 준비 (캐싱용)

    Args:
        window_size: 윈도우 크기
        data_dir: 데이터 디렉토리
        train_ratio, val_ratio, test_ratio: 데이터 분할 비율
        random_seed: 랜덤 시드

    Returns:
        준비된 데이터 딕셔너리
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Preparing data for window_size={window_size}")
    logger.info(f"{'='*80}")

    # 데이터 로더
    data_loader = ADFADataLoader(
        data_dir=data_dir,
        window_size=window_size
    )

    # 데이터 로드
    normal_sequences = data_loader.load_normal_data(data_loader.train_dir)
    attack_data = data_loader.load_attack_data()

    # 데이터 분할
    train_seqs, val_seqs, test_normal_seqs = data_loader.split_data(
        normal_sequences,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )

    # 시스템 호출 매핑
    syscall_mapping = data_loader.get_syscall_mapping(train_seqs)
    n_observations = len(syscall_mapping)

    # 매핑 적용
    train_seqs = data_loader.apply_mapping(train_seqs, syscall_mapping)
    val_seqs = data_loader.apply_mapping(val_seqs, syscall_mapping)
    test_normal_seqs = data_loader.apply_mapping(test_normal_seqs, syscall_mapping)

    for attack_type in attack_data.keys():
        attack_data[attack_type] = data_loader.apply_mapping(
            attack_data[attack_type], syscall_mapping
        )

    logger.info(f"✓ Data prepared: {len(train_seqs)} train, {len(val_seqs)} val, "
                f"{len(test_normal_seqs)} test, {n_observations} syscalls")

    return {
        'window_size': window_size,
        'train_seqs': train_seqs,
        'val_seqs': val_seqs,
        'test_normal_seqs': test_normal_seqs,
        'attack_data': attack_data,
        'n_observations': n_observations
    }


def run_single_experiment_cached(n_states: int, threshold_percentile: float,
                                  cached_data: Dict, random_seed: int = 42) -> Dict:
    """
    캐싱된 데이터로 단일 실험 실행

    Args:
        n_states: Hidden states 개수
        threshold_percentile: Threshold percentile
        cached_data: 캐싱된 데이터 딕셔너리
        random_seed: 랜덤 시드

    Returns:
        메트릭 딕셔너리
    """
    try:
        # 캐싱된 데이터 추출
        window_size = cached_data['window_size']
        train_seqs = cached_data['train_seqs']
        val_seqs = cached_data['val_seqs']
        test_normal_seqs = cached_data['test_normal_seqs']
        attack_data = cached_data['attack_data']
        n_observations = cached_data['n_observations']

        # HMM 학습
        detector = AnomalyDetectorHMM(
            n_states=n_states,
            n_observations=n_observations,
            random_state=random_seed
        )
        detector.fit(train_seqs)

        # Threshold 설정
        detector.set_threshold_percentile(val_seqs, percentile=threshold_percentile)

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
            np.array(all_attack_preds)
        ])

        evaluator = AnomalyDetectionEvaluator()
        metrics = evaluator.compute_metrics(y_true, y_pred)
        attack_detection_rates = evaluator.evaluate_attack_types(attack_data, attack_predictions)

        # 공격 유형별 탐지율 추가
        for attack_type, rate in attack_detection_rates.items():
            metrics[f'DR_{attack_type}'] = rate

        # 파라미터 추가
        metrics['window_size'] = window_size
        metrics['n_states'] = n_states
        metrics['threshold_percentile'] = threshold_percentile

        return metrics

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        return {
            'window_size': window_size,
            'n_states': n_states,
            'threshold_percentile': threshold_percentile,
            'error': str(e)
        }


class OptimizedGridSearchHMM:
    """최적화된 HMM 파라미터 그리드 서치"""

    def __init__(self, data_dir: str = 'adfa-ld/ADFA-LD', n_jobs: int = -1):
        """
        Args:
            data_dir: 데이터 디렉토리
            n_jobs: 병렬 작업 수 (-1: 모든 코어 사용)
        """
        self.data_dir = data_dir
        self.n_jobs = n_jobs
        self.results = []

    def run_grid_search(self, param_grid: Dict[str, List]) -> pd.DataFrame:
        """
        최적화된 그리드 서치 실행

        Args:
            param_grid: 파라미터 그리드

        Returns:
            결과 DataFrame
        """
        logger.info("=" * 80)
        logger.info("OPTIMIZED Grid Search (Data Caching + Parallel)")
        logger.info("=" * 80)
        logger.info(f"\nParameter Grid:")
        for param, values in param_grid.items():
            logger.info(f"  {param}: {values}")

        window_sizes = param_grid['window_size']
        n_states_list = param_grid['n_states']
        threshold_percentiles = param_grid['threshold_percentile']

        total_experiments = len(window_sizes) * len(n_states_list) * len(threshold_percentiles)
        logger.info(f"\nTotal experiments: {total_experiments}")
        logger.info(f"Data loading: {len(window_sizes)} times (instead of {total_experiments})")
        logger.info(f"Speedup: {total_experiments / len(window_sizes):.1f}x faster data loading!")
        logger.info(f"Parallel jobs: {self.n_jobs if self.n_jobs > 0 else 'ALL CORES'}")

        # Step 1: 각 window_size별로 데이터 준비 (캐싱)
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Preparing & Caching Data")
        logger.info("=" * 80)

        cached_datasets = {}
        for window_size in window_sizes:
            cached_datasets[window_size] = prepare_data_for_window_size(
                window_size=window_size,
                data_dir=self.data_dir,
                train_ratio=param_grid.get('train_ratio', [0.6])[0] if isinstance(param_grid.get('train_ratio', 0.6), list) else 0.6,
                val_ratio=param_grid.get('val_ratio', [0.2])[0] if isinstance(param_grid.get('val_ratio', 0.2), list) else 0.2,
                test_ratio=param_grid.get('test_ratio', [0.2])[0] if isinstance(param_grid.get('test_ratio', 0.2), list) else 0.2,
                random_seed=param_grid.get('random_seed', [42])[0] if isinstance(param_grid.get('random_seed', 42), list) else 42
            )

        # Step 2: 병렬 실험 실행
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Running Experiments (Parallel)")
        logger.info("=" * 80)

        # 실험 조합 생성
        experiment_params = []
        for window_size in window_sizes:
            for n_states in n_states_list:
                for threshold in threshold_percentiles:
                    experiment_params.append({
                        'window_size': window_size,
                        'n_states': n_states,
                        'threshold_percentile': threshold
                    })

        # 병렬 실행
        results_list = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(run_single_experiment_cached)(
                n_states=params['n_states'],
                threshold_percentile=params['threshold_percentile'],
                cached_data=cached_datasets[params['window_size']],
                random_seed=param_grid.get('random_seed', [42])[0] if isinstance(param_grid.get('random_seed', 42), list) else 42
            )
            for params in experiment_params
        )

        # 결과 저장
        self.results = results_list
        results_df = pd.DataFrame(self.results)

        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/grid_search_optimized_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        logger.info(f"\n\n✓ Grid search results saved to: {results_file}")

        return results_df

    def get_best_params(self, results_df: pd.DataFrame,
                        metric: str = 'F1',
                        fpr_constraint: float = 0.05) -> Tuple[Dict, float]:
        """최적 파라미터 찾기"""
        # 에러 결과 제거
        if 'error' in results_df.columns:
            results_df = results_df[results_df['error'].isna()]

        # FPR 제약 조건 적용
        valid_results = results_df[results_df['FPR'] <= fpr_constraint]

        if len(valid_results) == 0:
            logger.warning(f"No results satisfy FPR <= {fpr_constraint}. Using all results.")
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
        """그리드 서치 결과 시각화"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 에러 결과 제거
        if 'error' in results_df.columns:
            results_df = results_df[results_df['error'].isna()]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Optimized Grid Search Results', fontsize=16, fontweight='bold')

        # 1. FPR vs TPR
        ax = axes[0, 0]
        for n_states in sorted(results_df['n_states'].unique()):
            subset = results_df[results_df['n_states'] == n_states]
            ax.scatter(subset['FPR'], subset['TPR'],
                      label=f'states={n_states}', alpha=0.7, s=100)
        ax.axvline(x=0.05, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target FPR=5%')
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate (Detection Rate)', fontsize=12, fontweight='bold')
        ax.set_title('ROC: FPR vs TPR', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. F1-Score vs Window Size
        ax = axes[0, 1]
        for threshold in sorted(results_df['threshold_percentile'].unique()):
            subset = results_df[results_df['threshold_percentile'] == threshold]
            grouped = subset.groupby('window_size')['F1'].mean()
            ax.plot(grouped.index, grouped.values, marker='o',
                   label=f'threshold={threshold}%', linewidth=2, markersize=8)
        ax.set_xlabel('Window Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        ax.set_title('F1-Score vs Window Size', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Heatmap
        ax = axes[1, 0]
        pivot_data = results_df.pivot_table(
            values='F1',
            index='n_states',
            columns='threshold_percentile',
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                    vmin=0, vmax=1, cbar_kws={'label': 'F1-Score'})
        ax.set_title('F1-Score Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Threshold Percentile (%)', fontsize=12)
        ax.set_ylabel('Hidden States', fontsize=12)

        # 4. Best Configuration
        ax = axes[1, 1]
        best_params, _ = self.get_best_params(results_df, metric='F1')
        best_result = results_df[
            (results_df['window_size'] == best_params['window_size']) &
            (results_df['n_states'] == best_params['n_states']) &
            (results_df['threshold_percentile'] == best_params['threshold_percentile'])
        ].iloc[0]

        metrics = ['FPR', 'TPR', 'Precision', 'Recall', 'F1', 'Accuracy']
        values = [best_result[m] * 100 for m in metrics]
        colors = ['#ff4444' if m == 'FPR' else '#44ff44' for m in metrics]

        bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.axhline(y=5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='FPR Target')
        ax.axhline(y=90, color='green', linestyle='--', linewidth=2, alpha=0.5, label='High Performance')

        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{value:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Best Model Metrics\n{best_params}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Plot saved to: {save_path}")

        plt.close()


def main():
    """메인 함수"""
    logger.info("=" * 80)
    logger.info("OPTIMIZED HMM Grid Search")
    logger.info("Data Caching + Parallel Processing")
    logger.info("=" * 80)

    # 그리드 서치 설정
    param_grid = {
        'window_size': [300, 500, 700],
        'n_states': [10, 15, 20],
        'threshold_percentile': [5.0, 10.0, 15.0, 20.0]
    }

    # 최적화된 그리드 서치 실행
    searcher = OptimizedGridSearchHMM(data_dir='adfa-ld/ADFA-LD', n_jobs=-1)
    results_df = searcher.run_grid_search(param_grid)

    # 결과 분석
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    # 통계 요약
    logger.info("\nMetrics Statistics:")
    logger.info(results_df[['FPR', 'TPR', 'F1', 'Accuracy']].describe())

    # 최적 파라미터 (FPR <= 5% 제약)
    best_params_f1, best_f1 = searcher.get_best_params(results_df, metric='F1', fpr_constraint=0.05)
    logger.info(f"\n🏆 Best Parameters (F1-Score, FPR<=5%):")
    logger.info(f"  {best_params_f1}")
    logger.info(f"  F1-Score: {best_f1:.4f}")

    # 최적 파라미터 (TPR 기준)
    best_params_tpr, best_tpr = searcher.get_best_params(results_df, metric='TPR', fpr_constraint=0.05)
    logger.info(f"\n🎯 Best Parameters (TPR/Detection Rate, FPR<=5%):")
    logger.info(f"  {best_params_tpr}")
    logger.info(f"  TPR: {best_tpr:.4f}")

    # 시각화
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"results/grid_search_optimized_{timestamp}.png"
    searcher.plot_results(results_df, save_path=plot_path)

    logger.info("\n" + "=" * 80)
    logger.info("✓ Grid Search Completed Successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
