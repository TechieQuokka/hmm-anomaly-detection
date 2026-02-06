"""
집중 Grid Search - FPR < 5% 목표 달성
"""
import sys
from pathlib import Path

# grid_search 모듈 임포트
from grid_search import GridSearchHMM
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """집중 Grid Search 실행"""
    logger.info("=" * 80)
    logger.info("Focused Grid Search - Optimizing for FPR < 5%")
    logger.info("=" * 80)

    # Phase 1 결과 기반 집중 탐색
    # - n_states=15가 좋은 결과를 보였으므로 15와 20만 테스트
    # - window_size=500이 표준이므로 고정
    # - threshold를 7-13% 범위에서 미세 조정
    param_grid = {
        'window_size': [500],  # 고정
        'n_states': [15, 20],  # 15는 검증됨, 20은 더 복잡한 패턴 포착 가능
        'threshold_percentile': [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
    }

    logger.info("\nFocused Parameter Grid:")
    logger.info("  window_size: [500] (fixed)")
    logger.info("  n_states: [15, 20]")
    logger.info("  threshold_percentile: [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]")
    logger.info(f"\nTotal experiments: {1 * 2 * 7} = 14")
    logger.info("Estimated time: 30-40 minutes\n")

    # Grid search 실행
    searcher = GridSearchHMM(data_dir='adfa-ld/ADFA-LD')
    results_df = searcher.run_grid_search(param_grid)

    # 결과 분석
    logger.info("\n" + "=" * 80)
    logger.info("Focused Grid Search Summary")
    logger.info("=" * 80)

    # FPR < 5% 조건 만족하는 결과만 필터링
    valid_results = results_df[results_df['FPR'] <= 0.05]

    if len(valid_results) == 0:
        logger.warning("No configurations satisfy FPR <= 5%!")
        logger.info("\nShowing all results:")
        logger.info(results_df[['n_states', 'threshold_percentile', 'FPR', 'TPR', 'F1']].to_string(index=False))
    else:
        logger.info(f"\nConfigurations satisfying FPR <= 5%: {len(valid_results)}")
        logger.info("\nTop 5 by F1-Score:")
        top5_f1 = valid_results.nlargest(5, 'F1')[['n_states', 'threshold_percentile', 'FPR', 'TPR', 'F1', 'Precision', 'Recall']]
        print("\n" + top5_f1.to_string(index=False))

        logger.info("\nTop 5 by TPR (Detection Rate):")
        top5_tpr = valid_results.nlargest(5, 'TPR')[['n_states', 'threshold_percentile', 'FPR', 'TPR', 'F1', 'Precision', 'Recall']]
        print("\n" + top5_tpr.to_string(index=False))

        # 최적 파라미터 (F1 기준)
        best_params_f1, best_f1 = searcher.get_best_params(results_df, metric='F1', fpr_constraint=0.05)
        best_row_f1 = valid_results[
            (valid_results['n_states'] == best_params_f1['n_states']) &
            (valid_results['threshold_percentile'] == best_params_f1['threshold_percentile'])
        ].iloc[0]

        logger.info(f"\n{'=' * 80}")
        logger.info("RECOMMENDED CONFIGURATION (Best F1-Score with FPR < 5%)")
        logger.info(f"{'=' * 80}")
        logger.info(f"  n_states: {int(best_params_f1['n_states'])}")
        logger.info(f"  threshold_percentile: {best_params_f1['threshold_percentile']}%")
        logger.info(f"\nPerformance:")
        logger.info(f"  FPR: {best_row_f1['FPR']*100:.2f}%")
        logger.info(f"  TPR: {best_row_f1['TPR']*100:.2f}%")
        logger.info(f"  F1-Score: {best_row_f1['F1']:.4f}")
        logger.info(f"  Precision: {best_row_f1['Precision']:.4f}")
        logger.info(f"  Recall: {best_row_f1['Recall']:.4f}")
        logger.info(f"  Accuracy: {best_row_f1['Accuracy']:.4f}")

        # 공격 유형별 탐지율
        logger.info(f"\nDetection Rates by Attack Type:")
        attack_types = ['Adduser', 'Hydra_FTP', 'Hydra_SSH', 'Java_Meterpreter', 'Meterpreter', 'Web_Shell']
        for attack in attack_types:
            col_name = f'DR_{attack}'
            if col_name in best_row_f1:
                logger.info(f"  {attack:20s}: {best_row_f1[col_name]*100:.2f}%")

    # 시각화
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"results/focused_grid_search_plot_{timestamp}.png"
    searcher.plot_results(results_df, save_path=plot_path)

    logger.info("\n" + "=" * 80)
    logger.info("Focused Grid Search Completed!")
    logger.info("=" * 80)

    return results_df


if __name__ == "__main__":
    main()
