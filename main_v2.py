"""
HMM 기반 시스템 호출 이상탐지 - 메인 실행 스크립트 (개선 버전)
"""
import numpy as np
import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime

# 로컬 모듈 임포트
sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_loader import ADFADataLoader
from hmm_model import AnomalyDetectorHMM
from evaluator import AnomalyDetectionEvaluator
from visualizer import ResultVisualizer

# 설정 임포트
from config import ExperimentConfig, get_config_by_name

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_experiment(config: ExperimentConfig):
    """실험 실행 함수

    Args:
        config: 실험 설정
    """

    # 로그 파일 핸들러 추가
    log_file = Path(config.log_dir) / f"experiment_{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 60)
    logger.info(f"HMM-Based System Call Anomaly Detection")
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info("=" * 60)
    logger.info(f"\n{config}")

    # ========================================
    # 1. 데이터 로드
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Loading Data")
    logger.info("=" * 60)

    data_loader = ADFADataLoader(
        data_dir=config.data_dir,
        window_size=config.window_size
    )

    # 정상 데이터 로드
    normal_sequences = data_loader.load_normal_data(data_loader.train_dir)
    logger.info(f"Loaded {len(normal_sequences)} normal sequences")

    # 공격 데이터 로드
    attack_data = data_loader.load_attack_data()
    total_attacks = sum(len(seqs) for seqs in attack_data.values())
    logger.info(f"Loaded {total_attacks} attack sequences from {len(attack_data)} attack types")

    # ========================================
    # 2. 데이터 분할
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Splitting Data")
    logger.info("=" * 60)

    train_seqs, val_seqs, test_normal_seqs = data_loader.split_data(
        normal_sequences,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_seed=config.random_seed
    )

    # ========================================
    # 3. 시스템 호출 매핑
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Creating System Call Mapping")
    logger.info("=" * 60)

    syscall_mapping = data_loader.get_syscall_mapping(train_seqs)
    n_observations = len(syscall_mapping)
    logger.info(f"Number of unique system calls: {n_observations}")

    # 매핑 적용
    train_seqs = data_loader.apply_mapping(train_seqs, syscall_mapping)
    val_seqs = data_loader.apply_mapping(val_seqs, syscall_mapping)
    test_normal_seqs = data_loader.apply_mapping(test_normal_seqs, syscall_mapping)

    # 공격 데이터에도 매핑 적용
    for attack_type in attack_data.keys():
        attack_data[attack_type] = data_loader.apply_mapping(
            attack_data[attack_type], syscall_mapping
        )

    # ========================================
    # 4. HMM 학습
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Training HMM Model")
    logger.info("=" * 60)

    detector = AnomalyDetectorHMM(
        n_states=config.n_states,
        n_observations=n_observations,
        random_state=config.random_seed
    )

    detector.fit(train_seqs)

    # 모델 저장
    model_path = Path(config.model_dir) / f"hmm_model_{config.experiment_name}.pkl"
    detector.save_model(str(model_path))

    # ========================================
    # 5. Threshold 설정
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Setting Threshold")
    logger.info("=" * 60)

    detector.set_threshold_percentile(
        val_seqs,
        percentile=config.threshold_percentile
    )

    # ========================================
    # 6. 평가
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("Step 6: Evaluation")
    logger.info("=" * 60)

    # Test set 정상 데이터 예측
    test_normal_preds, test_normal_scores = detector.predict_with_scores(test_normal_seqs)
    logger.info(f"Normal test sequences: {len(test_normal_seqs)}")
    logger.info(f"  Predicted as normal: {np.sum(test_normal_preds == 0)}")
    logger.info(f"  Predicted as attack: {np.sum(test_normal_preds == 1)}")

    # 공격 데이터 예측
    all_attack_preds = []
    all_attack_scores = []
    attack_predictions = {}

    logger.info("\nAttack data predictions:")
    for attack_type, sequences in attack_data.items():
        preds, scores = detector.predict_with_scores(sequences)
        attack_predictions[attack_type] = preds

        all_attack_preds.extend(preds)
        all_attack_scores.extend(scores)

        detected = np.sum(preds == 1)
        total = len(preds)
        logger.info(f"  {attack_type:20s}: {detected}/{total} detected ({detected/total*100:.2f}%)")

    # 전체 레이블 및 예측 통합
    y_true = np.concatenate([
        np.zeros(len(test_normal_preds)),  # 정상 = 0
        np.ones(len(all_attack_preds))      # 공격 = 1
    ])

    y_pred = np.concatenate([
        test_normal_preds,
        all_attack_preds
    ])

    # ========================================
    # 7. 메트릭 계산
    # ========================================
    evaluator = AnomalyDetectionEvaluator()

    metrics = evaluator.compute_metrics(y_true, y_pred)
    attack_detection_rates = evaluator.evaluate_attack_types(attack_data, attack_predictions)

    # 결과 출력
    evaluator.print_evaluation_report(metrics, attack_detection_rates)

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(config.output_dir) / f"evaluation_{config.experiment_name}_{timestamp}.txt"

    additional_info = {
        'Experiment Name': config.experiment_name,
        'Description': config.description,
        'Timestamp': timestamp,
        'Window Size': config.window_size,
        'Hidden States': config.n_states,
        'Observations': n_observations,
        'Threshold': f"{detector.threshold:.4f}",
        'Threshold Percentile': f"{config.threshold_percentile}%",
        'Train Sequences': len(train_seqs),
        'Validation Sequences': len(val_seqs),
        'Test Normal Sequences': len(test_normal_seqs),
        'Test Attack Sequences': len(all_attack_preds)
    }

    evaluator.save_results_to_file(str(results_file), metrics, attack_detection_rates, additional_info)

    # ========================================
    # 8. 시각화
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("Step 7: Creating Visualizations")
    logger.info("=" * 60)

    # 실험별 서브디렉토리 생성
    viz_dir = Path(config.output_dir) / config.experiment_name
    viz_dir.mkdir(exist_ok=True)

    ResultVisualizer.create_all_visualizations(
        normal_scores=test_normal_scores,
        attack_scores=np.array(all_attack_scores),
        threshold=detector.threshold,
        metrics=metrics,
        attack_rates=attack_detection_rates,
        output_dir=str(viz_dir)
    )

    # ========================================
    # 9. 최종 요약
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("Experiment Completed Successfully!")
    logger.info("=" * 60)
    logger.info(f"\nExperiment: {config.experiment_name}")
    logger.info(f"Key Results:")
    logger.info(f"  FPR: {metrics['FPR']*100:.2f}% (Target: <5%)")
    logger.info(f"  TPR: {metrics['TPR']*100:.2f}% (Detection Rate)")
    logger.info(f"  F1-Score: {metrics['F1']:.4f}")
    logger.info(f"  Precision: {metrics['Precision']:.4f}")
    logger.info(f"\nOutputs saved to:")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Results: {results_file}")
    logger.info(f"  Visualizations: {viz_dir}/")
    logger.info(f"  Log: {log_file}")

    return metrics, attack_detection_rates


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='HMM-based System Call Anomaly Detection'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='baseline',
        choices=['baseline', 'improved', 'window_300', 'window_700'],
        help='Configuration to use'
    )
    parser.add_argument(
        '--config-file',
        type=str,
        help='Path to YAML config file (overrides --config)'
    )

    args = parser.parse_args()

    # 설정 로드
    if args.config_file:
        config = ExperimentConfig.from_yaml(args.config_file)
    else:
        config = get_config_by_name(args.config)

    # 실험 실행
    run_experiment(config)


if __name__ == "__main__":
    main()
