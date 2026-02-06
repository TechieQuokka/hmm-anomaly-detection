"""
HMM + Hybrid 모델 통합 실행 스크립트
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
from hybrid_model import HybridHMMClassifier
from evaluator import AnomalyDetectionEvaluator
from visualizer import ResultVisualizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_hmm_experiment(
    data_dir='adfa-ld/ADFA-LD',
    window_size=500,
    n_states=15,
    threshold_percentile=5.0,
    experiment_name='hmm'
):
    """Pure HMM 모델 실험"""

    logger.info("=" * 80)
    logger.info("HMM Model Experiment")
    logger.info("=" * 80)

    # 데이터 로드
    data_loader = ADFADataLoader(data_dir=data_dir, window_size=window_size)
    normal_sequences = data_loader.load_normal_data(data_loader.train_dir)
    attack_data = data_loader.load_attack_data()

    logger.info(f"Loaded {len(normal_sequences)} normal sequences")
    logger.info(f"Loaded {sum(len(s) for s in attack_data.values())} attack sequences")

    # 데이터 분할
    train_seqs, val_seqs, test_normal_seqs = data_loader.split_data(
        normal_sequences, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42
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
    logger.info("\nTraining HMM...")
    detector = AnomalyDetectorHMM(
        n_states=n_states,
        n_observations=n_observations,
        random_state=42
    )
    detector.fit(train_seqs)
    detector.set_threshold_percentile(val_seqs, percentile=threshold_percentile)

    # 모델 저장
    Path("models").mkdir(exist_ok=True)  # 디렉토리 자동 생성
    model_path = f"models/hmm_model_{experiment_name}.pkl"
    detector.save_model(model_path)

    # 평가
    logger.info("\nEvaluating...")
    test_normal_preds, test_normal_scores = detector.predict_with_scores(test_normal_seqs)

    all_attack_preds = []
    all_attack_scores = []
    attack_predictions = {}

    for attack_type, sequences in attack_data.items():
        preds, scores = detector.predict_with_scores(sequences)
        attack_predictions[attack_type] = preds
        all_attack_preds.extend(preds)
        all_attack_scores.extend(scores)

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

    # 결과 출력
    evaluator.print_evaluation_report(metrics, attack_detection_rates)

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/evaluation_{experiment_name}_{timestamp}.txt"

    additional_info = {
        'Model Type': 'HMM',
        'Window Size': window_size,
        'Hidden States': n_states,
        'Threshold Percentile': f"{threshold_percentile}%",
        'Observations': n_observations
    }

    evaluator.save_results_to_file(results_file, metrics, attack_detection_rates, additional_info)

    # 시각화
    viz_dir = Path("results") / experiment_name
    viz_dir.mkdir(exist_ok=True)

    ResultVisualizer.create_all_visualizations(
        normal_scores=test_normal_scores,
        attack_scores=np.array(all_attack_scores),
        threshold=detector.threshold,
        metrics=metrics,
        attack_rates=attack_detection_rates,
        output_dir=str(viz_dir)
    )

    logger.info(f"\n✅ HMM Model Results:")
    logger.info(f"  FPR: {metrics['FPR']*100:.2f}%")
    logger.info(f"  TPR: {metrics['TPR']*100:.2f}%")
    logger.info(f"  F1:  {metrics['F1']:.4f}")
    logger.info(f"  Model saved: {model_path}")

    return metrics, attack_detection_rates


def run_hybrid_experiment(
    data_dir='adfa-ld/ADFA-LD',
    window_size=500,
    n_states=20,
    use_ngrams=True,
    experiment_name='hybrid'
):
    """Hybrid HMM + RF 모델 실험"""

    logger.info("=" * 80)
    logger.info("Hybrid HMM + Random Forest Experiment")
    logger.info("=" * 80)

    # 데이터 로드
    data_loader = ADFADataLoader(data_dir=data_dir, window_size=window_size)
    normal_sequences = data_loader.load_normal_data(data_loader.train_dir)
    attack_data = data_loader.load_attack_data()

    logger.info(f"Loaded {len(normal_sequences)} normal sequences")
    logger.info(f"Loaded {sum(len(s) for s in attack_data.values())} attack sequences")

    # 데이터 분할
    train_normal, val_normal, test_normal = data_loader.split_data(
        normal_sequences, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42
    )

    # 공격 데이터 분할
    train_attack = []
    test_attack_data = {}

    for attack_type, sequences in attack_data.items():
        n_train = int(len(sequences) * 0.5)
        train_attack.extend(sequences[:n_train])
        test_attack_data[attack_type] = sequences[n_train:]

    # 시스템 호출 매핑
    syscall_mapping = data_loader.get_syscall_mapping(train_normal)
    n_observations = len(syscall_mapping)

    train_normal = data_loader.apply_mapping(train_normal, syscall_mapping)
    val_normal = data_loader.apply_mapping(val_normal, syscall_mapping)
    test_normal = data_loader.apply_mapping(test_normal, syscall_mapping)
    train_attack = data_loader.apply_mapping(train_attack, syscall_mapping)

    for attack_type in test_attack_data.keys():
        test_attack_data[attack_type] = data_loader.apply_mapping(
            test_attack_data[attack_type], syscall_mapping
        )

    # Hybrid 모델 학습
    logger.info("\nTraining Hybrid Model...")
    hybrid_model = HybridHMMClassifier(
        n_states=n_states,
        n_observations=n_observations,
        random_state=42,
        use_ngrams=use_ngrams,
        n_estimators=100
    )

    hybrid_model.fit(
        normal_sequences=train_normal,
        attack_sequences=train_attack,
        val_normal_sequences=val_normal
    )

    # 모델 저장
    Path("models").mkdir(exist_ok=True)  # 디렉토리 자동 생성
    model_path = f"models/hybrid_model_{experiment_name}.pkl"
    hybrid_model.save_model(model_path)

    # 평가
    logger.info("\nEvaluating...")
    test_normal_preds, test_normal_scores = hybrid_model.predict_with_scores(test_normal)

    all_attack_preds = []
    all_attack_scores = []
    attack_predictions = {}

    for attack_type, sequences in test_attack_data.items():
        preds, scores = hybrid_model.predict_with_scores(sequences)
        attack_predictions[attack_type] = preds
        all_attack_preds.extend(preds)
        all_attack_scores.extend(scores)

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
    attack_detection_rates = evaluator.evaluate_attack_types(test_attack_data, attack_predictions)

    # 결과 출력
    evaluator.print_evaluation_report(metrics, attack_detection_rates)

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/evaluation_{experiment_name}_{timestamp}.txt"

    additional_info = {
        'Model Type': 'Hybrid HMM + Random Forest',
        'Window Size': window_size,
        'HMM Hidden States': n_states,
        'Use N-grams': use_ngrams,
        'RF n_estimators': 100,
        'Observations': n_observations
    }

    evaluator.save_results_to_file(results_file, metrics, attack_detection_rates, additional_info)

    # 시각화
    viz_dir = Path("results") / experiment_name
    viz_dir.mkdir(exist_ok=True)

    ResultVisualizer.create_all_visualizations(
        normal_scores=test_normal_scores,
        attack_scores=np.array(all_attack_scores),
        threshold=0.5,
        metrics=metrics,
        attack_rates=attack_detection_rates,
        output_dir=str(viz_dir)
    )

    logger.info(f"\n✅ Hybrid Model Results:")
    logger.info(f"  FPR: {metrics['FPR']*100:.2f}%")
    logger.info(f"  TPR: {metrics['TPR']*100:.2f}%")
    logger.info(f"  F1:  {metrics['F1']:.4f}")
    logger.info(f"  Model saved: {model_path}")

    return metrics, attack_detection_rates


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='HMM-based System Call Anomaly Detection'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='hybrid',
        choices=['hmm', 'hybrid'],
        help='Model type to use (default: hybrid)'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=500,
        help='Window size for sequences (default: 500)'
    )
    parser.add_argument(
        '--n-states',
        type=int,
        default=20,
        help='Number of hidden states for HMM (default: 20)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=5.0,
        help='Threshold percentile for HMM (default: 5.0)'
    )
    parser.add_argument(
        '--use-ngrams',
        action='store_true',
        default=True,
        help='Use n-grams for hybrid model (default: True)'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name (default: auto-generated)'
    )

    args = parser.parse_args()

    # 실험 이름 생성
    if args.experiment_name is None:
        if args.model == 'hmm':
            args.experiment_name = f"hmm_s{args.n_states}_t{args.threshold:.0f}"
        else:
            args.experiment_name = f"hybrid_s{args.n_states}_{'ngram' if args.use_ngrams else 'basic'}"

    # 실험 실행
    logger.info("\n" + "=" * 80)
    logger.info(f"Starting Experiment: {args.experiment_name}")
    logger.info(f"Model Type: {args.model.upper()}")
    logger.info("=" * 80 + "\n")

    if args.model == 'hmm':
        metrics, attack_rates = run_hmm_experiment(
            window_size=args.window_size,
            n_states=args.n_states,
            threshold_percentile=args.threshold,
            experiment_name=args.experiment_name
        )
    else:  # hybrid
        metrics, attack_rates = run_hybrid_experiment(
            window_size=args.window_size,
            n_states=args.n_states,
            use_ngrams=args.use_ngrams,
            experiment_name=args.experiment_name
        )

    # 최종 요약
    logger.info("\n" + "=" * 80)
    logger.info("✅ Experiment Completed Successfully!")
    logger.info("=" * 80)
    logger.info(f"\n📊 Final Results:")
    logger.info(f"   Model: {args.model.upper()}")
    logger.info(f"   FPR: {metrics['FPR']*100:.2f}% {'✅' if metrics['FPR'] <= 0.05 else '⚠️'}")
    logger.info(f"   TPR: {metrics['TPR']*100:.2f}% {'✅' if metrics['TPR'] >= 0.8 else '⚠️'}")
    logger.info(f"   F1:  {metrics['F1']:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
