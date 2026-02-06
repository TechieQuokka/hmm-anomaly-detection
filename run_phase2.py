"""
Phase 2: Hybrid HMM + RF 실험 실행
"""
import numpy as np
import sys
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd

# 로컬 모듈
sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_loader import ADFADataLoader
from hybrid_model import HybridHMMClassifier
from evaluator import AnomalyDetectionEvaluator
from visualizer import ResultVisualizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_phase2_experiment(
    data_dir: str = 'adfa-ld/ADFA-LD',
    window_size: int = 500,
    n_states: int = 20,
    use_ngrams: bool = True,
    random_seed: int = 42
):
    """
    Phase 2 실험 실행

    Args:
        data_dir: 데이터 디렉토리
        window_size: 윈도우 크기
        n_states: HMM hidden states
        use_ngrams: N-gram 특징 사용 여부
        random_seed: 랜덤 시드
    """
    logger.info("=" * 80)
    logger.info("Phase 2: Hybrid HMM + Random Forest Experiment")
    logger.info("=" * 80)
    logger.info(f"\nConfiguration:")
    logger.info(f"  data_dir: {data_dir}")
    logger.info(f"  window_size: {window_size}")
    logger.info(f"  n_states: {n_states}")
    logger.info(f"  use_ngrams: {use_ngrams}")
    logger.info(f"  random_seed: {random_seed}")

    # ========================================
    # 1. 데이터 로드
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading Data")
    logger.info("=" * 80)

    data_loader = ADFADataLoader(
        data_dir=data_dir,
        window_size=window_size
    )

    # 정상 데이터
    normal_sequences = data_loader.load_normal_data(data_loader.train_dir)
    logger.info(f"Loaded {len(normal_sequences)} normal sequences")

    # 공격 데이터
    attack_data = data_loader.load_attack_data()
    total_attacks = sum(len(seqs) for seqs in attack_data.values())
    logger.info(f"Loaded {total_attacks} attack sequences from {len(attack_data)} attack types")

    # ========================================
    # 2. 데이터 분할
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Splitting Data")
    logger.info("=" * 80)

    # 정상 데이터 분할
    train_normal, val_normal, test_normal = data_loader.split_data(
        normal_sequences,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_seed=random_seed
    )

    # 공격 데이터 분할 (train/test만, validation 없음)
    # Train: 모델 학습용, Test: 평가용
    train_attack = []
    test_attack_data = {}

    for attack_type, sequences in attack_data.items():
        n_train = int(len(sequences) * 0.5)  # 50% 학습용
        train_attack.extend(sequences[:n_train])
        test_attack_data[attack_type] = sequences[n_train:]

    logger.info(f"Data split:")
    logger.info(f"  Train Normal: {len(train_normal)}")
    logger.info(f"  Val Normal: {len(val_normal)}")
    logger.info(f"  Test Normal: {len(test_normal)}")
    logger.info(f"  Train Attack: {len(train_attack)}")
    logger.info(f"  Test Attack: {sum(len(seqs) for seqs in test_attack_data.values())}")

    # ========================================
    # 3. 시스템 호출 매핑
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Creating System Call Mapping")
    logger.info("=" * 80)

    syscall_mapping = data_loader.get_syscall_mapping(train_normal)
    n_observations = len(syscall_mapping)
    logger.info(f"Number of unique system calls: {n_observations}")

    # 매핑 적용
    train_normal = data_loader.apply_mapping(train_normal, syscall_mapping)
    val_normal = data_loader.apply_mapping(val_normal, syscall_mapping)
    test_normal = data_loader.apply_mapping(test_normal, syscall_mapping)
    train_attack = data_loader.apply_mapping(train_attack, syscall_mapping)

    for attack_type in test_attack_data.keys():
        test_attack_data[attack_type] = data_loader.apply_mapping(
            test_attack_data[attack_type], syscall_mapping
        )

    # ========================================
    # 4. Hybrid 모델 학습
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Training Hybrid Model")
    logger.info("=" * 80)

    hybrid_model = HybridHMMClassifier(
        n_states=n_states,
        n_observations=n_observations,
        random_state=random_seed,
        use_ngrams=use_ngrams,
        n_estimators=100
    )

    hybrid_model.fit(
        normal_sequences=train_normal,
        attack_sequences=train_attack,
        val_normal_sequences=val_normal
    )

    # 모델 저장
    model_path = f"models/hybrid_model_phase2.pkl"
    hybrid_model.save_model(model_path)

    # ========================================
    # 5. 평가
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Evaluation")
    logger.info("=" * 80)

    # 정상 데이터 예측
    test_normal_preds, test_normal_scores = hybrid_model.predict_with_scores(test_normal)
    logger.info(f"Normal test sequences: {len(test_normal)}")
    logger.info(f"  Predicted as normal: {np.sum(test_normal_preds == 0)}")
    logger.info(f"  Predicted as attack: {np.sum(test_normal_preds == 1)}")

    # 공격 데이터 예측
    all_attack_preds = []
    all_attack_scores = []
    attack_predictions = {}

    logger.info("\nAttack data predictions:")
    for attack_type, sequences in test_attack_data.items():
        preds, scores = hybrid_model.predict_with_scores(sequences)
        attack_predictions[attack_type] = preds

        all_attack_preds.extend(preds)
        all_attack_scores.extend(scores)

        detected = np.sum(preds == 1)
        total = len(preds)
        logger.info(f"  {attack_type:20s}: {detected}/{total} detected ({detected/total*100:.2f}%)")

    # 전체 레이블 및 예측
    y_true = np.concatenate([
        np.zeros(len(test_normal_preds)),
        np.ones(len(all_attack_preds))
    ])

    y_pred = np.concatenate([
        test_normal_preds,
        all_attack_preds
    ])

    # ========================================
    # 6. 메트릭 계산
    # ========================================
    evaluator = AnomalyDetectionEvaluator()
    metrics = evaluator.compute_metrics(y_true, y_pred)
    attack_detection_rates = evaluator.evaluate_attack_types(test_attack_data, attack_predictions)

    # 결과 출력
    evaluator.print_evaluation_report(metrics, attack_detection_rates)

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/evaluation_phase2_{timestamp}.txt"

    additional_info = {
        'Phase': 'Phase 2 - Hybrid HMM + Random Forest',
        'Timestamp': timestamp,
        'Window Size': window_size,
        'HMM Hidden States': n_states,
        'Observations': n_observations,
        'Use N-grams': use_ngrams,
        'RF n_estimators': 100,
        'Train Normal': len(train_normal),
        'Train Attack': len(train_attack),
        'Validation Normal': len(val_normal),
        'Test Normal': len(test_normal),
        'Test Attack': len(all_attack_preds)
    }

    evaluator.save_results_to_file(results_file, metrics, attack_detection_rates, additional_info)

    # ========================================
    # 7. 시각화
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Creating Visualizations")
    logger.info("=" * 80)

    viz_dir = Path("results") / "phase2"
    viz_dir.mkdir(exist_ok=True)

    ResultVisualizer.create_all_visualizations(
        normal_scores=test_normal_scores,
        attack_scores=np.array(all_attack_scores),
        threshold=0.5,  # Hybrid 모델은 확률 0.5 기준
        metrics=metrics,
        attack_rates=attack_detection_rates,
        output_dir=str(viz_dir)
    )

    # ========================================
    # 8. 최종 요약
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("Phase 2 Experiment Completed Successfully!")
    logger.info("=" * 80)
    logger.info(f"\nKey Results:")
    logger.info(f"  FPR: {metrics['FPR']*100:.2f}% (Target: <5%)")
    logger.info(f"  TPR: {metrics['TPR']*100:.2f}% (Detection Rate)")
    logger.info(f"  F1-Score: {metrics['F1']:.4f}")
    logger.info(f"  Precision: {metrics['Precision']:.4f}")
    logger.info(f"\nOutputs saved to:")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Results: {results_file}")
    logger.info(f"  Visualizations: {viz_dir}/")

    return metrics, attack_detection_rates


def compare_with_phase1(phase2_metrics: dict, phase2_attack_rates: dict):
    """Phase 1 최적 결과와 비교"""
    logger.info("\n" + "=" * 80)
    logger.info("Comparison: Phase 2 vs Phase 1 (Best: n_states=20, threshold=7%)")
    logger.info("=" * 80)

    # Phase 1 최적 결과 (grid search 결과에서)
    phase1_metrics = {
        'FPR': 0.0556,
        'TPR': 0.3581,
        'F1': 0.524,
        'Precision': 0.975,
        'Accuracy': 0.442
    }

    phase1_attack_rates = {
        'Adduser': 0.3667,
        'Hydra_FTP': 0.5714,
        'Hydra_SSH': 0.30,
        'Java_Meterpreter': 0.2955,
        'Meterpreter': 0.4286,
        'Web_Shell': 0.2889
    }

    # 비교 테이블
    comparison = {
        'Metric': [],
        'Phase 1': [],
        'Phase 2': [],
        'Change': [],
        'Improvement': []
    }

    metrics_to_compare = ['FPR', 'TPR', 'F1', 'Precision', 'Accuracy']
    for metric in metrics_to_compare:
        p1_val = phase1_metrics.get(metric, 0) * 100
        p2_val = phase2_metrics.get(metric, 0) * 100
        change = p2_val - p1_val

        comparison['Metric'].append(metric)
        comparison['Phase 1'].append(f"{p1_val:.2f}%")
        comparison['Phase 2'].append(f"{p2_val:.2f}%")
        comparison['Change'].append(f"{change:+.2f}%")

        if metric == 'FPR':
            improvement = "✅" if change < 0 else "⚠️"
        else:
            improvement = "✅" if change > 0 else "⚠️"
        comparison['Improvement'].append(improvement)

    df = pd.DataFrame(comparison)
    print("\n" + df.to_string(index=False))

    logger.info("\nAttack Detection Rate Comparison:")
    for attack_type in phase1_attack_rates.keys():
        p1_rate = phase1_attack_rates.get(attack_type, 0) * 100
        p2_rate = phase2_attack_rates.get(attack_type, 0) * 100
        change = p2_rate - p1_rate
        improvement = "✅" if change > 0 else "⚠️"
        logger.info(f"  {attack_type:20s}: P1={p1_rate:5.1f}% → P2={p2_rate:5.1f}% "
                   f"({change:+5.1f}%) {improvement}")


if __name__ == "__main__":
    # Phase 2 실험 실행
    metrics, attack_rates = run_phase2_experiment(
        data_dir='adfa-ld/ADFA-LD',
        window_size=500,
        n_states=20,
        use_ngrams=True,
        random_seed=42
    )

    # Phase 1과 비교
    compare_with_phase1(metrics, attack_rates)
