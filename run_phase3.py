"""
Phase 3: 최적화 및 미세 조정
- Threshold 최적화
- Probability Calibration
- Hyperparameter Tuning
"""
import numpy as np
import sys
from pathlib import Path
import logging
from datetime import datetime
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

# 로컬 모듈
sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_loader import ADFADataLoader
from hybrid_model import HybridHMMClassifier
from evaluator import AnomalyDetectionEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_optimal_threshold(model, X_val, y_val, target_fpr=0.02):
    """
    최적 threshold 찾기 (FPR 제약 하에 TPR 최대화)

    Args:
        model: 학습된 모델
        X_val: Validation 특징
        y_val: Validation 레이블
        target_fpr: 목표 FPR (default: 2%)

    Returns:
        optimal_threshold
    """
    logger.info(f"\nFinding optimal threshold (target FPR <= {target_fpr*100}%)...")

    # 확률 예측
    y_proba = model.predict_proba(X_val)[:, 1]

    # 다양한 threshold 시도
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_tpr = 0

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        # FPR, TPR 계산
        tn = np.sum((y_val == 0) & (y_pred == 0))
        fp = np.sum((y_val == 0) & (y_pred == 1))
        fn = np.sum((y_val == 1) & (y_pred == 0))
        tp = np.sum((y_val == 1) & (y_pred == 1))

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

        # FPR 제약 만족하면서 TPR 최대화
        if fpr <= target_fpr and tpr > best_tpr:
            best_tpr = tpr
            best_threshold = threshold

    logger.info(f"Optimal threshold: {best_threshold:.3f} (TPR: {best_tpr:.3f})")
    return best_threshold


class OptimizedHybridClassifier:
    """Phase 3: 최적화된 Hybrid 분류기"""

    def __init__(self, base_model, optimal_threshold=0.5, use_calibration=True):
        """
        Args:
            base_model: Phase 2 Hybrid 모델
            optimal_threshold: 최적 threshold
            use_calibration: Calibration 사용 여부
        """
        self.base_model = base_model
        self.optimal_threshold = optimal_threshold
        self.use_calibration = use_calibration
        self.calibrated_model = None

    def fit_calibration(self, X_cal, y_cal):
        """
        Probability Calibration 학습

        Args:
            X_cal: Calibration 데이터 특징
            y_cal: Calibration 데이터 레이블
        """
        if not self.use_calibration:
            return

        logger.info("\nApplying Probability Calibration...")

        # Isotonic regression calibration
        self.calibrated_model = CalibratedClassifierCV(
            self.base_model.rf_classifier,
            method='isotonic',
            cv='prefit'
        )

        # Base 모델에서 특징 추출
        X_cal_features = self.base_model._extract_hybrid_features(X_cal)
        X_cal_scaled = self.base_model.scaler.transform(X_cal_features)

        self.calibrated_model.fit(X_cal_scaled, y_cal)
        logger.info("Calibration completed")

    def predict(self, sequences):
        """예측"""
        proba = self.predict_proba(sequences)
        return (proba[:, 1] >= self.optimal_threshold).astype(int)

    def predict_proba(self, sequences):
        """확률 예측"""
        X = self.base_model._extract_hybrid_features(sequences)
        X_scaled = self.base_model.scaler.transform(X)

        if self.use_calibration and self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X_scaled)
        else:
            return self.base_model.rf_classifier.predict_proba(X_scaled)

    def predict_with_scores(self, sequences):
        """예측 레이블과 확률 반환"""
        probabilities = self.predict_proba(sequences)
        predictions = (probabilities[:, 1] >= self.optimal_threshold).astype(int)
        return predictions, probabilities[:, 1]


def run_phase3_experiment(
    data_dir: str = 'adfa-ld/ADFA-LD',
    window_size: int = 500,
    n_states: int = 20,
    target_fpr: float = 0.02,
    use_calibration: bool = True,
    random_seed: int = 42
):
    """Phase 3 실험 실행"""

    logger.info("=" * 80)
    logger.info("Phase 3: Optimization & Fine-tuning")
    logger.info("=" * 80)
    logger.info(f"\nConfiguration:")
    logger.info(f"  target_fpr: {target_fpr*100}%")
    logger.info(f"  use_calibration: {use_calibration}")

    # ========================================
    # 1. 데이터 로드 및 준비
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading Data")
    logger.info("=" * 80)

    data_loader = ADFADataLoader(data_dir=data_dir, window_size=window_size)

    normal_sequences = data_loader.load_normal_data(data_loader.train_dir)
    attack_data = data_loader.load_attack_data()

    logger.info(f"Loaded {len(normal_sequences)} normal sequences")
    logger.info(f"Loaded {sum(len(seqs) for seqs in attack_data.values())} attack sequences")

    # 데이터 분할
    train_normal, val_normal, test_normal = data_loader.split_data(
        normal_sequences, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=random_seed
    )

    train_attack = []
    val_attack = []
    test_attack_data = {}

    for attack_type, sequences in attack_data.items():
        n_train = int(len(sequences) * 0.4)
        n_val = int(len(sequences) * 0.2)

        train_attack.extend(sequences[:n_train])
        val_attack.extend(sequences[n_train:n_train+n_val])
        test_attack_data[attack_type] = sequences[n_train+n_val:]

    logger.info(f"\nData split:")
    logger.info(f"  Train: Normal={len(train_normal)}, Attack={len(train_attack)}")
    logger.info(f"  Val: Normal={len(val_normal)}, Attack={len(val_attack)}")
    logger.info(f"  Test: Normal={len(test_normal)}, Attack={sum(len(seqs) for seqs in test_attack_data.values())}")

    # 시스템 호출 매핑
    syscall_mapping = data_loader.get_syscall_mapping(train_normal)
    n_observations = len(syscall_mapping)

    train_normal = data_loader.apply_mapping(train_normal, syscall_mapping)
    val_normal = data_loader.apply_mapping(val_normal, syscall_mapping)
    test_normal = data_loader.apply_mapping(test_normal, syscall_mapping)
    train_attack = data_loader.apply_mapping(train_attack, syscall_mapping)
    val_attack = data_loader.apply_mapping(val_attack, syscall_mapping)

    for attack_type in test_attack_data.keys():
        test_attack_data[attack_type] = data_loader.apply_mapping(
            test_attack_data[attack_type], syscall_mapping
        )

    # ========================================
    # 2. Base 모델 학습 (Phase 2와 동일)
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Training Base Model")
    logger.info("=" * 80)

    base_model = HybridHMMClassifier(
        n_states=n_states,
        n_observations=n_observations,
        random_state=random_seed,
        use_ngrams=True,
        n_estimators=150  # Phase 2: 100 → 150
    )

    base_model.fit(train_normal, train_attack, val_normal_sequences=val_normal)

    # ========================================
    # 3. Threshold 최적화
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Optimizing Threshold")
    logger.info("=" * 80)

    # Validation 데이터로 threshold 찾기
    val_sequences = val_normal + val_attack
    val_labels = np.concatenate([np.zeros(len(val_normal)), np.ones(len(val_attack))])

    optimal_threshold = find_optimal_threshold(
        base_model, val_sequences, val_labels, target_fpr=target_fpr
    )

    # ========================================
    # 4. Calibration (선택)
    # ========================================
    optimized_model = OptimizedHybridClassifier(
        base_model=base_model,
        optimal_threshold=optimal_threshold,
        use_calibration=use_calibration
    )

    if use_calibration:
        optimized_model.fit_calibration(val_sequences, val_labels)

    # 모델 저장
    import pickle
    model_path = "models/optimized_model_phase3.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(optimized_model, f)
    logger.info(f"\nModel saved to {model_path}")

    # ========================================
    # 5. 평가
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Evaluation")
    logger.info("=" * 80)

    # 정상 데이터 예측
    test_normal_preds, test_normal_scores = optimized_model.predict_with_scores(test_normal)
    logger.info(f"Normal: {np.sum(test_normal_preds == 0)}/{len(test_normal)} correct")

    # 공격 데이터 예측
    all_attack_preds = []
    all_attack_scores = []
    attack_predictions = {}

    for attack_type, sequences in test_attack_data.items():
        preds, scores = optimized_model.predict_with_scores(sequences)
        attack_predictions[attack_type] = preds
        all_attack_preds.extend(preds)
        all_attack_scores.extend(scores)

        detected = np.sum(preds == 1)
        logger.info(f"  {attack_type:20s}: {detected}/{len(sequences)} detected ({detected/len(sequences)*100:.2f}%)")

    # 메트릭 계산
    y_true = np.concatenate([np.zeros(len(test_normal_preds)), np.ones(len(all_attack_preds))])
    y_pred = np.concatenate([test_normal_preds, all_attack_preds])

    evaluator = AnomalyDetectionEvaluator()
    metrics = evaluator.compute_metrics(y_true, y_pred)
    attack_detection_rates = evaluator.evaluate_attack_types(test_attack_data, attack_predictions)

    evaluator.print_evaluation_report(metrics, attack_detection_rates)

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/evaluation_phase3_{timestamp}.txt"

    additional_info = {
        'Phase': 'Phase 3 - Optimized Hybrid',
        'Optimal Threshold': f"{optimal_threshold:.3f}",
        'Target FPR': f"{target_fpr*100}%",
        'Use Calibration': use_calibration,
        'RF n_estimators': 150
    }

    evaluator.save_results_to_file(results_file, metrics, attack_detection_rates, additional_info)

    # ========================================
    # 최종 요약
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("Phase 3 Completed!")
    logger.info("=" * 80)
    logger.info(f"\nResults:")
    logger.info(f"  FPR: {metrics['FPR']*100:.2f}% (Target: <{target_fpr*100}%)")
    logger.info(f"  TPR: {metrics['TPR']*100:.2f}%")
    logger.info(f"  F1-Score: {metrics['F1']:.4f}")
    logger.info(f"  Precision: {metrics['Precision']:.4f}")

    return metrics, attack_detection_rates


if __name__ == "__main__":
    metrics, attack_rates = run_phase3_experiment(
        target_fpr=0.02,  # 2% 목표
        use_calibration=True
    )
