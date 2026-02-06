"""
성능 평가 모듈
"""
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix, classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetectionEvaluator:
    """이상탐지 성능 평가기"""

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        성능 메트릭 계산

        Args:
            y_true: 실제 레이블 (0: 정상, 1: 공격)
            y_pred: 예측 레이블 (0: 정상, 1: 공격)

        Returns:
            메트릭 딕셔너리
        """
        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # 메트릭 계산
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Detection Rate
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        metrics = {
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'FPR': fpr,
            'TPR': tpr,
            'Detection_Rate': tpr,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Accuracy': accuracy
        }

        return metrics

    @staticmethod
    def evaluate_attack_types(attack_data: Dict[str, List],
                              predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        공격 유형별 Detection Rate 계산

        Args:
            attack_data: {attack_type: sequences} 딕셔너리
            predictions: {attack_type: prediction_array} 딕셔너리

        Returns:
            {attack_type: detection_rate} 딕셔너리
        """
        attack_detection_rates = {}

        for attack_type in attack_data.keys():
            if attack_type not in predictions:
                continue

            preds = predictions[attack_type]
            # 모두 공격(1)이 정답
            detection_rate = np.mean(preds == 1)
            attack_detection_rates[attack_type] = detection_rate

        return attack_detection_rates

    @staticmethod
    def print_evaluation_report(metrics: Dict[str, float],
                               attack_detection_rates: Dict[str, float] = None):
        """
        평가 결과 출력

        Args:
            metrics: 전체 성능 메트릭
            attack_detection_rates: 공격 유형별 Detection Rate
        """
        print("\n" + "=" * 60)
        print("Performance Evaluation Report")
        print("=" * 60)

        print("\n[Confusion Matrix]")
        print(f"  True Negative  (TN): {metrics['TN']}")
        print(f"  False Positive (FP): {metrics['FP']}")
        print(f"  False Negative (FN): {metrics['FN']}")
        print(f"  True Positive  (TP): {metrics['TP']}")

        print("\n[Key Metrics]")
        print(f"  False Positive Rate (FPR): {metrics['FPR']:.4f} ({metrics['FPR']*100:.2f}%)")
        print(f"  True Positive Rate  (TPR): {metrics['TPR']:.4f} ({metrics['TPR']*100:.2f}%)")
        print(f"  Detection Rate:            {metrics['Detection_Rate']:.4f} ({metrics['Detection_Rate']*100:.2f}%)")
        print(f"  Precision:                 {metrics['Precision']:.4f}")
        print(f"  Recall:                    {metrics['Recall']:.4f}")
        print(f"  F1-Score:                  {metrics['F1']:.4f}")
        print(f"  Accuracy:                  {metrics['Accuracy']:.4f}")

        if attack_detection_rates:
            print("\n[Detection Rate by Attack Type]")
            for attack_type, rate in sorted(attack_detection_rates.items()):
                print(f"  {attack_type:20s}: {rate:.4f} ({rate*100:.2f}%)")

        print("\n" + "=" * 60)

    @staticmethod
    def save_results_to_file(filepath: str,
                            metrics: Dict[str, float],
                            attack_detection_rates: Dict[str, float] = None,
                            additional_info: Dict = None):
        """
        결과를 파일로 저장

        Args:
            filepath: 저장 경로
            metrics: 성능 메트릭
            attack_detection_rates: 공격 유형별 Detection Rate
            additional_info: 추가 정보
        """
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("HMM-Based Anomaly Detection - Evaluation Results\n")
            f.write("=" * 60 + "\n\n")

            if additional_info:
                f.write("[Configuration]\n")
                for key, value in additional_info.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

            f.write("[Confusion Matrix]\n")
            f.write(f"  True Negative  (TN): {metrics['TN']}\n")
            f.write(f"  False Positive (FP): {metrics['FP']}\n")
            f.write(f"  False Negative (FN): {metrics['FN']}\n")
            f.write(f"  True Positive  (TP): {metrics['TP']}\n")
            f.write("\n")

            f.write("[Performance Metrics]\n")
            f.write(f"  False Positive Rate (FPR): {metrics['FPR']:.4f} ({metrics['FPR']*100:.2f}%)\n")
            f.write(f"  True Positive Rate  (TPR): {metrics['TPR']:.4f} ({metrics['TPR']*100:.2f}%)\n")
            f.write(f"  Detection Rate:            {metrics['Detection_Rate']:.4f} ({metrics['Detection_Rate']*100:.2f}%)\n")
            f.write(f"  Precision:                 {metrics['Precision']:.4f}\n")
            f.write(f"  Recall:                    {metrics['Recall']:.4f}\n")
            f.write(f"  F1-Score:                  {metrics['F1']:.4f}\n")
            f.write(f"  Accuracy:                  {metrics['Accuracy']:.4f}\n")
            f.write("\n")

            if attack_detection_rates:
                f.write("[Detection Rate by Attack Type]\n")
                for attack_type, rate in sorted(attack_detection_rates.items()):
                    f.write(f"  {attack_type:20s}: {rate:.4f} ({rate*100:.2f}%)\n")
                f.write("\n")

            f.write("=" * 60 + "\n")

        logger.info(f"Results saved to {filepath}")
