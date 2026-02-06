"""
HMM 기반 이상탐지 모델
"""
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple
from hmmlearn import hmm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetectorHMM:
    """Single-class HMM 기반 이상탐지기"""

    def __init__(self, n_states: int = 5, n_observations: int = 48,
                 random_state: int = 42):
        """
        Args:
            n_states: Hidden state 개수
            n_observations: 관측 심볼(시스템 호출) 개수
            random_state: 랜덤 시드
        """
        self.n_states = n_states
        self.n_observations = n_observations
        self.random_state = random_state
        self.threshold = None

        # Discrete HMM 생성 (CategoricalHMM 사용)
        self.model = hmm.CategoricalHMM(
            n_components=n_states,
            n_features=n_observations,
            random_state=random_state,
            n_iter=100,  # Baum-Welch 최대 반복 횟수
            tol=1e-4,  # 수렴 임계값
            verbose=True
        )

        logger.info(f"Created HMM with {n_states} states, "
                   f"{n_observations} observations")

    def prepare_sequences(self, sequences: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        hmmlearn 입력 형식으로 변환

        Args:
            sequences: 시퀀스 리스트

        Returns:
            (X, lengths)
            - X: 모든 시퀀스를 연결한 1D 배열
            - lengths: 각 시퀀스의 길이
        """
        X = np.concatenate(sequences).reshape(-1, 1)
        lengths = [len(seq) for seq in sequences]
        return X, lengths

    def fit(self, train_sequences: List[List[int]]):
        """
        정상 데이터로 HMM 학습

        Args:
            train_sequences: 학습 시퀀스 리스트
        """
        logger.info(f"Training HMM on {len(train_sequences)} sequences...")

        X, lengths = self.prepare_sequences(train_sequences)

        # Baum-Welch 알고리즘으로 학습
        self.model.fit(X, lengths)

        logger.info("Training completed")
        logger.info(f"Final log-likelihood: {self.model.score(X, lengths):.2f}")

    def compute_log_likelihood(self, sequences: List[List[int]]) -> np.ndarray:
        """
        시퀀스들의 log-likelihood 계산

        Args:
            sequences: 시퀀스 리스트

        Returns:
            각 시퀀스의 log-likelihood 배열
        """
        log_likelihoods = []

        for seq in sequences:
            X = np.array(seq).reshape(-1, 1)
            lengths = [len(seq)]
            log_prob = self.model.score(X, lengths)
            log_likelihoods.append(log_prob)

        return np.array(log_likelihoods)

    def set_threshold_percentile(self, val_sequences: List[List[int]],
                                 percentile: float = 5.0):
        """
        Validation set 기반 threshold 설정 (percentile 방식)

        Args:
            val_sequences: Validation 시퀀스 리스트
            percentile: 백분위 (default: 5.0 = 하위 5%)
        """
        logger.info(f"Computing threshold from {len(val_sequences)} "
                   f"validation sequences...")

        log_likelihoods = self.compute_log_likelihood(val_sequences)
        self.threshold = np.percentile(log_likelihoods, percentile)

        logger.info(f"Threshold set to {self.threshold:.4f} "
                   f"({percentile}th percentile)")

        # 통계 정보 출력
        logger.info(f"Validation log-likelihood stats:")
        logger.info(f"  Mean: {np.mean(log_likelihoods):.4f}")
        logger.info(f"  Std: {np.std(log_likelihoods):.4f}")
        logger.info(f"  Min: {np.min(log_likelihoods):.4f}")
        logger.info(f"  Max: {np.max(log_likelihoods):.4f}")

    def predict(self, sequences: List[List[int]]) -> np.ndarray:
        """
        시퀀스들을 정상/공격으로 분류

        Args:
            sequences: 테스트 시퀀스 리스트

        Returns:
            예측 레이블 (0: 정상, 1: 공격)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold_percentile first.")

        log_likelihoods = self.compute_log_likelihood(sequences)

        # Threshold보다 낮으면 공격(1), 높으면 정상(0)
        predictions = (log_likelihoods < self.threshold).astype(int)

        return predictions

    def predict_with_scores(self, sequences: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        예측 레이블과 log-likelihood 반환

        Args:
            sequences: 테스트 시퀀스 리스트

        Returns:
            (predictions, log_likelihoods)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold_percentile first.")

        log_likelihoods = self.compute_log_likelihood(sequences)
        predictions = (log_likelihoods < self.threshold).astype(int)

        return predictions, log_likelihoods

    def save_model(self, filepath: str):
        """
        모델 저장

        Args:
            filepath: 저장 경로
        """
        model_data = {
            'model': self.model,
            'threshold': self.threshold,
            'n_states': self.n_states,
            'n_observations': self.n_observations
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        모델 로드

        Args:
            filepath: 모델 파일 경로
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.threshold = model_data['threshold']
        self.n_states = model_data['n_states']
        self.n_observations = model_data['n_observations']

        logger.info(f"Model loaded from {filepath}")
