"""
Hybrid HMM + Random Forest 모델
"""
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging

# 로컬 모듈
from hmm_model import AnomalyDetectorHMM
from feature_extractor import StatisticalFeatureExtractor, NGramFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridHMMClassifier:
    """HMM + Random Forest 하이브리드 분류기"""

    def __init__(self,
                 n_states: int = 15,
                 n_observations: int = 92,
                 random_state: int = 42,
                 use_ngrams: bool = True,
                 n_estimators: int = 100):
        """
        Args:
            n_states: HMM hidden states 개수
            n_observations: 관측 심볼 개수
            random_state: 랜덤 시드
            use_ngrams: N-gram 특징 사용 여부
            n_estimators: Random Forest 트리 개수
        """
        self.n_states = n_states
        self.n_observations = n_observations
        self.random_state = random_state
        self.use_ngrams = use_ngrams

        # HMM 모델
        self.hmm_model = AnomalyDetectorHMM(
            n_states=n_states,
            n_observations=n_observations,
            random_state=random_state
        )

        # 특징 추출기
        self.stat_extractor = StatisticalFeatureExtractor()
        self.ngram_extractor = NGramFeatureExtractor(
            n_values=[2, 3, 4],
            max_features=50
        ) if use_ngrams else None

        # Random Forest 분류기
        self.rf_classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,  # 모든 코어 사용
            class_weight='balanced'  # 불균형 데이터 처리
        )

        # 특징 스케일러
        self.scaler = StandardScaler()

        # 특징 차원 정보
        self.n_features_ = 0
        self.feature_names_ = []

    def fit(self,
            normal_sequences: List[List[int]],
            attack_sequences: List[List[int]],
            val_normal_sequences: List[List[int]] = None):
        """
        학습

        Args:
            normal_sequences: 정상 시퀀스 (학습용)
            attack_sequences: 공격 시퀀스 (학습용)
            val_normal_sequences: Validation 정상 시퀀스 (HMM threshold용)
        """
        logger.info("=" * 60)
        logger.info("Training Hybrid HMM + Random Forest Classifier")
        logger.info("=" * 60)

        # 1. HMM 학습 (정상 데이터만)
        logger.info("\n[1/4] Training HMM on normal data...")
        self.hmm_model.fit(normal_sequences)

        # Threshold 설정 (validation 데이터 있으면 사용, 없으면 train 데이터)
        if val_normal_sequences and len(val_normal_sequences) > 0:
            self.hmm_model.set_threshold_percentile(val_normal_sequences, percentile=50.0)
            logger.info("HMM threshold set using validation data")
        else:
            self.hmm_model.set_threshold_percentile(normal_sequences, percentile=50.0)
            logger.info("HMM threshold set using training data")

        # 2. N-gram vocabulary 구축 (사용하는 경우)
        if self.use_ngrams:
            logger.info("\n[2/4] Building N-gram vocabulary...")
            self.ngram_extractor.fit(normal_sequences + attack_sequences)
            logger.info(f"N-gram vocabulary size: {len(self.ngram_extractor.vocabulary_)}")

        # 3. 특징 추출
        logger.info("\n[3/4] Extracting features from training data...")

        # 정상 데이터 특징
        X_normal = self._extract_hybrid_features(normal_sequences)
        y_normal = np.zeros(len(normal_sequences))

        # 공격 데이터 특징
        X_attack = self._extract_hybrid_features(attack_sequences)
        y_attack = np.ones(len(attack_sequences))

        # 결합
        X_train = np.vstack([X_normal, X_attack])
        y_train = np.concatenate([y_normal, y_attack])

        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"  Normal samples: {len(X_normal)}")
        logger.info(f"  Attack samples: {len(X_attack)}")

        # 특징 이름 저장
        self._build_feature_names()

        # 4. 특징 스케일링 및 RF 학습
        logger.info("\n[4/4] Training Random Forest classifier...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.rf_classifier.fit(X_train_scaled, y_train)

        logger.info("Hybrid model training completed!")

        # 특징 중요도 출력
        self._print_feature_importance()

    def _extract_hybrid_features(self, sequences: List[List[int]]) -> np.ndarray:
        """
        하이브리드 특징 추출 (HMM + 통계 + N-gram)

        Args:
            sequences: 시퀀스 리스트

        Returns:
            특징 행렬
        """
        features_list = []

        for seq in sequences:
            # 1. HMM log-likelihood
            hmm_score = self.hmm_model.compute_log_likelihood([seq])[0]

            # 2. 통계 특징
            stat_features = self.stat_extractor.extract_features(seq)

            # 3. N-gram 특징 (사용하는 경우)
            if self.use_ngrams:
                ngram_features = self.ngram_extractor.transform([seq])[0]
                combined = np.concatenate([[hmm_score], stat_features, ngram_features])
            else:
                combined = np.concatenate([[hmm_score], stat_features])

            features_list.append(combined)

        return np.array(features_list)

    def _build_feature_names(self):
        """특징 이름 구축"""
        self.feature_names_ = ['hmm_log_likelihood']
        self.feature_names_.extend(self.stat_extractor.get_feature_names())

        if self.use_ngrams:
            self.feature_names_.extend(self.ngram_extractor.get_feature_names())

        self.n_features_ = len(self.feature_names_)
        logger.info(f"Total features: {self.n_features_}")

    def _print_feature_importance(self, top_k: int = 15):
        """특징 중요도 출력"""
        importances = self.rf_classifier.feature_importances_
        indices = np.argsort(importances)[::-1]

        logger.info(f"\nTop {top_k} Feature Importances:")
        for i in range(min(top_k, len(indices))):
            idx = indices[i]
            logger.info(f"  {i+1}. {self.feature_names_[idx]:40s}: {importances[idx]:.4f}")

    def predict(self, sequences: List[List[int]]) -> np.ndarray:
        """
        예측

        Args:
            sequences: 테스트 시퀀스

        Returns:
            예측 레이블 (0: 정상, 1: 공격)
        """
        X_test = self._extract_hybrid_features(sequences)
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.rf_classifier.predict(X_test_scaled)
        return predictions

    def predict_proba(self, sequences: List[List[int]]) -> np.ndarray:
        """
        예측 확률

        Args:
            sequences: 테스트 시퀀스

        Returns:
            예측 확률 (n_samples, 2)
        """
        X_test = self._extract_hybrid_features(sequences)
        X_test_scaled = self.scaler.transform(X_test)
        probabilities = self.rf_classifier.predict_proba(X_test_scaled)
        return probabilities

    def predict_with_scores(self, sequences: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        예측 레이블과 확률 반환

        Args:
            sequences: 테스트 시퀀스

        Returns:
            (predictions, probabilities)
        """
        probabilities = self.predict_proba(sequences)
        predictions = (probabilities[:, 1] > 0.5).astype(int)
        return predictions, probabilities[:, 1]

    def save_model(self, filepath: str):
        """
        모델 저장

        Args:
            filepath: 저장 경로
        """
        model_data = {
            'hmm_model': self.hmm_model,
            'stat_extractor': self.stat_extractor,
            'ngram_extractor': self.ngram_extractor,
            'rf_classifier': self.rf_classifier,
            'scaler': self.scaler,
            'n_states': self.n_states,
            'n_observations': self.n_observations,
            'use_ngrams': self.use_ngrams,
            'n_features': self.n_features_,
            'feature_names': self.feature_names_
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Hybrid model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        모델 로드

        Args:
            filepath: 모델 파일 경로
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.hmm_model = model_data['hmm_model']
        self.stat_extractor = model_data['stat_extractor']
        self.ngram_extractor = model_data['ngram_extractor']
        self.rf_classifier = model_data['rf_classifier']
        self.scaler = model_data['scaler']
        self.n_states = model_data['n_states']
        self.n_observations = model_data['n_observations']
        self.use_ngrams = model_data['use_ngrams']
        self.n_features_ = model_data['n_features']
        self.feature_names_ = model_data['feature_names']

        logger.info(f"Hybrid model loaded from {filepath}")


if __name__ == "__main__":
    # 간단한 테스트
    print("=" * 60)
    print("Hybrid HMM+RF Classifier Test")
    print("=" * 60)

    # 더미 데이터
    normal_seqs = [[1, 2, 3, 2, 1] * 20 for _ in range(50)]
    attack_seqs = [[10, 11, 12, 13, 14] * 20 for _ in range(30)]
    test_seqs = [[1, 2, 3] * 20, [10, 11, 12] * 20]

    # 모델 생성 및 학습
    hybrid = HybridHMMClassifier(n_states=5, n_observations=15, use_ngrams=True)
    hybrid.fit(normal_seqs, attack_seqs)

    # 예측
    predictions, scores = hybrid.predict_with_scores(test_seqs)
    print(f"\nPredictions: {predictions}")
    print(f"Scores: {scores}")

    print("\n" + "=" * 60)
    print("Test Completed!")
