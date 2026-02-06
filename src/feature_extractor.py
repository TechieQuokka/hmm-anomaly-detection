"""
통계적 특징 추출 모듈
"""
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalFeatureExtractor:
    """시스템 콜 시퀀스에서 통계적 특징 추출"""

    def __init__(self):
        """초기화"""
        self.feature_names = []
        self._build_feature_names()

    def _build_feature_names(self):
        """특징 이름 정의"""
        self.feature_names = [
            # 기본 통계 (5개)
            'unique_syscalls',
            'entropy',
            'mean_syscall',
            'std_syscall',
            'median_syscall',

            # 빈도 기반 (4개)
            'most_common_ratio',
            'rare_syscalls_ratio',
            'top3_concentration',
            'frequency_variance',

            # Transition 기반 (5개)
            'unique_transitions',
            'transition_entropy',
            'self_transition_ratio',
            'transition_diversity',
            'avg_transition_distance',

            # 패턴 기반 (4개)
            'repeated_bigrams',
            'repeated_trigrams',
            'max_run_length',
            'pattern_complexity'
        ]

    def extract_features(self, sequence: List[int]) -> np.ndarray:
        """
        시퀀스에서 모든 통계적 특징 추출

        Args:
            sequence: 시스템 콜 시퀀스

        Returns:
            특징 벡터 (18차원)
        """
        if len(sequence) == 0:
            return np.zeros(len(self.feature_names))

        seq_array = np.array(sequence)

        features = []

        # 1. 기본 통계
        features.extend(self._extract_basic_stats(seq_array))

        # 2. 빈도 기반
        features.extend(self._extract_frequency_features(seq_array))

        # 3. Transition 기반
        features.extend(self._extract_transition_features(seq_array))

        # 4. 패턴 기반
        features.extend(self._extract_pattern_features(seq_array))

        return np.array(features)

    def _extract_basic_stats(self, sequence: np.ndarray) -> List[float]:
        """
        기본 통계 특징 (5개)

        Returns:
            [unique_syscalls, entropy, mean, std, median]
        """
        # 1. 고유 시스템 콜 개수
        unique_count = len(np.unique(sequence))

        # 2. Shannon entropy
        entropy = self._compute_entropy(sequence)

        # 3-5. 평균, 표준편차, 중앙값
        mean_val = np.mean(sequence)
        std_val = np.std(sequence)
        median_val = np.median(sequence)

        return [
            float(unique_count),
            float(entropy),
            float(mean_val),
            float(std_val),
            float(median_val)
        ]

    def _extract_frequency_features(self, sequence: np.ndarray) -> List[float]:
        """
        빈도 기반 특징 (4개)

        Returns:
            [most_common_ratio, rare_ratio, top3_concentration, frequency_variance]
        """
        counter = Counter(sequence)
        total = len(sequence)
        frequencies = np.array(list(counter.values()))

        # 1. 가장 많이 등장한 시스템 콜 비율
        most_common_ratio = float(max(frequencies)) / total

        # 2. 희귀 시스템 콜 비율 (1-2번만 등장)
        rare_count = sum(1 for freq in frequencies if freq <= 2)
        rare_ratio = float(rare_count) / len(frequencies) if len(frequencies) > 0 else 0.0

        # 3. 상위 3개 시스템 콜 집중도
        top3 = sorted(frequencies, reverse=True)[:3]
        top3_concentration = float(sum(top3)) / total

        # 4. 빈도 분산 (분포의 불균등성)
        frequency_variance = float(np.var(frequencies))

        return [
            most_common_ratio,
            rare_ratio,
            top3_concentration,
            frequency_variance
        ]

    def _extract_transition_features(self, sequence: np.ndarray) -> List[float]:
        """
        Transition 기반 특징 (5개)

        Returns:
            [unique_transitions, transition_entropy, self_transition_ratio,
             transition_diversity, avg_transition_distance]
        """
        if len(sequence) < 2:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        # Transition 생성
        transitions = [(sequence[i], sequence[i+1]) for i in range(len(sequence)-1)]

        # 1. 고유 transition 개수
        unique_transitions = len(set(transitions))

        # 2. Transition entropy
        transition_counter = Counter(transitions)
        transition_probs = np.array(list(transition_counter.values())) / len(transitions)
        transition_entropy = -np.sum(transition_probs * np.log2(transition_probs + 1e-10))

        # 3. Self-transition 비율 (같은 시스템 콜 연속)
        self_transitions = sum(1 for i in range(len(sequence)-1) if sequence[i] == sequence[i+1])
        self_transition_ratio = float(self_transitions) / (len(sequence) - 1)

        # 4. Transition diversity (normalized unique transitions)
        max_possible_transitions = len(np.unique(sequence)) ** 2
        transition_diversity = float(unique_transitions) / max_possible_transitions if max_possible_transitions > 0 else 0.0

        # 5. 평균 transition 거리 (시스템 콜 ID 차이)
        transition_distances = [abs(sequence[i+1] - sequence[i]) for i in range(len(sequence)-1)]
        avg_transition_distance = float(np.mean(transition_distances))

        return [
            float(unique_transitions),
            float(transition_entropy),
            self_transition_ratio,
            transition_diversity,
            avg_transition_distance
        ]

    def _extract_pattern_features(self, sequence: np.ndarray) -> List[float]:
        """
        패턴 기반 특징 (4개)

        Returns:
            [repeated_bigrams, repeated_trigrams, max_run_length, pattern_complexity]
        """
        # 1. 반복되는 bigram 수
        bigrams = [tuple(sequence[i:i+2]) for i in range(len(sequence)-1)]
        bigram_counts = Counter(bigrams)
        repeated_bigrams = sum(1 for count in bigram_counts.values() if count > 1)

        # 2. 반복되는 trigram 수
        if len(sequence) >= 3:
            trigrams = [tuple(sequence[i:i+3]) for i in range(len(sequence)-2)]
            trigram_counts = Counter(trigrams)
            repeated_trigrams = sum(1 for count in trigram_counts.values() if count > 1)
        else:
            repeated_trigrams = 0

        # 3. 최대 run length (같은 값이 연속으로 나타나는 최대 길이)
        max_run_length = 1
        current_run = 1
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i-1]:
                current_run += 1
                max_run_length = max(max_run_length, current_run)
            else:
                current_run = 1

        # 4. Pattern complexity (Lempel-Ziv complexity의 간단한 버전)
        # 서로 다른 subsequence의 개수로 근사
        unique_patterns = set()
        for length in [2, 3, 4]:
            if len(sequence) >= length:
                for i in range(len(sequence) - length + 1):
                    pattern = tuple(sequence[i:i+length])
                    unique_patterns.add(pattern)
        pattern_complexity = len(unique_patterns)

        return [
            float(repeated_bigrams),
            float(repeated_trigrams),
            float(max_run_length),
            float(pattern_complexity)
        ]

    def _compute_entropy(self, sequence: np.ndarray) -> float:
        """
        Shannon entropy 계산

        Args:
            sequence: 시퀀스

        Returns:
            Entropy 값
        """
        _, counts = np.unique(sequence, return_counts=True)
        probabilities = counts / len(sequence)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return float(entropy)

    def extract_features_batch(self, sequences: List[List[int]]) -> np.ndarray:
        """
        여러 시퀀스에서 특징 추출 (배치 처리)

        Args:
            sequences: 시퀀스 리스트

        Returns:
            특징 행렬 (n_sequences, n_features)
        """
        features_list = []
        for seq in sequences:
            features = self.extract_features(seq)
            features_list.append(features)

        return np.array(features_list)

    def get_feature_names(self) -> List[str]:
        """특징 이름 반환"""
        return self.feature_names.copy()


class NGramFeatureExtractor:
    """N-gram 특징 추출기"""

    def __init__(self, n_values: List[int] = [2, 3, 4],
                 max_features: int = 100):
        """
        Args:
            n_values: N-gram 크기 리스트
            max_features: 최대 특징 개수 (상위 빈도 n-gram만 사용)
        """
        self.n_values = n_values
        self.max_features = max_features
        self.vocabulary_ = {}  # n-gram → index
        self.feature_names_ = []

    def fit(self, sequences: List[List[int]]) -> 'NGramFeatureExtractor':
        """
        학습 데이터에서 n-gram vocabulary 구축

        Args:
            sequences: 학습 시퀀스 리스트

        Returns:
            self
        """
        # 모든 n-gram 수집
        ngram_counter = Counter()

        for seq in sequences:
            for n in self.n_values:
                if len(seq) >= n:
                    for i in range(len(seq) - n + 1):
                        ngram = tuple(seq[i:i+n])
                        ngram_counter[ngram] += 1

        # 상위 max_features개 선택
        most_common = ngram_counter.most_common(self.max_features)

        self.vocabulary_ = {ngram: idx for idx, (ngram, _) in enumerate(most_common)}
        self.feature_names_ = [f"ngram_{ngram}" for ngram, _ in most_common]

        logger.info(f"Built n-gram vocabulary: {len(self.vocabulary_)} features")

        return self

    def transform(self, sequences: List[List[int]]) -> np.ndarray:
        """
        시퀀스를 n-gram 특징 벡터로 변환

        Args:
            sequences: 시퀀스 리스트

        Returns:
            특징 행렬 (n_sequences, n_features)
        """
        n_samples = len(sequences)
        n_features = len(self.vocabulary_)

        X = np.zeros((n_samples, n_features))

        for i, seq in enumerate(sequences):
            ngram_counts = Counter()

            for n in self.n_values:
                if len(seq) >= n:
                    for j in range(len(seq) - n + 1):
                        ngram = tuple(seq[j:j+n])
                        if ngram in self.vocabulary_:
                            ngram_counts[ngram] += 1

            # TF (Term Frequency) 정규화
            for ngram, count in ngram_counts.items():
                idx = self.vocabulary_[ngram]
                X[i, idx] = count / len(seq)  # 시퀀스 길이로 정규화

        return X

    def fit_transform(self, sequences: List[List[int]]) -> np.ndarray:
        """Fit and transform"""
        return self.fit(sequences).transform(sequences)

    def get_feature_names(self) -> List[str]:
        """특징 이름 반환"""
        return self.feature_names_.copy()


if __name__ == "__main__":
    # 테스트
    print("=" * 60)
    print("Feature Extractor Test")
    print("=" * 60)

    # 테스트 시퀀스
    test_sequences = [
        [1, 2, 3, 2, 1, 4, 5, 4, 3, 2],
        [10, 10, 11, 12, 11, 10, 13, 14, 15],
        [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    ]

    # 통계적 특징 추출
    stat_extractor = StatisticalFeatureExtractor()
    print(f"\nStatistical Features: {len(stat_extractor.get_feature_names())}")
    print("Feature names:", stat_extractor.get_feature_names())

    for i, seq in enumerate(test_sequences):
        features = stat_extractor.extract_features(seq)
        print(f"\nSequence {i+1}: {seq}")
        print(f"Features: {features[:5]}...")  # 처음 5개만 출력

    # N-gram 특징 추출
    print("\n" + "=" * 60)
    ngram_extractor = NGramFeatureExtractor(n_values=[2, 3], max_features=20)
    ngram_features = ngram_extractor.fit_transform(test_sequences)
    print(f"\nN-gram Features: {ngram_features.shape}")
    print(f"Vocabulary size: {len(ngram_extractor.vocabulary_)}")

    print("\n" + "=" * 60)
    print("Feature Extractor Test Completed!")
