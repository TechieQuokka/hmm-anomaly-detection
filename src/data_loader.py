"""
ADFA-LD 데이터 로더 및 전처리 모듈
"""
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ADFADataLoader:
    """ADFA-LD 데이터셋 로더"""

    def __init__(self, data_dir: str, window_size: int = 200):
        """
        Args:
            data_dir: ADFA-LD 데이터셋 루트 디렉토리
            window_size: 추출할 윈도우 크기 (default: 200)
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size

        # 디렉토리 경로 설정
        self.train_dir = self.data_dir / "Training_Data_Master"
        self.attack_dir = self.data_dir / "Attack_Data_Master"
        self.validation_dir = self.data_dir / "Validation_Data_Master"

    def load_sequence_file(self, file_path: Path) -> List[int]:
        """
        단일 시퀀스 파일 로드

        Args:
            file_path: 시퀀스 파일 경로

        Returns:
            시스템 호출 ID 리스트
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
                sequence = [int(x) for x in content.split()]
            return sequence
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            return []

    def extract_window(self, sequence: List[int]) -> List[int]:
        """
        시퀀스에서 첫 window_size 개 추출

        Args:
            sequence: 시스템 호출 시퀀스

        Returns:
            추출된 윈도우 (길이가 window_size 미만이면 None)
        """
        if len(sequence) < self.window_size:
            return None
        return sequence[:self.window_size]

    def load_normal_data(self, directory: Path) -> List[List[int]]:
        """
        정상 데이터 로드

        Args:
            directory: 정상 데이터 디렉토리

        Returns:
            윈도우 리스트
        """
        windows = []
        files = sorted(directory.glob("*.txt"))

        logger.info(f"Loading from {directory}")

        for file_path in files:
            sequence = self.load_sequence_file(file_path)
            if not sequence:
                continue

            window = self.extract_window(sequence)
            if window is not None:
                windows.append(window)

        logger.info(f"Loaded {len(windows)} sequences (window_size={self.window_size})")
        return windows

    def load_attack_data(self) -> Dict[str, List[List[int]]]:
        """
        공격 데이터 로드 (유형별로 분류)

        Returns:
            {attack_type: [windows]} 딕셔너리
        """
        attack_types = [
            'Adduser', 'Hydra_FTP', 'Hydra_SSH',
            'Java_Meterpreter', 'Meterpreter', 'Web_Shell'
        ]

        attack_data = {}

        for attack_type in attack_types:
            windows = []

            # 각 공격 유형은 1~10번 폴더로 구성
            for i in range(1, 11):
                attack_folder = self.attack_dir / f"{attack_type}_{i}"

                if not attack_folder.exists():
                    continue

                # 폴더 내 모든 파일 로드
                for file_path in attack_folder.glob("*.txt"):
                    sequence = self.load_sequence_file(file_path)
                    if not sequence:
                        continue

                    window = self.extract_window(sequence)
                    if window is not None:
                        windows.append(window)

            attack_data[attack_type] = windows
            logger.info(f"Loaded {attack_type}: {len(windows)} sequences")

        return attack_data

    def split_data(self, sequences: List[List[int]],
                   train_ratio: float = 0.6,
                   val_ratio: float = 0.2,
                   test_ratio: float = 0.2,
                   random_seed: int = 42) -> Tuple[List, List, List]:
        """
        데이터를 Train/Validation/Test로 분할

        Args:
            sequences: 전체 시퀀스 리스트
            train_ratio: Train 비율
            val_ratio: Validation 비율
            test_ratio: Test 비율
            random_seed: 랜덤 시드

        Returns:
            (train_sequences, val_sequences, test_sequences)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"

        np.random.seed(random_seed)
        indices = np.random.permutation(len(sequences))

        n_train = int(len(sequences) * train_ratio)
        n_val = int(len(sequences) * val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train_seqs = [sequences[i] for i in train_idx]
        val_seqs = [sequences[i] for i in val_idx]
        test_seqs = [sequences[i] for i in test_idx]

        logger.info(f"Data split - Train: {len(train_seqs)}, "
                   f"Val: {len(val_seqs)}, Test: {len(test_seqs)}")

        return train_seqs, val_seqs, test_seqs

    def get_vocabulary_size(self, sequences: List[List[int]]) -> int:
        """
        시퀀스들에서 고유 시스템 호출 개수 계산

        Args:
            sequences: 시퀀스 리스트

        Returns:
            고유 시스템 호출 개수
        """
        unique_syscalls = set()
        for seq in sequences:
            unique_syscalls.update(seq)
        return len(unique_syscalls)

    def get_syscall_mapping(self, sequences: List[List[int]]) -> Dict[int, int]:
        """
        시스템 호출 ID를 연속된 인덱스로 매핑

        Args:
            sequences: 시퀀스 리스트

        Returns:
            {original_id: mapped_index} 딕셔너리
        """
        unique_syscalls = set()
        for seq in sequences:
            unique_syscalls.update(seq)

        # ID 정렬 후 0부터 시작하는 인덱스 부여
        sorted_syscalls = sorted(unique_syscalls)
        mapping = {syscall: idx for idx, syscall in enumerate(sorted_syscalls)}

        logger.info(f"Created mapping for {len(mapping)} unique system calls")
        return mapping

    def apply_mapping(self, sequences: List[List[int]],
                     mapping: Dict[int, int]) -> List[List[int]]:
        """
        시퀀스에 매핑 적용

        Args:
            sequences: 원본 시퀀스 리스트
            mapping: 시스템 호출 매핑

        Returns:
            매핑된 시퀀스 리스트
        """
        mapped_sequences = []
        for seq in sequences:
            mapped_seq = [mapping.get(syscall, 0) for syscall in seq]
            mapped_sequences.append(mapped_seq)
        return mapped_sequences
