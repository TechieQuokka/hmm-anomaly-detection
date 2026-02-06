"""
실험 설정 파일
"""
from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path
import yaml


@dataclass
class ExperimentConfig:
    """HMM 이상탐지 실험 설정"""

    # ========================================
    # 데이터 설정
    # ========================================
    data_dir: str = 'adfa-ld/ADFA-LD'
    window_size: int = 500

    # ========================================
    # 모델 설정
    # ========================================
    n_states: int = 10
    n_observations: int = 92  # 시스템 콜 개수 (자동 계산됨)
    n_iter: int = 100  # Baum-Welch 최대 반복 횟수
    random_seed: int = 42

    # ========================================
    # 데이터 분할
    # ========================================
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    # ========================================
    # Threshold 설정
    # ========================================
    threshold_method: str = 'percentile'  # 'percentile', 'roc', 'adaptive'
    threshold_percentile: float = 5.0
    target_fpr: float = 0.05  # adaptive method용

    # ========================================
    # 출력 디렉토리
    # ========================================
    output_dir: str = 'results'
    model_dir: str = 'models'
    log_dir: str = 'logs'

    # ========================================
    # 실험 메타데이터
    # ========================================
    experiment_name: str = 'baseline'
    description: str = 'Baseline HMM experiment'

    def __post_init__(self):
        """검증 및 디렉토리 생성"""
        # 비율 검증
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

        # 출력 디렉토리 생성
        Path(self.output_dir).mkdir(exist_ok=True)
        Path(self.model_dir).mkdir(exist_ok=True)
        Path(self.log_dir).mkdir(exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'data_dir': self.data_dir,
            'window_size': self.window_size,
            'n_states': self.n_states,
            'n_observations': self.n_observations,
            'n_iter': self.n_iter,
            'random_seed': self.random_seed,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'threshold_method': self.threshold_method,
            'threshold_percentile': self.threshold_percentile,
            'target_fpr': self.target_fpr,
            'output_dir': self.output_dir,
            'model_dir': self.model_dir,
            'log_dir': self.log_dir,
            'experiment_name': self.experiment_name,
            'description': self.description
        }

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """YAML 파일에서 설정 로드"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def save_yaml(self, yaml_path: str):
        """설정을 YAML 파일로 저장"""
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def __str__(self) -> str:
        """문자열 표현"""
        lines = ["Experiment Configuration:"]
        lines.append("-" * 50)
        for key, value in self.to_dict().items():
            lines.append(f"  {key:25s}: {value}")
        lines.append("-" * 50)
        return "\n".join(lines)


# ========================================
# 사전 정의된 설정들
# ========================================

def get_baseline_config() -> ExperimentConfig:
    """기본 설정 (현재 설정)"""
    return ExperimentConfig(
        experiment_name='baseline',
        description='Original baseline configuration',
        window_size=500,
        n_states=10,
        threshold_percentile=5.0
    )


def get_improved_config() -> ExperimentConfig:
    """개선된 설정 (Phase 1)"""
    return ExperimentConfig(
        experiment_name='phase1_improved',
        description='Phase 1: Improved threshold and states',
        window_size=500,
        n_states=15,  # 10 → 15
        threshold_percentile=15.0  # 5.0 → 15.0
    )


def get_window_300_config() -> ExperimentConfig:
    """Window size 300 실험"""
    return ExperimentConfig(
        experiment_name='window_300',
        description='Experiment with window_size=300',
        window_size=300,
        n_states=15,
        threshold_percentile=15.0
    )


def get_window_700_config() -> ExperimentConfig:
    """Window size 700 실험"""
    return ExperimentConfig(
        experiment_name='window_700',
        description='Experiment with window_size=700',
        window_size=700,
        n_states=15,
        threshold_percentile=15.0
    )


def get_config_by_name(name: str) -> ExperimentConfig:
    """이름으로 설정 가져오기"""
    configs = {
        'baseline': get_baseline_config,
        'improved': get_improved_config,
        'window_300': get_window_300_config,
        'window_700': get_window_700_config
    }

    if name not in configs:
        raise ValueError(f"Unknown config name: {name}. "
                        f"Available: {list(configs.keys())}")

    return configs[name]()


# ========================================
# 사용 예시
# ========================================
if __name__ == "__main__":
    # 기본 설정 출력
    print("=" * 60)
    print("Available Configurations")
    print("=" * 60)

    for name in ['baseline', 'improved', 'window_300', 'window_700']:
        config = get_config_by_name(name)
        print(f"\n{config.experiment_name.upper()}:")
        print(f"  Window Size: {config.window_size}")
        print(f"  Hidden States: {config.n_states}")
        print(f"  Threshold Percentile: {config.threshold_percentile}%")
        print(f"  Description: {config.description}")

    print("\n" + "=" * 60)
    print("\nTo use a config in main.py:")
    print("  from config import get_config_by_name")
    print("  config = get_config_by_name('improved')")
