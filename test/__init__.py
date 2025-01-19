# src/test/__init__.py

# 필요한 모듈들 import
from .collector import BasketballShotCollector
from .preprocessor import ShootingFormPreprocessor
from .dataset import ShootingDataset
from .model import BasketballShootingAnalyzer

# 버전 정보
__version__ = '1.0.0'

# 외부에서 접근 가능한 클래스/함수 정의
__all__ = [
    'BasketballShotCollector',
    'ShootingFormPreprocessor',
    'ShootingDataset',
    'BasketballShootingAnalyzer',
]

# 기본 설정값들
DEFAULT_CONFIG = {
    'max_training_hours': 12,
    'max_videos_per_query': 50,
    'sequence_length': 90,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'num_epochs': 200
}

# 경로 설정
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
TEMP_DIR = os.path.join(BASE_DIR, 'temp')

# 필요한 디렉토리 생성
for directory in [MODELS_DIR, DATA_DIR, TEMP_DIR]:
    os.makedirs(directory, exist_ok=True)