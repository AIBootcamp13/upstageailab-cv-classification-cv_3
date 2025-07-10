import os
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


load_dotenv()

TIMEZONE: ZoneInfo = ZoneInfo("Asia/Seoul")
_ROOT_MARKER = "pyproject.toml"


def find_project_root() -> Path:
    """
    pyproject.toml 파일을 기준으로 루트 디렉토리를 찾는다.
    :return: Path: 프로젝트 루트 디렉토리 경로
    """

    current_path = Path().resolve()

    while current_path != current_path.parent:
        if (current_path / _ROOT_MARKER).exists():
            return current_path

        current_path = current_path.parent

    raise FileNotFoundError("프로젝트 루트 디렉토리를 찾을 수 없습니다.")


def get_env_or_raise(key: str) -> str:
    env_value = os.getenv(key)
    if env_value is None:
        raise ValueError(f"{key} not set in environment variables")
    return env_value


# %% directory

ROOT_DIR = find_project_root()
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

PREDICTION_DIR = RAW_DATA_DIR / "prediction"
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = RAW_DATA_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

LOG_ROOT_DIR = ROOT_DIR / "log"
LOG_ROOT_DIR.mkdir(parents=True, exist_ok=True)


# %% filepath
TRAIN_META_CSV_PATH = RAW_DATA_DIR / "train.csv"
TRAIN_IMG_DIR = RAW_DATA_DIR / "train"

TEST_META_CSV_PATH = RAW_DATA_DIR / "sample_submission.csv"
TEST_IMG_DIR = RAW_DATA_DIR / "test"

CLASS_META_CSV_PATH = RAW_DATA_DIR / "meta.csv"
# %% 환경변수

# ncloud api key
NCLOUD_ACCESS_KEY = get_env_or_raise("NCLOUD_ACCESS_KEY")
NCLOUD_SECRET_KEY = get_env_or_raise("NCLOUD_SECRET_KEY")

# ncloud storage 정보
NCLOUD_STORAGE_REGION = get_env_or_raise("NCLOUD_STORAGE_REGION")
NCLOUD_STORAGE_ENDPOINT_URL = get_env_or_raise("NCLOUD_STORAGE_ENDPOINT_URL")
NCLOUD_STORAGE_BUCKET = get_env_or_raise("NCLOUD_STORAGE_BUCKET")
NCLOUD_STORAGE_PERSONAL_DIR = get_env_or_raise("NCLOUD_STORAGE_BUCKET_PERSONAL_DIR")

# Weights & Biases
WANDB_API_KEY = get_env_or_raise("WANDB_API_KEY")
WANDB_PROJECT = get_env_or_raise("WANDB_PROJECT")
WANDB_ENTITY = get_env_or_raise("WANDB_ENTITY")

# %%
TRAINED_MODEL_CONFIGS = [
    {
        "checkpoint_path": CHECKPOINT_DIR / "20250710125544_convnextv2_base_lr8e-05_ep80.ckpt",
        "model_name": "convnextv2_base",
        "weight": 0.65,
        "description": "convnextv2_base_lr8e-05_ep80",
    },
    {
        "checkpoint_path": CHECKPOINT_DIR / "convnextv2_base_lr4e-05_ep120.ckpt",
        "model_name": "convnextv2_base",
        "weight": 0.3,
        "description": "convnextv2_base_lr4e-05_ep120",
    },
    {
        "checkpoint_path": CHECKPOINT_DIR / "convnextv2_base_lr5e-05_ep150.ckpt",
        "model_name": "convnextv2_base",
        "weight": 0.2,
        "description": "convnextv2_tiny_lr0.0001_ep70",
    },
    {
        "checkpoint_path": CHECKPOINT_DIR / "convnextv2_base_lr6e-05_ep100.ckpt",
        "model_name": "convnextv2_base",
        "weight": 0.15,
        "description": "convnextv2_tiny_lr0.00015_ep50",
    },
]
