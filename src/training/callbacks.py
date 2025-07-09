from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint

from src import config


def get_callbacks(monitor: str = "val_loss") -> list[Callback]:
    return [
        # 모델 체크포인트 저장
        ModelCheckpoint(
            dirpath=config.CHECKPOINT_DIR,  # 체크포인트 파일을 저장할 디렉토리 경로
            filename="{epoch:02d}-{val_loss:.2f}",  # 체크포인트 파일명 패턴
            monitor=monitor,  # 모니터링할 메트릭 이름 (예: 'val_loss', 'val_accuracy')
            mode="min" if "loss" in monitor else "max",  # 메트릭 최적화 방향 ('min' 또는 'max')
            save_top_k=1,  # 저장할 최고 성능 모델 개수 (-1이면 모든 체크포인트 저장)
            save_last=True,  # 마지막 에포크 모델 저장 여부
            verbose=True,  # 로그 출력 여부
        ),
        # 조기 종료
        EarlyStopping(
            monitor=monitor,  # 모니터링할 메트릭 이름
            mode="min" if "loss" in monitor else "max",  # 메트릭 최적화 방향 ('min' 또는 'max')
            patience=2,  # 개선이 없어도 기다릴 에포크 수
            verbose=True,  # 로그 출력 여부
        ),
        # 학습률 모니터링
        LearningRateMonitor(logging_interval="epoch"),
    ]
