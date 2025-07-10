import argparse

import pytorch_lightning as pl
import wandb
from torch import nn

from src.data.datamodules import DocumentImageDataModule
from src.data.datasets import DocumentImageSet
from src.model.classifier import DocumentImageClassifier
from src.training.callbacks import get_callbacks
from src.training.loggers import get_loggers
from src.transforms import create_train_test_transforms
from src.util.helper import add_timestamp_prefix, fix_random_seed
from src.util.log import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="문서 이미지 분류 모델 학습기")

    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="convnextv2_atto",
        help="timm 기준의 사전학습된 모델 이름(예시. convnextv2_tiny, convnextv2_base 등)",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="학습률")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay for optimizer")
    parser.add_argument("--drop-rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--drop-path-rate", type=float, default=0.1, help="Drop path rate (stochastic depth)")

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=16, help="학습한 데이터의 배치 사이즈")
    parser.add_argument("--epochs", type=int, default=100, help="학습한 epochs")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam"], help="Optimizer 선택")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "cosine_warm_restarts", "step", "exponential"],
        help="Learning rate scheduler 선택",
    )
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs")

    # Regularization arguments
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument("--mixup-alpha", type=float, default=0.2, help="MixUp alpha (0 to disable)")
    parser.add_argument("--cutmix-alpha", type=float, default=0.0, help="CutMix alpha (0 to disable)")

    # Data arguments
    parser.add_argument("--val-rate", type=float, default=0.2, help="검증 데이터 나누는 비율")
    parser.add_argument("--num-workers", type=int, default=4, help="데이터 로딩에 사용할 worker 개수")
    parser.add_argument("--pin-memory", action="store_true", help="데이터 로딩에 pin_memory 할 지")

    # Training arguments
    parser.add_argument("--seed", type=int, default=4321, help="랜덤 시드 설정")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="체크 포인트에서 시작할 경로")

    return parser.parse_args()


def create_experiment_name(model_name: str, learning_rate: float, batch_size: int, epochs: int) -> str:
    """wandb 실험 이름 생성"""
    original_name = f"{model_name}_lr{learning_rate}_bs{batch_size}_ep{epochs}"
    return add_timestamp_prefix(original_name)


def main():
    args = parse_args()

    experiment_name = create_experiment_name(args.model_name, args.learning_rate, args.batch_size, args.epochs)

    logger = get_logger(f"train-{experiment_name}")
    logger.info("-" * 80)
    logger.info("문서 이미지 분류기 학습")
    logger.info("-" * 80)

    # Random seed 설정
    fix_random_seed(args.seed)

    # 이미지 데이터셋의 transform 준비
    train_transform, test_transform = create_train_test_transforms(args.model_name)
    logger.info("이미지 데이터셋의 transform 준비 완료")

    # DataModule 준비
    data_module = DocumentImageDataModule(
        batch_size=args.batch_size,
        train_transform=train_transform,
        test_transform=test_transform,
        with_augmentation=True,
        val_rate=args.val_rate,
        random_seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    logger.info("DataModule 준비 완료")

    # 손실함수 정의
    criterion = (
        nn.CrossEntropyLoss(label_smoothing=args.label_smoothing) if args.label_smoothing else nn.CrossEntropyLoss()
    )

    # model 정의
    model = DocumentImageClassifier(
        model_name=args.model_name,
        num_classes=DocumentImageSet.calculate_metadata_classes(),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        criterion=criterion,
        pretrained=True,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
    )
    logger.info("model 정의 완료")

    # callback 준비
    callbacks = get_callbacks(experiment_name)

    # logger 준비
    loggers = get_loggers(experiment_name)

    # trainer 준비
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=loggers,
    )
    logger.info("trainer 정의 완료")

    # 학습!
    if args.checkpoint_path:
        logger.info("체크 포인트에서 학습 시작")
        trainer.fit(model, data_module, ckpt_path=args.checkpoint_path)
    else:
        logger.info("학습 시작")
        trainer.fit(model, data_module)

    wandb.finish()


if __name__ == "__main__":
    main()
