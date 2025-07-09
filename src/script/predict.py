import argparse

import pandas as pd
import pytorch_lightning as pl

from src import config
from src.data.datamodules import DocumentImageDataModule
from src.model.classifier import DocumentImageClassifier
from src.transforms import create_train_test_transforms
from src.util.helper import add_timestamp_prefix, fix_random_seed
from src.util.log import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="문서 이미지 분류 모델 예측기")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="학습된 모델 체크포인트 경로")
    parser.add_argument("--batch-size", type=int, default=64, help="예측 배치 사이즈")
    parser.add_argument("--num-workers", type=int, default=4, help="데이터 로딩에 사용할 worker 개수")
    parser.add_argument("--pin-memory", type=bool, default=False, help="데이터 로딩에 pin_memory 사용")
    parser.add_argument("--seed", type=int, default=4321, help="랜덤 시드 설정")

    return parser.parse_args()


def main():
    args = parse_args()

    logger = get_logger("predict")
    logger.info("-" * 80)
    logger.info("문서 이미지 분류기 예측")
    logger.info("-" * 80)

    # 랜덤 시드 설정
    if args.seed is not None:
        fix_random_seed(args.seed)

    # 체크포인트에서 모델 로드
    logger.info(f"모델 로드: {args.checkpoint_path}")
    model = DocumentImageClassifier.load_from_checkpoint(args.checkpoint_path)
    logger.info("모델 로드 완료")

    # 모델 이름 추출 (하이퍼파라미터에서)
    model_name = model.hparams.model_name

    # 변환 생성
    _, test_transform = create_train_test_transforms(model_name)
    logger.info("이미지 변환 준비 완료")

    # DataModule 생성
    data_module = DocumentImageDataModule(
        batch_size=args.batch_size,
        train_transform=test_transform,  # predict 에서는 test_transform 사용
        test_transform=test_transform,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # predict stage 로 setup
    data_module.setup(stage="predict")
    logger.info("DataModule 준비 완료")

    # Trainer 생성
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    # 예측 수행
    logger.info("예측 시작")
    predictions = trainer.predict(model, data_module.predict_dataloader())
    logger.info("예측 완료")

    # 예측 결과 처리
    all_predictions = []
    for batch_predictions in predictions:
        all_predictions.extend(batch_predictions.cpu().numpy())

    # 원본 테스트 메타데이터에서 ID 가져오기
    test_meta_df = pd.read_csv(config.TEST_META_CSV_PATH)

    # 결과 DataFrame 생성
    result_df = pd.DataFrame({"ID": test_meta_df["ID"], "target": all_predictions})

    # 결과 저장
    original_name = f"{model_name}_lr{model.hparams.learning_rate}_ep{model.hparams.max_epochs}"
    output_filename = add_timestamp_prefix(f"{original_name}_submission.csv")
    output_path = config.PREDICTION_DIR / output_filename
    result_df.to_csv(output_path, index=False)
    logger.info(f"예측 결과 저장 완료: {output_path}")

    # 예측 결과 요약
    logger.info(f"총 예측 샘플 수: {len(result_df)}")
    logger.info("예측 클래스 분포:")
    for class_id, count in result_df["target"].value_counts().sort_index().items():
        logger.info(f"  클래스 {class_id}: {count}개")

    # 결과 미리보기
    logger.info("예측 결과 미리보기:")
    logger.info(result_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
