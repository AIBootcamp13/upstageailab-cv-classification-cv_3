import argparse

import pandas as pd

from src import config
from src.inference.ensemble import EnsemblePredictor
from src.util.helper import add_timestamp_prefix, fix_random_seed
from src.util.log import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="문서 이미지 분류 앙상블 예측기")

    # 앙상블 설정
    parser.add_argument(
        "--ensemble-method",
        type=str,
        default="hard_voting",
        choices=["soft_voting", "hard_voting", "weighted_average"],
        help="앙상블 방법",
    )
    # 데이터 로딩 설정
    parser.add_argument("--batch-size", type=int, default=32, help="예측 배치 사이즈 (앙상블시 작게 설정 권장)")
    parser.add_argument("--num-workers", type=int, default=4, help="데이터 로딩 worker 수")
    parser.add_argument("--pin-memory", action="store_true", default=False, help="pin_memory 사용")

    # 기타
    parser.add_argument("--seed", type=int, default=4321, help="랜덤 시드")
    parser.add_argument("--output-suffix", type=str, default="ensemble", help="출력 파일명 접미사")

    return parser.parse_args()


def main():
    args = parse_args()

    logger = get_logger("ensemble_predict")
    logger.info("=" * 80)
    logger.info("문서 이미지 분류 앙상블 예측기")
    logger.info("=" * 80)

    # 랜덤 시드 설정
    if args.seed is not None:
        fix_random_seed(args.seed)

    # 모델 설정 로드
    logger.info("모델 설정")
    logger.info(f"총 {len(config.TRAINED_MODEL_CONFIGS)}개 모델 앙상블")

    # 설정 로그
    for i, model_config in enumerate(config.TRAINED_MODEL_CONFIGS):
        logger.info(f"  모델 {i + 1}: {model_config['checkpoint_path']} (가중치: {model_config.get('weight', 1.0)})")

    # 앙상블 예측기 생성
    ensemble_predictor = EnsemblePredictor(
        model_configs=config.TRAINED_MODEL_CONFIGS,
        ensemble_method=args.ensemble_method,
    )

    # 모델 로드
    ensemble_predictor.load_models()

    # 앙상블 예측 수행
    logger.info(f"앙상블 방법: {args.ensemble_method}")
    predictions = ensemble_predictor.ensemble_predict(
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory
    )

    # 결과 저장
    test_meta_df = pd.read_csv(config.TEST_META_CSV_PATH)
    result_df = pd.DataFrame({"ID": test_meta_df["ID"], "target": predictions.astype(int)})

    # 출력 파일명 생성
    output_filename = f"submission_{args.output_suffix}_{args.ensemble_method}.csv"

    output_path = config.PREDICTION_DIR / add_timestamp_prefix(output_filename)
    result_df.to_csv(output_path, index=False)

    logger.info(f"앙상블 예측 결과 저장: {output_path}")

    # 결과 요약
    logger.info(f"총 예측 샘플 수: {len(result_df)}")
    logger.info("앙상블 예측 클래스 분포:")
    for class_id, count in result_df["target"].value_counts().sort_index().items():
        logger.info(f"  클래스 {class_id}: {count}개")

    # 결과 미리보기
    logger.info("앙상블 예측 결과 미리보기:")
    logger.info(result_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
