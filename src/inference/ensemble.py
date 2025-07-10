import numpy as np
import pytorch_lightning as pl

from src.data.datamodules import DocumentImageDataModule
from src.model.classifier import DocumentImageClassifier
from src.transforms import create_train_test_transforms
from src.util.log import get_logger


class EnsemblePredictor:
    """다중 모델 앙상블 예측기"""

    def __init__(self, model_configs: list[dict], ensemble_method: str = "soft_voting"):
        """
        Args:
            model_configs: 모델 설정 리스트
                [{'checkpoint_path': str, 'weight': float, 'model_name': str}, ...]
            ensemble_method: 'soft_voting', 'hard_voting', 'weighted_average'
        """
        self.model_configs = model_configs
        self.ensemble_method = ensemble_method
        self.models = []
        self.trainer = pl.Trainer(
            accelerator="auto",
            devices="auto",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            enable_model_summary=False,
        )

        self.logger = get_logger("ensemble_predictor")

    def load_models(self):
        """모든 모델 로드"""
        self.logger.info(f"총 {len(self.model_configs)}개 모델 로드 중...")

        for i, model_config in enumerate(self.model_configs):
            checkpoint_path = model_config["checkpoint_path"]
            self.logger.info(f"모델 {i + 1} 로드: {checkpoint_path}")

            model = DocumentImageClassifier.load_from_checkpoint(checkpoint_path)
            model.eval()
            self.models.append(model)

        self.logger.info("모든 모델 로드 완료")

    def ensemble_predict(self, batch_size: int = 64, num_workers: int = 4, pin_memory: bool = False) -> np.ndarray:
        """앙상블 예측 수행"""
        self.logger.info("앙상블 예측 시작")

        # 모든 모델의 예측 수집
        all_model_predictions = []
        model_weights = []

        # 모델마다 DataModule 생성
        for i, (model, model_config) in enumerate(zip(self.models, self.model_configs, strict=False)):
            self.logger.info(f"모델 {i + 1} / {len(self.models)} 예측 중...")

            # 각 모델의 변환 설정
            model_name = model_config.get("model_name", model.hparams.model_name)
            _, test_transform = create_train_test_transforms(model_name)

            # DataModule 생성
            data_module = DocumentImageDataModule(
                batch_size=batch_size,
                train_transform=test_transform,
                test_transform=test_transform,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            data_module.setup(stage="predict")

            # 모델 예측
            model_predictions = self._predict_single_model(model, data_module)
            all_model_predictions.append(model_predictions)
            model_weights.append(model_config.get("weight", 1.0))

            self.logger.info(f"모델 {i + 1} 예측 완료")

        # 앙상블 조합
        ensemble_predictions = self._combine_predictions(all_model_predictions, model_weights)
        self.logger.info("앙상블 예측 완료")
        return ensemble_predictions

    def _predict_single_model(self, model: DocumentImageClassifier, data_module: DocumentImageDataModule) -> np.ndarray:
        """단일 모델 예측"""
        predictions = self.trainer.predict(model, data_module.predict_dataloader())
        all_predictions = []
        for batch_predictions in predictions:
            all_predictions.extend(batch_predictions.cpu().numpy())
        return np.array(all_predictions)

    def _combine_predictions(self, predictions_list: list[np.ndarray], weights: list[float]) -> np.ndarray:
        """예측 결과 조합"""
        predictions_array = np.stack(predictions_list, axis=0)  # (n_models, n_samples)
        weights_array = np.array(weights) / np.sum(weights)  # 가중치 정규화

        if self.ensemble_method == "soft_voting":
            # 소프트 보팅: 가중 평균
            return np.average(predictions_array, axis=0, weights=weights_array)

        if self.ensemble_method == "hard_voting":
            # 하드 보팅: 다수결
            return np.apply_along_axis(
                lambda x: np.bincount(x.astype(int), weights=weights_array).argmax(), axis=0, arr=predictions_array
            )

        if self.ensemble_method == "weighted_average":
            # 단순 가중 평균
            return np.average(predictions_array, axis=0, weights=weights_array)

        raise ValueError(f"지원하지 않는 앙상블 방법: {self.ensemble_method}")
