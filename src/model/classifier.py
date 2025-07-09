import timm
import torchmetrics
from pytorch_lightning import LightningModule
from torch import Tensor, argmax, nn, optim, stack


class DocumentImageClassifier(LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        learning_rate: float,
        criterion: nn.Module | None = None,
        pretrained: bool = True,
    ):
        super().__init__()

        # 하이퍼파라미터 저장
        self.save_hyperparameters(ignore=["criterion"])

        # 모델 설정
        self.model = timm.create_model(
            model_name=self.hparams.model_name,
            num_classes=self.hparams.num_classes,
            pretrained=self.hparams.pretrained,
        )

        # 손실 함수 설정
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()

        # 메트릭 설정 - torchmetrics가 자동으로 상태 관리
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average="macro")

        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average="macro")

        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average="macro")

    def forward(self, x: Tensor) -> Tensor:
        """
        순전파 정의

        Args:
            x (torch.Tensor): 입력 이미지 텐서

        Returns:
            torch.Tensor: 모델 예측 결과
        """
        return self.model(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        훈련 단계

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): (이미지, 타겟) 배치
            batch_idx (int): 배치 인덱스

        Returns:
            torch.Tensor: 손실 값
        """
        images, targets = batch

        # 순전파
        predictions = self(images)
        loss = self.criterion(predictions, targets)

        # 메트릭 계산
        predicted_targets = argmax(predictions, dim=1)
        self.train_accuracy(predicted_targets, targets)
        self.train_f1(predicted_targets, targets)

        # 로깅
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        검증 단계

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): (이미지, 타겟) 배치
            batch_idx (int): 배치 인덱스

        Returns:
            torch.Tensor: 손실 값
        """
        images, targets = batch

        # 순전파
        predictions = self(images)
        loss = self.criterion(predictions, targets)

        # 메트릭 계산
        predicted_targets = argmax(predictions, dim=1)
        self.val_accuracy(predicted_targets, targets)
        self.val_f1(predicted_targets, targets)

        # logging
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        테스트 단계

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): (이미지, 타겟) 배치
            batch_idx (int): 배치 인덱스

        Returns:
            torch.Tensor: 손실 값
        """
        images, targets = batch

        # 순전파
        predictions = self(images)
        loss = self.criterion(predictions, targets)

        # 메트릭 계산
        predicted_targets = argmax(predictions, dim=1)
        self.test_accuracy(predicted_targets, targets)
        self.test_f1(predicted_targets, targets)

        # logging
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_accuracy", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def predict_step(
        self, batch: tuple[Tensor, Tensor] | list | Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """
        예측 단계

        Args:
            batch: 입력 데이터 배치 (tuple, list, 또는 Tensor)
            batch_idx: 배치 인덱스
            dataloader_idx: 데이터로더 인덱스

        Returns:
            torch.Tensor: 예측 결과 (클래스 인덱스)
        """
        # batch 형태 확인 및 이미지 추출
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            images, _ = batch
        else:
            images = batch

        # 이미지가 리스트인 경우 텐서로 변환
        if isinstance(images, list):
            images = stack(images)

        # 모델 추론
        predictions = self(images)

        # 클래스 인덱스 반환
        return argmax(predictions, dim=1)

    def configure_optimizers(self) -> tuple:
        """
        옵티마이저 설정
        :return: Adam 옵티마이저, 학습률 스케쥴러
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1e-4)
        return [optimizer], [scheduler]
