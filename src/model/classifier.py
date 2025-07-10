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
        weight_decay: float = 0.05,  # weight_decay 추가
        drop_rate: float = 0.0,  # dropout rate 추가
        drop_path_rate: float = 0.0,  # drop path rate 추가
        criterion: nn.Module | None = None,
        pretrained: bool = True,
        # 옵티마이저 및 스케줄러 설정
        optimizer_name: str = "adamw",
        scheduler_name: str = "cosine",
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        # 추가 정규화 옵션들
        label_smoothing: float = 0.0,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
    ):
        super().__init__()

        # 하이퍼파라미터 저장
        self.save_hyperparameters(ignore=["criterion"])

        # 모델 설정 (dropout 포함)
        self.model = timm.create_model(
            model_name=self.hparams.model_name,
            num_classes=self.hparams.num_classes,
            pretrained=self.hparams.pretrained,
            drop_rate=self.hparams.drop_rate,  # 일반 dropout
            drop_path_rate=self.hparams.drop_path_rate,  # stochastic depth
        )

        # 손실 함수 설정 (label smoothing 포함)
        if criterion is None:
            if self.hparams.label_smoothing > 0:
                self.criterion = nn.CrossEntropyLoss(label_smoothing=self.hparams.label_smoothing)
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

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

    def _apply_mixup_cutmix(self, images: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor, float]:
        """MixUp 또는 CutMix 적용"""
        import numpy as np
        import torch

        if self.hparams.mixup_alpha > 0 and self.hparams.cutmix_alpha > 0:
            # MixUp과 CutMix 중 랜덤 선택
            use_mixup = np.random.rand() < 0.5
        elif self.hparams.mixup_alpha > 0:
            use_mixup = True
        elif self.hparams.cutmix_alpha > 0:
            use_mixup = False
        else:
            return images, targets, targets, 1.0

        batch_size = images.size(0)
        indices = torch.randperm(batch_size).to(images.device)

        if use_mixup:
            # MixUp
            lam = np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha)
            mixed_images = lam * images + (1 - lam) * images[indices]
        else:
            # CutMix
            lam = np.random.beta(self.hparams.cutmix_alpha, self.hparams.cutmix_alpha)

            # CutMix를 위한 박스 생성
            W, H = images.size(2), images.size(3)
            cut_rat = np.sqrt(1.0 - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)

            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

            mixed_images = images.clone()
            mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]

            # 실제 면적 비율로 lam 조정
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        return mixed_images, targets, targets[indices], lam

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

        # MixUp/CutMix 적용 (훈련 시에만)
        if self.training and (self.hparams.mixup_alpha > 0 or self.hparams.cutmix_alpha > 0):
            mixed_images, targets_a, targets_b, lam = self._apply_mixup_cutmix(images, targets)

            # 순전파
            predictions = self(mixed_images)

            # MixUp/CutMix 손실 계산
            loss = lam * self.criterion(predictions, targets_a) + (1 - lam) * self.criterion(predictions, targets_b)

            # 메트릭은 원본 타겟으로 계산
            predicted_targets = argmax(predictions, dim=1)
            self.train_accuracy(predicted_targets, targets)
            self.train_f1(predicted_targets, targets)
        else:
            # 일반적인 훈련
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

    def configure_optimizers(self) -> dict:
        """
        옵티마이저 및 스케줄러 설정
        """
        # 옵티마이저 선택
        if self.hparams.optimizer_name.lower() == "adamw":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        elif self.hparams.optimizer_name.lower() == "adam":
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        else:
            raise ValueError(f"지원하지 않는 옵티마이저: {self.hparams.optimizer_name}")

        # 스케줄러 선택
        if self.hparams.scheduler_name.lower() == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.learning_rate * 0.01
            )
            scheduler_name = "lr-cosine"
        elif self.hparams.scheduler_name.lower() == "cosine_warm_restarts":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=self.hparams.max_epochs // 4, T_mult=2, eta_min=self.hparams.learning_rate * 0.01
            )
            scheduler_name = "lr-cosine_warm_restarts"
        elif self.hparams.scheduler_name.lower() == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            scheduler_name = "lr-step"
        elif self.hparams.scheduler_name.lower() == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
            scheduler_name = "lr-exponential"
        else:
            raise ValueError(f"지원하지 않는 스케줄러: {self.hparams.scheduler_name}")

        # 워밍업이 있는 경우
        if self.hparams.warmup_epochs > 0:
            from torch.optim.lr_scheduler import LinearLR, SequentialLR

            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.hparams.warmup_epochs
            )

            main_scheduler = scheduler

            scheduler = SequentialLR(
                optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[self.hparams.warmup_epochs]
            )
            scheduler_name = f"lr-warmup_{self.hparams.scheduler_name}"

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1",
                "interval": "epoch",
                "frequency": 1,
                "name": scheduler_name,  # 스케줄러 이름 명시
            },
        }

    def __repr__(self) -> dict:
        """모델 요약 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": self.hparams.model_name,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "learning_rate": self.hparams.learning_rate,
            "weight_decay": self.hparams.weight_decay,
            "drop_rate": self.hparams.drop_rate,
            "drop_path_rate": self.hparams.drop_path_rate,
            "optimizer": self.hparams.optimizer_name,
            "scheduler": self.hparams.scheduler_name,
        }
