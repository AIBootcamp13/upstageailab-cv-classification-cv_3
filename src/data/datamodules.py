from collections.abc import Callable

from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from src.data.datasets import DocumentImageSet
from src.util.log import get_logger


class DocumentImageDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        train_transform: Callable,
        test_transform: Callable,
        val_rate: float = 0.2,
        stratify: bool = True,
        random_seed: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """
        DocumentImageDataModule 초기화

        Args:
            batch_size (int): 배치 크기
            num_workers (int): 데이터 로더에서 사용할 워커 스레드 개수 (기본값: 0)
            pin_memory (bool): GPU 메모리 고정 여부 (기본값: False)
        """
        super().__init__()

        self.batch_size = batch_size
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.val_rate = val_rate
        self.stratify = stratify
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset: DocumentImageSet | None = None
        self.val_dataset: DocumentImageSet | None = None
        self.test_dataset: DocumentImageSet | None = None
        self.predict_dataset: DocumentImageSet | None = None

        self._logger = get_logger("document_image_data_module")

    def prepare_data(self) -> None:
        """
        데이터 준비 작업 수행

        **실행 환경**: CPU 에서만 실행
        **병렬 처리**: 분산 훈련 시에도 단일 프로세스에서만 실행 (rank 0)
        **호출 시점**: 모든 설정 작업 이전, 한 번만 실행

        데이터 다운로드, 압축 해제, 전처리 등 한 번만 수행하면 되는 작업을 처리합니다.
        멀티 GPU 환경에서도 중복 실행되지 않아 안전합니다.
        """
        from src import config

        if not config.TRAIN_META_CSV_PATH.exists():
            raise RuntimeError("train 이미지의 metadata 를 가진 csv 파일이 없습니다.")

        if not config.TRAIN_IMG_DIR.exists():
            raise RuntimeError("train 이미지 directory 가 없습니다.")

        if not config.TEST_META_CSV_PATH.exists():
            raise RuntimeError("test 이미지의 metadata 를 가진 csv 파일이 없습니다.")

        if not config.TEST_META_CSV_PATH.exists():
            raise RuntimeError("test 이미지 directory 가 없습니다.")

        if not config.CLASS_META_CSV_PATH.exists():
            raise RuntimeError("문서 타입 metadata 를 가진 csv 파일이 없습니다.")

        self._logger.info("Done preparing data.")

    def setup(self, stage: str | None = None) -> None:
        """
        데이터셋 설정 및 분할

        **실행 환경**: CPU 에서 실행
        **병렬 처리**: 각 프로세스/GPU 에서 개별적으로 실행됨
        **호출 시점**: 각 단계(fit/validate/test/predict) 시작 전

        Args:
            stage (str | None): 실행 단계 ('fit', 'validate', 'test', 'predict' 중 하나)
                               None 인 경우 모든 단계에 대해 설정

        각 워커 프로세스에서 데이터셋 객체를 생성하고 train/val/test로 분할합니다.
        prepare_data()와 달리 모든 프로세스에서 실행되므로 다운로드 작업은 피해야 합니다.
        """
        if stage == "fit" or stage is None:
            total_train_dataset = DocumentImageSet.create_train_dataset(self.train_transform)
            train_indices, val_indices = self._split_train_val_indices(total_train_dataset)

            self.train_dataset = Subset(total_train_dataset, train_indices)
            self.val_dataset = Subset(total_train_dataset, val_indices)

            self.test_dataset = DocumentImageSet.create_test_dataset(self.test_transform)
            self.predict_dataset = DocumentImageSet.create_test_dataset(self.test_transform)
        elif stage == "validate":
            total_train_dataset = DocumentImageSet.create_train_dataset(self.train_transform)
            _, val_indices = self._split_train_val_indices(total_train_dataset)
            self.val_dataset = Subset(total_train_dataset, val_indices)
        elif stage == "test":
            self.test_dataset = DocumentImageSet.create_test_dataset(self.test_transform)
        elif stage == "predict":
            self.predict_dataset = DocumentImageSet.create_test_dataset(self.test_transform)

        self._logger.info("Done setup")

    def train_dataloader(self) -> DataLoader:
        """
        훈련용 데이터 로더 반환

        **실행 환경**: CPU에서 DataLoader 객체 생성, 실제 데이터 로딩은 워커 프로세스
        **병렬 처리**: num_workers 개의 서브프로세스에서 데이터 로딩
        **데이터 전송**: pin_memory=True 시 GPU로 비동기 전송

        Returns:
            DataLoader: 훈련 데이터셋을 위한 데이터 로더

        num_workers 만큼의 멀티프로세싱으로 데이터 로딩을 병렬 처리합니다.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """
        검증용 데이터 로더 반환

        **실행 환경**: CPU에서 DataLoader 객체 생성, 실제 데이터 로딩은 워커 프로세스
        **병렬 처리**: num_workers 개의 서브프로세스에서 데이터 로딩
        **데이터 전송**: pin_memory=True 시 GPU로 비동기 전송

        Returns:
            DataLoader: 검증 데이터셋을 위한 데이터 로더
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        """
        테스트용 데이터 로더 반환

        **실행 환경**: CPU에서 DataLoader 객체 생성, 실제 데이터 로딩은 워커 프로세스
        **병렬 처리**: num_workers 개의 서브프로세스에서 데이터 로딩
        **데이터 전송**: pin_memory=True 시 GPU로 비동기 전송

        Returns:
            DataLoader: 테스트 데이터셋을 위한 데이터 로더
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self) -> DataLoader:
        """
        예측용 데이터 로더 반환

        **실행 환경**: CPU에서 DataLoader 객체 생성, 실제 데이터 로딩은 워커 프로세스
        **병렬 처리**: num_workers 개의 서브프로세스에서 데이터 로딩
        **데이터 전송**: pin_memory=True 시 GPU로 비동기 전송

        Returns:
            DataLoader: 예측 데이터셋을 위한 데이터 로더
        """
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def _split_train_val_indices(self, dataset: DocumentImageSet) -> tuple[list[int], list[int]]:
        """테스및와 평가을 나눌 수 있는 index list 만들기"""
        indices = list(range(len(dataset)))
        if self.stratify:
            targets = [dataset[idx][1] for idx in indices]
            return train_test_split(
                indices,
                stratify=targets,
                test_size=self.val_rate,
                random_state=self.random_seed,
            )
        return train_test_split(
            indices,
            test_size=self.val_rate,
            random_state=self.random_seed,
        )
