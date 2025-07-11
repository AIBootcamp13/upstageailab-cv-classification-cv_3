# Document Image Classification Project

## Overview

이번 대회는 **문서 타입 분류**를 위한 이미지 분류 대회를 위한 프로젝트입니다.

- **태스크**: 17개 클래스의 문서 이미지 분류
- **도메인**: Computer Vision - Image Classification
- **활용 분야**: 금융, 의료, 보험, 물류 등 산업 전반의 문서 자동화 처리
- **데이터**: 실제 현업에서 사용하는 데이터를 기반으로 제작
- **평가 지표**: Macro F1 Score

## Team

| [조의영](https://github.com/yuiyeong) | [김두환](https://github.com/korea202a) | [나주영](https://github.com/najuyoung) | [조재형](https://github.com/Bitefriend) |
|:----------------------------------:|:-----------------------------------:|:-----------------------------------:|:------------------------------------:|
|                 팀장                 |                 팀원                  |                 팀원                  |                  팀원                  |

## 0. Getting Started

- 본 프로젝트는 클라우드 인스턴스에서 진행됩니다.
- 다음 단계를 따라 환경을 설정하고 프로젝트를 시작하세요.
  제공해주신 pyproject.toml 파일을 분석하여 README의 "Environment"와 "Requirements" 섹션을 업데이트하겠습니다.

### Environment

- **OS**: macOS (Darwin) / Linux (CUDA 12.1 지원)
- **Python**: 3.11+
- **Package Manager**: Poetry
- **Deep Learning Framework**: PyTorch 2.6.0 + CUDA 12.1 (Linux) / PyTorch 2.6.0 (macOS)
- **주요 라이브러리**: timm, albumentations, OpenCV, scikit-learn, pytorch-lightning
- **실험 관리**: Weights & Biases (wandb)
- **클라우드 스토리지**: Naver Cloud Storage (boto3)

### Requirements

본 프로젝트는 Poetry를 사용하여 의존성을 관리합니다.

#### 플랫폼별 PyTorch 설치

- **Linux**: PyTorch 2.6.0 + CUDA 12.1 지원
- **macOS**: PyTorch 2.6.0 (CPU/MPS 지원)

#### 주요 의존성

- `torch==2.6.0` / `torchvision==0.21.0` / `torchaudio==2.6.0` - PyTorch 딥러닝 프레임워크
- `pytorch-lightning>=2.5.2` - PyTorch 고수준 훈련 프레임워크
- `timm>=1.0.16` - 사전 훈련된 컴퓨터 비전 모델
- `albumentations>=2.0.8` - 이미지 증강 라이브러리
- `opencv-python>=4.11.0` - 컴퓨터 비전 라이브러리
- `scikit-learn>=1.7.0` - 머신러닝 유틸리티
- `scikit-image>=0.25.2` - 이미지 처리 라이브러리
- `augraphy>=8.2.6` - 문서 이미지 증강 라이브러리

#### 데이터 처리 및 분석

- `numpy>=2.2.6` - 수치 연산 라이브러리
- `pandas>=2.3.1` - 데이터 처리 및 분석
- `matplotlib>=3.10.3` - 데이터 시각화
- `seaborn>=0.13.2` - 통계 데이터 시각화
- `pillow>=11.2.1` - 이미지 처리

#### 실험 관리 및 도구

- `wandb>=0.20.1` - 실험 추적 및 관리
- `boto3>=1.38.46` - AWS SDK (클라우드 스토리지)
- `tqdm>=4.67.1` - 진행률 표시

#### 개발 환경

- `jupyter>=1.1.1` - Jupyter 노트북
- `jupyterlab>=4.4.3` - JupyterLab 환경
- `notebook>=7.4.3` - Jupyter 노트북 서버
- `ipykernel>=6.29.5` - Jupyter 커널
- `python-dotenv>=1.1.0` - 환경 변수 관리

#### 개발 도구 (dev-dependencies)

- `ruff>=0.12.0` - 코드 린팅 및 포맷팅
- `pre-commit>=4.2.0` - Git 훅 관리

### 1. 환경 설정

- 클라우드 인스턴스에 접속한 후, 다음 명령어를 실행하여 개발 환경을 자동으로 설정합니다,

```bash
# 환경 설정 스크립트 다운로드 및 실행
wget https://gist.githubusercontent.com/yuiyeong/8ae3f167e97aeff90785a4ccda41e5fe/raw/d5e030ea64bbd9c41ce2b4c825bc03c86f0c3dac/setup_env.sh

chmod +x setup_env.sh
./setup_env
```

**설정 내용**

- [init-cloud-instance.sh](scripts/init-cloud-instance.sh)
- Python 3.11 conda 환경 (py311) 생성
- Poetry 설치 및 PATH 설정
- /workspace 작업 디렉토리 생성
- SSH 로그인 시 자동으로 /workspace로 이동

### 2. 환경 적용

스크립트 실행 후 SSH를 재접속하여 변경사항을 적용합니다.

```bash
# SSH 재접속 후 환경 확인
python --version  # Python 3.11.x 확인
poetry --version  # Poetry 설치 확인
pwd              # /workspace 확인
```

### 3. Git Config

- 다음 명령어를 실행하여 git config 를 설정합니다.
- `{username}` 과 `{emailaddr}` 에 본인의 github name 과 email 을 적어주세요

```bash
git config --global user.name "{username}"
git config --global user.email "{emailaddr}"
git config --global core.editor "vim"
git config --global core.pager "cat"
```

- 설정된 내용은 `git config --list` 로 확인합니다.
- 수정이 필요할 경우, `vi ~/.gitconfig` 를 실행해서 값을 수정합니다.

### 4. 프로젝트 복제 및 설정

```bash
# /workspace 디렉토리에서 프로젝트 복제
cd /workspace
git clone https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_3.git
cd upstageailab-cv-classification-cv-3

# Poetry를 사용하여 의존성 설치
poetry install
```

### 5. 환경 변수 설정

```bash
# 환경 변수 템플릿 파일 복사
cp .env.template .env

# 환경 변수 파일 편집
vi .env
```

**필요한 환경 변수**

- `PYTHONPATH`: 프로젝트 루트 경로 설정
- `NCLOUD_ACCESS_KEY`: Naver Cloud Storage 접근 키
- `NCLOUD_SECRET_KEY`: Naver Cloud Storage 시크릿 키
- `NCLOUD_STORAGE_REGION`: Naver Cloud Storage 리전
- `NCLOUD_STORAGE_ENDPOINT_URL`: Naver Cloud Storage 엔드포인트
- `NCLOUD_STORAGE_BUCKET`: Naver Cloud Storage 버킷명
- `NCLOUD_STORAGE_BUCKET_PERSONAL_DIR`: 개인 디렉토리 경로
- `WANDB_API_KEY`: Weights & Biases API 키
- `WANDB_ENTITY`: Weights & Biases 엔터티
- `WANDB_PROJECT`: Weights & Biases 프로젝트명

### 6. 대회 데이터 다운로드

```bash
# 데이터 다운로드 (대회 페이지에서 URL 확인)
wget [DATA_URL] -O data.tar.gz

# 압축 해제
tar -zxvf data.tar.gz

mv data/ upstageailab-cv-classification-cv-3/data/raw
```

## 1. Competition Info

### Dataset Statistics

- **학습 데이터**: 1,570장의 이미지
- **평가 데이터**: 3,140장의 이미지
- **클래스 수**: 17개 문서 타입
- **이미지 형식**: JPG
- **특징**: 다양한 문서 상태 (회전, 뒤집힘, 훼손 등)

### Class Information

총 17개의 문서 클래스로 구성

- account_number (계좌번호)
- application_for_payment_of_pregnancy_medical_e (임신의료비 지급신청서)
- car_dashboard (차량 대시보드)
- confirmation_of_admission_and_discharge (입퇴원 확인서)
- diagnosis (진단서)
- driver_licence (운전면허증)
- medical_bill_receipts (의료비 영수증)
- medical_outpatient_certificate (의료 외래 증명서)
- national_id_card (주민등록증)
- passport (여권)
- payment_confirmation (결제 확인서)
- pharmaceutical_receipt (약국 영수증)
- prescription (처방전)
- resume (이력서)
- statement_of_opinion (의견서)
- vehicle_registration_certificate (차량등록증)
- vehicle_registration_plate (차량번호판)

### Timeline

- **Start Date**: 2025-06-30
- **Final submission deadline**: 2025-07-11

## 2. Components

### Directory Structure

```
├── data/                           # 데이터 저장 디렉토리
│   ├── fonts/                      # 문서 이미지 생성/처리용 폰트 파일
│   └── raw/                        # 원본 데이터 (train, test 이미지, CSV 파일)
├── docs/                           # 프로젝트 문서 및 자료
│   └── img/                        # 문서용 이미지
├── notebooks/                      # Jupyter 노트북 디렉토리
│   ├── duhwan/                     # 김두환 개인 실험 노트북
│   ├── jaehyeong/                  # 조재형 개인 실험 노트북
│   ├── juyoung/                    # 나주영 개인 실험 노트북
│   ├── yuiyeong/                   # 조의영 개인 실험 노트북
│   └── notebook_template.ipynb     # 노트북 작성을 위한 공통 템플릿
├── scripts/                        # 유틸리티 스크립트 모음
│   └── init-cloud-instance.sh      # 클라우드 인스턴스 환경 설정 스크립트
├── src/                            # 소스 코드 메인 디렉토리
│   ├── config/                     # 설정 관련 모듈
│   ├── data/                       # 데이터 로더 및 데이터셋 관련
│   ├── libs/                       # 공통 라이브러리 및 유틸리티
│   ├── model/                      # 모델 정의 및 구현
│   ├── script/                     # 실행 스크립트
│   ├── training/                   # 훈련 관련 모듈
│   ├── transforms/                 # 이미지 변환 및 증강
│   └── util/                       # 유틸리티 함수 모음
├── .env.template                   # 환경 변수 템플릿 파일
└── pyproject.toml                  # Poetry 의존성 관리 파일
```

제공해주신 세 개의 스크립트를 바탕으로 "스크립트 사용법" 섹션을 업데이트하겠습니다.

### 스크립트 사용법

- `src/script` 디렉토리에 있는 세 가지 주요 스크립트의 사용법을 설명합니다.

#### 1. src/script/train.py

모델 학습을 위한 스크립트입니다.

```bash
python src/script/train.py \
    --model-name efficientnet_b4 \
    --learning-rate 0.001 \
    --batch-size 32 \
    --epochs 50 \
    --val-rate 0.2 \
    --num-workers 8 \
    --pin-memory \
    --seed 4321 \
    --checkpoint-path /path/to/checkpoint.ckpt
```

##### 주요 Arguments

**모델 관련**

- `--model-name`: timm 기준 사전학습된 모델 이름 (기본값: convnextv2_atto)
- `--learning-rate`: 학습률 (기본값: 5e-4)
- `--weight-decay`: 가중치 감소율 (기본값: 0.05)
- `--drop-rate`: 드롭아웃 비율 (기본값: 0.1)
- `--drop-path-rate`: 드롭 패스 비율 (기본값: 0.1)

**학습 관련**

- `--batch-size`: 배치 사이즈 (기본값: 16)
- `--epochs`: 학습 에포크 수 (기본값: 100)
- `--optimizer`: 옵티마이저 선택 (기본값: adamw, 선택: adamw, adam)
- `--scheduler`: 학습률 스케줄러 (기본값: cosine, 선택: cosine, cosine_warm_restarts, step, exponential)
- `--warmup-epochs`: 웜업 에포크 수 (기본값: 5)

**정규화 관련**

- `--label-smoothing`: 라벨 스무딩 계수 (기본값: 0.1)
- `--mixup-alpha`: MixUp 알파 값 (기본값: 0.2, 0으로 설정하면 비활성화)
- `--cutmix-alpha`: CutMix 알파 값 (기본값: 0.0, 0으로 설정하면 비활성화)

**데이터 관련**

- `--val-rate`: 검증 데이터 분할 비율 (기본값: 0.2)
- `--num-workers`: 데이터 로딩 워커 수 (기본값: 4)
- `--pin-memory`: 데이터 로딩 시 pin_memory 사용 (플래그)

**기타**

- `--seed`: 랜덤 시드 설정 (기본값: 4321)
- `--checkpoint-path`: 체크포인트에서 시작할 경우 경로 (기본값: None)

#### 2. src/script/predict.py

단일 모델을 사용한 예측을 위한 스크립트입니다.

```bash
python src/script/predict.py \
    --checkpoint-path /path/to/trained_model.ckpt \
    --batch-size 128 \
    --num-workers 8 \
    --pin-memory \
    --seed 4321
```

##### 주요 Arguments

- `--checkpoint-path`: **필수** - 학습된 모델 체크포인트 경로
- `--batch-size`: 예측 배치 사이즈 (기본값: 64)
- `--num-workers`: 데이터 로딩 워커 수 (기본값: 4)
- `--pin-memory`: 데이터 로딩 시 pin_memory 사용 (플래그, 기본값: False)
- `--seed`: 랜덤 시드 설정 (기본값: 4321)

#### 3. src/script/predict_with_ensemble.py

여러 모델을 결합한 앙상블 예측을 위한 스크립트입니다.

```bash
python src/script/predict_with_ensemble.py \
    --ensemble-method soft_voting \
    --batch-size 32 \
    --num-workers 4 \
    --pin-memory \
    --seed 4321 \
    --output-suffix my_ensemble
```

##### 주요 Arguments

**앙상블 관련**

- `--ensemble-method`: 앙상블 방법 (기본값: hard_voting, 선택: soft_voting, hard_voting, weighted_average)
- `--output-suffix`: 출력 파일명 접미사 (기본값: ensemble)

**데이터 로딩 관련**

- `--batch-size`: 예측 배치 사이즈 (기본값: 32, 앙상블 시 작게 설정 권장)
- `--num-workers`: 데이터 로딩 워커 수 (기본값: 4)
- `--pin-memory`: 데이터 로딩 시 pin_memory 사용 (플래그, 기본값: False)

**기타**

- `--seed`: 랜덤 시드 설정 (기본값: 4321)

### 실사용 예시

#### 기본 학습 (최소 필수 arguments)

```bash
python src/script/train.py --model-name efficientnet_b4
```

#### 고성능 학습 (모든 arguments 최적화)

```bash
python src/script/train.py \
    --model-name efficientnet_b4 \
    --learning-rate 0.0005 \
    --batch-size 16 \
    --epochs 100 \
    --val-rate 0.15 \
    --num-workers 12 \
    --pin-memory \
    --seed 42 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.3 \
    --weight-decay 0.01
```

#### 체크포인트에서 재시작

```bash
python src/script/train.py \
    --model-name efficientnet_b4 \
    --learning-rate 0.0001 \
    --batch-size 16 \
    --epochs 50 \
    --checkpoint-path ./data/checkpoints/best_model.ckpt
```

#### 단일 모델 예측

```bash
python src/script/predict.py \
    --checkpoint-path ./data/checkpoints/best_model.ckpt \
    --batch-size 256 \
    --num-workers 16 \
    --pin-memory \
    --seed 42
```

#### 앙상블 예측

```bash
python src/script/predict_with_ensemble.py \
    --ensemble-method soft_voting \
    --batch-size 32 \
    --num-workers 8 \
    --pin-memory \
    --seed 42 \
    --output-suffix final_ensemble
```

#### 중요 참고사항

- train.py: `--checkpoint-path`는 선택사항이며, 체크포인트에서 재시작할 때만 사용합니다.
- predict.py: `--checkpoint-path`는 필수 argument입니다.
- predict_with_ensemble.py: 앙상블할 모델들의 설정은 `src/config.py`의 `TRAINED_MODEL_CONFIGS`에서 미리 정의되어 있어야 합니다.
- 앙상블 예측 시에는 메모리 사용량이 많아지므로 `--batch-size`를 작게 설정하는 것을 권장합니다.
- 모든 스크립트는 `--pin-memory` 플래그를 사용하여 GPU 메모리 전송 속도를 향상시킬 수 있습니다.

## 3. Data Description

### Dataset Overview

본 대회의 데이터셋은 실제 현업에서 사용되는 문서 이미지를 기반으로 구성되었습니다.

**학습 데이터 구성**

- `train/`: 1,570 장의 학습용 이미지
- `train.csv`: 학습 이미지의 파일명과 정답 클래스 정보
- `meta.csv`: 17개 클래스의 번호와 이름 매핑 정보

**평가 데이터 구성**

- `test/`: 3,140 장의 평가용 이미지
- `sample_submission.csv`: 제출 형식 템플릿

## 4. EDA

- [EDA 보고서](docs/report_of_eda.md)

## 5. Image Augmentation

- [이미지 증강 보고서](docs/make_data_with_Augraphy.md)

## 6. Presentation

- [발표 자료](docs/presentation.pdf)

## 7. etc

### Reference

- [TIMM Documentation](https://timm.fast.ai/)
- [Albumentations Documentation](https://albumentations.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [Naver Cloud Storage Documentation](https://guide.ncloud-docs.com/docs/storage-storage-8-1)
