<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# 📄 Augraphy로 증강 데이터 구현

## 🎯 개요

이 코드는 **Augraphy** 라이브러리를 사용하여 문서 이미지 데이터를 증강하는 완전한 파이프라인을 구현합니다. 문서 스캔, 복사, 인쇄 등의 실제 상황에서 발생하는 다양한 왜곡과 노이즈를 시뮬레이션하여 모델의 robustness를 향상시킵니다.

## 🏗️ 전체 구조

### 1. 📊 데이터셋 클래스 (`CvImageDatasetFastEx`)

```python
class CvImageDatasetFastEx(Dataset):
    def __init__(self, source, transform=None, img_size=config.IMAGE_SIZE):
        self.df = source.df.values
        self.img_source = source
        self.transform = transform
        self.img_size = img_size
```

**핵심 기능:**

- 🔄 PyTorch Dataset 인터페이스 구현
- 🎯 타겟 정보를 이미지 객체에 직접 저장
- 🔧 변환 파이프라인 적용


### 2. 🎨 이미지 변환 파이프라인

#### **To_BGR 클래스**

```python
class To_BGR(object):
    def __call__(self, image):
        target = image.target
        image_numpy = np.array(image)
        
        # RGB → BGR 변환
        if len(image_numpy.shape) < 3:
            bgr_image = cv2.cvtColor(image_numpy, cv2.COLOR_GRAY2BGR)
        else:
            bgr_image = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
        
        # 타겟 정보를 B 채널 하위 5비트에 저장
        bgr_image[0, 0, 0] = (bgr_image[0, 0, 0] & 0xE0) | (target & 0x1F)
        
        return bgr_image
```

**특징:**

- 🎯 **타겟 정보 임베딩**: B 채널의 첫 번째 픽셀에 클래스 정보 저장
- 🔄 **포맷 변환**: PIL RGB → OpenCV BGR 형식으로 변환
- 🖼️ **그레이스케일 처리**: 단일 채널 이미지를 3채널로 확장


#### **변환 파이프라인 구성**

```python
dirty_transforms = transforms.Compose([
    CustomRandomHorizontalFlip(),    # 🔄 수평 뒤집기
    To_BGR(),                       # 🎨 BGR 변환 + 타겟 임베딩
    FullAugraphyPipelineQueueEx(),  # 📄 Augraphy 증강
    default_transform               # 🔧 모델별 기본 변환
])
```


## 🚀 Augraphy 파이프라인 구현

### 3. 📋 `FullAugraphyPipelineQueueEx` 클래스

#### **초기화 과정**

```python
def __init__(self, num_pipelines=10):
    self.pipelines = []
    self.augument_image_manager = AugmentImageManager(csv_path=config.CV_CLS_AUGMENT_CSV)
```

**🎯 다중 파이프라인 생성:**

- 10개의 서로 다른 증강 파이프라인 생성
- 각 파이프라인마다 다른 랜덤 설정 적용
- 증강 이미지 관리자를 통한 메타데이터 관리


#### **워터마크 효과 생성**

```python
self.list_watermark = []

for i in range(7):  # 7개의 서로 다른 워터마크
    watermark = WaterMark(
        watermark_word="random",
        watermark_font_size=(1, 2),  
        watermark_font_thickness=(1, 2),
        watermark_rotation=(i*30, i*30 + 60),  # 각기 다른 회전각
        watermark_location="random",
        watermark_color=(200, 200, 200),
        watermark_method="darken",
        p=0.3
    )
    self.list_watermark.append(watermark)
```

**특징:**

- 🔄 **다양한 회전각**: 각 워터마크마다 30도씩 다른 회전 범위
- 🎨 **랜덤 위치**: 문서 전체에 무작위 배치
- 💧 **투명도 조절**: darken 모드로 자연스러운 워터마크 효과


#### **3단계 증강 파이프라인**

```python
for _ in range(num_pipelines):
    # 1️⃣ Ink Phase (잉크 효과)
    ink_effects = []
    
    # 2️⃣ Paper Phase (종이 효과)
    paper_effects = [
        SubtleNoise(subtle_range=random.randint(60, 90), p=0.3),
        ColorPaper(hue_range=(0, 360), saturation_range=(0, 5), p=0.3)
    ]
    paper_effects.extend(self.list_watermark)
    
    # 3️⃣ Post Phase (후처리 효과)
    post_effects = [
        Geometric(
            scale=(1, 1.1),
            translation=probabilistic_translation(),
            fliplr=0.5,
            flipud=0.5,
            p=0.5
        ),
    ]
```

**각 단계별 효과:**


| 단계 | 효과 | 설명 |
| :-- | :-- | :-- |
| 🖋️ **Ink Phase** | 잉크 관련 효과 | 현재 비활성화 (필요시 추가 가능) |
| 📄 **Paper Phase** | 종이 질감/색상 | SubtleNoise, ColorPaper, 워터마크 |
| 🔧 **Post Phase** | 기하학적 변형 | 스케일링, 이동, 뒤집기 |

### 4. 🔄 증강 실행 과정

#### **메인 증강 함수**

```python
def __call__(self, image):
    # 1️⃣ 타겟 정보 추출
    target = image[0, 0, 0] & 0x1F
    
    # 2️⃣ 회전 변환 적용
    rotated_image = rotate_first_then_augraphy(image)
    
    # 3️⃣ 랜덤 파이프라인 선택 및 적용
    pipeline = random.choice(self.pipelines)
    augmented = pipeline(rotated_image)
    
    # 4️⃣ 채널 정규화
    # ... 채널 수에 따른 처리 로직
    
    # 5️⃣ PIL 이미지로 변환
    pil_result = Image.fromarray(augmented.astype(np.uint8))
    
    # 6️⃣ 증강 이미지 저장 (선택적)
    if self.augument_image_manager.get_count() < 10000:
        self.save_augment_file(pil_result, target)
    
    return pil_result
```


#### **회전 변환 함수**

```python
def rotate_first_then_augraphy(image):
    height, width = image.shape[:2]
    angle = random.choice([0, 90, 180, 270, random.randint(1, 359)])
    
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    if angle % 180 == 0:
        # 0도 또는 180도: 크기 그대로
        rotated = cv2.warpAffine(image, M, (width, height), ...)
    elif angle % 90 == 0:
        # 90도 또는 270도: 폭/높이 뒤바꿈
        rotated = cv2.warpAffine(image, M, (height, width), ...)
    else:
        # 비정각: 바운딩 박스 계산
        # ... 바운딩 박스 계산 로직
        rotated = cv2.warpAffine(image, M, (bound_w, bound_h), ...)
```

**회전 처리 전략:**

- 🎯 **정각 회전**: 0°, 90°, 180°, 270° 정확한 처리
- 🔄 **임의 각도**: 바운딩 박스 계산으로 잘림 방지
- 🖼️ **크기 조정**: 각도에 따른 적절한 출력 크기 설정


### 5. 💾 증강 이미지 저장

```python
def save_augment_file(self, pil_image, target, prefix="augraphy", output_dir=config.CV_CLS_AUGMENT_DIR):
    # 파일명 생성: augraphy_20250710_213000_2_0000001.png
    id = f"{self.augument_image_manager.get_count() + 1:07d}"
    now = datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S")
    
    filename = f"{prefix}_{time_str}_{target}_{id}.png"
    filepath = os.path.join(output_dir, filename)
    
    # 저장 및 메타데이터 관리
    pil_image.save(filepath)
    self.augument_image_manager.add_image_data(
        image_path=filepath, 
        image_name=filename, 
        target=target
    )
```


## 📊 데이터셋 구성

### 6. 🎯 데이터셋 생성 및 분할

```python
def get_datasets(default_cfg=None):
    # 데이터 소스 로드
    train_source = MemoryImageSourceEx(config.CV_CLS_TRAIN_CSV, config.CV_CLS_TRAIN_DIR)
    test_source = MemoryImageSourceEx(config.CV_CLS_TEST_CSV, config.CV_CLS_TEST_DIR)
    
    # 증강 데이터셋 생성 (2배 증강)
    d1 = CvImageDatasetFastEx(train_source, transform=dirty_transforms)
    d2 = CvImageDatasetFastEx(train_source, transform=dirty_transforms)
    train_dataset = ConcatDataset([d1, d2])
    
    # 계층적 분할 (8:2)
    train_dataset, val_dataset = stratified_split(
        dataset=train_dataset, 
        test_size=0.2, 
        random_state=42
    )
    
    # 테스트 데이터셋 (증강 없음)
    test_dataset = CvImageDatasetFastEx(test_source, transform=default_transform)
    
    return train_dataset, val_dataset, test_dataset
```


### 7. 🎲 계층적 분할 함수

```python
def stratified_split(dataset, test_size=0.2, random_state=42):
    indices = list(range(len(dataset)))
    labels = [label for _, label in dataset]
    
    train_idx, val_idx = train_test_split(
        indices,
        stratify=labels,  # 클래스 비율 유지
        test_size=test_size,
        random_state=random_state
    )
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    return train_subset, val_subset
```


## 🎯 핵심 특징

### ✨ **주요 장점**

1. **🔄 다양성**: 10개의 서로 다른 파이프라인으로 다양한 증강 효과
2. **🎯 타겟 보존**: 이미지 픽셀에 클래스 정보 임베딩으로 정보 손실 방지
3. **📊 균형 유지**: 계층적 분할로 클래스 비율 유지
4. **💾 관리**: 증강 이미지 자동 저장 및 메타데이터 관리
5. **🔧 유연성**: 설정 가능한 확률과 파라미터로 세밀한 제어

### 🎨 **증강 효과 종류**

- **📄 종이 효과**: 색상 변화, 미세 노이즈, 워터마크
- **🔄 기하학적 변형**: 회전, 스케일링, 이동, 뒤집기
- **💧 워터마크**: 7가지 다른 각도의 워터마크 효과
- **🎯 확률적 적용**: 각 효과마다 적용 확률 설정


