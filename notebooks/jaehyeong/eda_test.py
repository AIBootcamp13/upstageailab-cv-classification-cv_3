import os
import zipfile # 이미지를 불러오기 위한 라이브러리 (Python Imaging Library)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image # zip 파일 열기용
import math # 행/열 수 계산용
import numpy as np
from sklearn.model_selection import train_test_split

# 한글 폰트 설정 (그래프에 깨지지 않도록 설정 - 윈도우 기본 Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스(-) 깨짐 방지

# train.csv에는 이미지 ID와 라벨(label) 정보가 들어있다.
train_df = pd.read_csv(r"C:\Users\재형띠\Desktop\코딩친구들\컴퓨터비전 프로젝트\datasets_folder\data\train.csv")
meta_df = pd.read_csv(r"C:\Users\재형띠\Desktop\코딩친구들\컴퓨터비전 프로젝트\datasets_folder\data\meta.csv")

# 경로 설정
DATA_DIR = r"C:\Users\재형띠\Desktop\코딩친구들\컴퓨터비전 프로젝트\datasets_folder\data"
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
META_CSV_PATH = os.path.join(DATA_DIR, "meta.csv")
TRAIN_ZIP_PATH = os.path.join(DATA_DIR, "train.zip")

# ───────────────────────────────
# 1. 기본 정보 EDA
def eda_basic_info(train_df, meta_df):
    print("📦 [train.csv 정보]")
    print(train_df.info())
    print("\n📦 [meta.csv 정보]")
    print(meta_df.info())

# ───────────────────────────────
# 2. 클래스 분포 시각화
def eda_label_distribution(train_df, meta_df):
    # [2] train.csv에는 1570개의 샘플과 17개의 클래스(target)가 있음
    # 라벨(target)의 분포 확인 (클래스 불균형이 있는지 보기 위함)
    label_counts = train_df['target'].value_counts().sort_index()

    # [3] 보기 좋게 DataFrame 형태로 변환 (시각화, 분석에 편리)
    label_df = pd.DataFrame({
        'target': label_counts.index,
        'count': label_counts.values
    })

    # [4] meta.csv의 class_name과 target을 merge해서 보기 쉽게 이름 붙이기
    label_df = label_df.merge(meta_df, on='target', how='left')

    # [5] 그래프 크기 설정
    plt.figure(figsize=(14, 6))

    # [6] Seaborn 막대그래프로 시각화 (x: 클래스명, y: 개수)
    sns.barplot(x='class_name', y='count', data=label_df)

    # [7] 그래프 제목, 라벨 설정
    plt.title("클래스별 데이터 분포 : train.csv 기준")
    plt.xlabel("클래스 이름")
    plt.ylabel("샘플 수")
    plt.xticks(rotation=90, fontsize=10)  # 글자가 겹치지 않게 회전
    plt.grid(axis='y')
    plt.tight_layout()

    # [8] 그래프 출력
    plt.show()

    
# ───────────────────────────────
# 3. 클래스별 이미지 시각화
def eda_visualize_images(train_df, zip_path, samples_per_class=2):
    """
    📌 설명:
    이 함수는 train.csv에서 각 라벨(class)마다 랜덤으로 이미지 몇 개씩 추출해서
    zip 안에서 직접 열고, 화면에 출력해주는 함수야.
    
    - train_df: train.csv를 pandas로 읽어온 데이터프레임
    - zip_path: train 이미지가 담긴 압축파일 경로 (예: train.zip)
    - samples_per_class: 클래스마다 출력할 이미지 수 (기본값은 2개)
    """
     # [1] 클래스(target) 별로 groupby해서, 클래스마다 무작위로 N개씩 샘플링
    # group_keys=False를 설정해서 경고 방지 (pandas 최신버전용)
    sample_df = train_df.groupby('target', group_keys=False).apply(
        lambda x: x.sample(n=samples_per_class, random_state=42)
    ).reset_index(drop=True)

    # [2] train.zip 파일을 연다 (파일을 압축 풀지 않고 내부에서 바로 읽을 수 있게 함)
    with zipfile.ZipFile(zip_path, 'r') as archive:

        num_images = len(sample_df)  # 전체 시각화할 이미지 수

        # [3] 출력할 subplot의 행/열 수를 자동으로 계산
        cols = 9                                   # 한 줄에 9개의 이미지 출력
        rows = math.ceil(num_images / cols)        # 필요한 줄 수 계산

        # [4] 행렬 형태로 subplot 그리기 (그래프 그릴 틀 만드는 단계)
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten()  # 2차원 → 1차원으로 펼쳐서 리스트로 다루기 쉽게 변환

        # [5] 이미지 하나씩 순회하면서 zip에서 열고 출력
        for i, (idx, row) in enumerate(sample_df.iterrows()):
            # [5-1] 이미지 파일 이름 불러오기 (예: '000001')
            img_name = row['ID']
            
            # [5-2] 확장자 누락 방지 → .jpg가 없으면 붙이기
            if not img_name.endswith(".jpg"):
                img_name += ".jpg"
            
            # [5-3] zip 파일 안에서의 경로 구성 (예: train/000001.jpg)
            img_path = f"train/{img_name}"

            # [5-4] 이미지 zip 안에서 열기 → 오류 생기면 KeyError 처리
            try:
                with archive.open(img_path) as file:
                    img = Image.open(file).convert("RGB")     # 이미지 불러오기 & RGB로 변환
                    axes[i].imshow(img)                       # subplot에 이미지 표시
                    axes[i].axis('off')                       # 축 제거 (깔끔하게 보이게)
                    axes[i].set_title(f"Label: {row['target']}", fontsize=8)  # 라벨 표시

            except KeyError:
                # 만약 이미지가 zip 안에 없으면 오류 발생 → 대신 빈 칸 처리
                print(f"❌ 이미지 파일이 없음: {img_path}")
                axes[i].axis('off')
                axes[i].set_title("이미지 없음", fontsize=8)

        # [6] 출력한 이미지 수보다 subplot이 더 많을 경우 → 남은 칸은 비우기
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # [7] 전체 레이아웃 정리 + 화면에 출력
        plt.tight_layout()
        plt.show()

# ───────────────────────────────
# 4. 이미지 해상도 분석
def eda_image_resolution_stats(train_df, zip_path, sample_size=300):
    """
    📌 설명:
    이 함수는 zip 안에 있는 이미지들의 해상도(너비, 높이)를 확인해서
    평균, 최소, 최대 크기와 분포를 시각화해주는 함수야.

    - train_df: train.csv를 불러온 pandas DataFrame
    - zip_path: train.zip 압축 파일 경로
    - sample_size: 샘플링할 이미지 수 (전체 다 확인하면 느릴 수 있어서 일부만 봐도 충분함)
    """

    widths = []   # 이미지 너비 저장용 리스트
    heights = []  # 이미지 높이 저장용 리스트

    # zip 파일 열기
    with zipfile.ZipFile(zip_path, 'r') as archive:
        # 무작위로 sample_size만큼 ID 추출
        for img_id in train_df['ID'].sample(n=sample_size, random_state=42):
            
            # 확장자 중복 방지 처리
            if not img_id.endswith(".jpg"):
                img_name = img_id + ".jpg"
            else:
                img_name = img_id

            # zip 안의 경로는 train 폴더 안에 있음
            img_path = f"train/{img_name}"

            try:
                # 이미지 파일 열기
                with archive.open(img_path) as file:
                    img = Image.open(file)

                    # 너비와 높이 저장
                    widths.append(img.width)
                    heights.append(img.height)

            except KeyError:
                print(f"❌ 이미지 파일 누락: {img_path} (zip 안에 없음)")
                continue  # 누락된 경우 그냥 넘어감

    # 📏 통계 출력
    print("📏 이미지 해상도 통계:")
    print(f" - 평균 해상도: {int(np.mean(widths))} x {int(np.mean(heights))}")
    print(f" - 최소 해상도: {min(widths)} x {min(heights)}")
    print(f" - 최대 해상도: {max(widths)} x {max(heights)}")

    # 📊 분포 시각화 (히스토그램)
    # 📊 너비/높이를 따로 subplot으로 그려서 구분!
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 너비 히스토그램
    sns.histplot(widths, kde=True, color="skyblue", ax=ax1)
    ax1.set_title("이미지 너비(width) 분포")
    ax1.set_xlabel("픽셀")
    ax1.set_ylabel("Count")
    ax1.grid()

    # 높이 히스토그램
    sns.histplot(heights, kde=True, color="orange", ax=ax2)
    ax2.set_title("이미지 높이(height) 분포")
    ax2.set_xlabel("픽셀")
    ax2.set_ylabel("Count")
    ax2.grid()

    plt.tight_layout()
    plt.show()

def compute_classwise_brightness(train_df, TRAIN_ZIP_PATH, sample_per_class=10):
    """
    각 클래스(class_name)별로 이미지 몇 장씩 무작위로 샘플링해서
    평균 밝기를 계산하고, 그 결과를 데이터프레임으로 반환하는 함수
    """
    train_df = train_df.merge(meta_df, on="target", how="left")

    brightness_data = []

    # zip 압축된 이미지 파일 열기
    with zipfile.ZipFile(TRAIN_ZIP_PATH, 'r') as archive:
        # 클래스별 그룹화
        grouped = train_df.groupby('class_name')

        for class_name, group in grouped:
            # 각 클래스에서 sample_per_class 개수만큼 무작위 샘플링
            sampled = group.sample(n=min(sample_per_class, len(group)), random_state=42)

            for _, row in sampled.iterrows():
                img_id = row['ID']
                if not img_id.endswith(".jpg"):
                    img_id += ".jpg"

                img_path = f"train/{img_id}"

                try:
                    # 이미지 열고 흑백으로 변환 → 밝기 계산
                    with archive.open(img_path) as file:
                        img = Image.open(file).convert("L")  # "L"은 흑백 모드
                        brightness = np.mean(np.array(img))  # 픽셀 평균 (0~255)
                        brightness_data.append({
                            'class_name': class_name,
                            'brightness': brightness
                        })
                except:
                    continue  # 이미지 없으면 넘어감

    return pd.DataFrame(brightness_data)


#eda_basic_info(train_df, meta_df)
#eda_label_distribution(train_df, meta_df)
# eda_visualize_images(train_df, TRAIN_ZIP_PATH, samples_per_class=2)
# eda_image_resolution_stats(train_df, TRAIN_ZIP_PATH)

# brightness_df = compute_classwise_brightness(train_df, TRAIN_ZIP_PATH, sample_per_class=100)

# 클래스별 평균 밝기 시각화
# plt.figure(figsize=(23, 8))
# sns.barplot(x='class_name', y='brightness', data=brightness_df, estimator=np.mean, palette='coolwarm')
# plt.title("클래스별 평균 이미지 밝기")
# plt.xlabel("문서 클래스")
# plt.ylabel("평균 밝기 (0~255)")
# plt.xticks(rotation=60, ha='right')
# plt.grid(axis='y')
# plt.tight_layout()
# plt.show()

#-----------------------------------------
# ---------- Validation Set 생성----------
#-----------------------------------------

# ✅ 1. 데이터 로드 (EDA와 동일한 경로 사용)
train_df = pd.read_csv(r"C:\Users\재형띠\Desktop\코딩친구들\컴퓨터비전 프로젝트\datasets_folder\data\train.csv")
meta_df  = pd.read_csv(r"C:\Users\재형띠\Desktop\코딩친구들\컴퓨터비전 프로젝트\datasets_folder\data\meta.csv")

# ✅ 2. 클래스 이름(class_name) 붙이기 위해 merge
# train.csv에는 숫자형 target이 있고, meta.csv에는 target에 해당하는 class_name이 있음
merged_df = train_df.merge(meta_df, on='target', how='left')

# ✅ 3. Stratified 방식으로 Validation Set 생성
# - stratify 파라미터로 클래스(target) 분포 유지
# - random_state: 실험 재현성 고정
train_split, valid_split = train_test_split(
    merged_df,
    test_size=0.2,              # 전체 데이터 중 20%를 validation set으로 사용
    stratify=merged_df['target'],  # 클래스 비율 유지
    random_state=42             # 랜덤성 통제
)

# ✅ 4. 결과 확인 (클래스별 비율 비교)
print("✔️ [Train 클래스 분포]")
print(train_split['target'].value_counts(normalize=True).sort_index())

print("\n✔️ [Validation 클래스 분포]")
print(valid_split['target'].value_counts(normalize=True).sort_index())

# ✅ 5. 선택적으로 저장 가능 (학습에서 사용할 경우)
train_split.to_csv("train_split.csv", index=False)
valid_split.to_csv("valid_split.csv", index=False)