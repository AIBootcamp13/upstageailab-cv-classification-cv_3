import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 저장 경로 설정
SAVE_DIR = r"C:\Users\재형띠\Desktop\코딩친구들\컴퓨터비전 프로젝트\vaildation_set_test"
os.makedirs(SAVE_DIR, exist_ok=True)

# train.csv에는 이미지 ID와 라벨(label) 정보가 들어있다.
# meta.csv에는 각 이미지의 나이, 성별, 인종 등의 부가정보가 있다.
train_df = pd.read_csv(r"C:\Users\재형띠\Desktop\코딩친구들\컴퓨터비전 프로젝트\datasets_folder\data\train.csv")
meta_df = pd.read_csv(r"C:\Users\재형띠\Desktop\코딩친구들\컴퓨터비전 프로젝트\datasets_folder\data\meta.csv")

# 함수 1: seed 고정 stratified split
def val_split_seed_fixed(train_df):
    """
    [ver1] seed 고정만 수행한 Stratified Split 함수
    - 클래스 비율을 유지하면서 데이터를 8:2로 나눔
    - rare class 처리나 meta.csv 병합은 없음
    """
    # Stratified split: 각 클래스 비율을 유지하며 분할
    train_split, valid_split = train_test_split(
        train_df,
        test_size=0.2,                 # validation 데이터 비율 = 20%
        stratify=train_df['target'],  # 클래스 비율을 유지
        random_state=42               # seed 고정 (결과 재현 가능)
    )

    # 분할된 데이터 저장
    train_split.to_csv(os.path.join(SAVE_DIR, "ver1_seed_fixed_train.csv"), index=False)
    valid_split.to_csv(os.path.join(SAVE_DIR, "ver1_seed_fixed_valid.csv"), index=False)

    # 클래스 분포 확인
    print("[ver1] Split 완료: seed 고정, 클래스 비율 유지")
    print("Train 분포:\n", train_split['target'].value_counts(normalize=True).sort_index())
    print("Valid 분포:\n", valid_split['target'].value_counts(normalize=True).sort_index())


# 함수 2: seed 고정 + rare class 최소 보장
def val_split_with_rare_class(train_df, min_samples=1):
    """
    [ver2] rare class를 train에만 포함하고 validation에는 포함하지 않도록 분리
    - rare class: 전체 샘플 수가 min_samples 이하인 클래스
    """
    # 클래스별 샘플 개수 계산
    class_counts = train_df['target'].value_counts()

    # min_samples 이하인 클래스를 rare class로 간주
    rare_classes = class_counts[class_counts <= min_samples].index

    # rare class만 따로 분리
    rare_df = train_df[train_df['target'].isin(rare_classes)]
    rest_df = train_df[~train_df['target'].isin(rare_classes)]  # 나머지 클래스

    # rare class 제외한 데이터로 stratified split
    train_rest, valid_rest = train_test_split(
        rest_df,
        test_size=0.2,
        stratify=rest_df['target'],
        random_state=42
    )

    # rare class는 train에만 포함
    train_final = pd.concat([train_rest, rare_df], ignore_index=True)
    valid_final = valid_rest.copy()

    # 결과 저장
    train_final.to_csv(os.path.join(SAVE_DIR, "ver2_rare_class_train.csv"), index=False)
    valid_final.to_csv(os.path.join(SAVE_DIR, "ver2_rare_class_valid.csv"), index=False)

    print("[ver2] Split 완료: rare class 최소 보장")
    print("Train 분포:\n", train_final['target'].value_counts(normalize=True).sort_index())
    print("Valid 분포:\n", valid_final['target'].value_counts(normalize=True).sort_index())

# 함수 3: meta.csv 병합 후 class_name 기준 stratified split
def val_split_with_meta(train_df, meta_df, min_samples=1):
    """
    [ver3] meta.csv 병합 후 class_name 기준으로 stratified split
    - rare class는 train에만 포함되도록 분리
    """
    # train_df에 meta 정보(class_name 등) 병합
    merged_df = train_df.merge(meta_df, on='target', how='left')

    # class_name(성별+나이 등)의 샘플 수 기준 rare class 추출
    class_counts = merged_df['class_name'].value_counts()
    rare_classes = class_counts[class_counts <= min_samples].index

    # rare class만 따로 분리
    rare_df = merged_df[merged_df['class_name'].isin(rare_classes)]
    rest_df = merged_df[~merged_df['class_name'].isin(rare_classes)]

    # class_name 기준 stratified split
    train_rest, valid_rest = train_test_split(
        rest_df,
        test_size=0.2,
        stratify=rest_df['class_name'],
        random_state=42
    )

    train_final = pd.concat([train_rest, rare_df], ignore_index=True)
    valid_final = valid_rest.copy()

    # 저장
    train_final.to_csv(os.path.join(SAVE_DIR, "ver3_with_meta_train.csv"), index=False)
    valid_final.to_csv(os.path.join(SAVE_DIR, "ver3_with_meta_valid.csv"), index=False)

    print("[ver3] Split 완료: meta 활용 및 rare class 보장")
    print("Train 분포:\n", train_final['target'].value_counts(normalize=True).sort_index())
    print("Valid 분포:\n", valid_final['target'].value_counts(normalize=True).sort_index())

# 함수 4: 기존 baseline 구조를 그대로 함수화
def val_split_baseline(train_df, meta_df):
    """
    [ver4] baseline 방식: meta.csv 병합만 한 뒤 target 기준 stratified split
    """
    # meta 정보 병합
    merged_df = train_df.merge(meta_df, on='target', how='left')

    # target 기준 stratified split (class_name은 사용하지 않음)
    train_split, valid_split = train_test_split(
        merged_df,
        test_size=0.2,
        stratify=merged_df['target'],
        random_state=42
    )

    # 저장
    train_split.to_csv(os.path.join(SAVE_DIR, "ver4_baseline_train.csv"), index=False)
    valid_split.to_csv(os.path.join(SAVE_DIR, "ver4_baseline_valid.csv"), index=False)

    print("[ver4] Split 완료: baseline 기준")
    print("Train 분포:\n", train_split['target'].value_counts(normalize=True).sort_index())
    print("Valid 분포:\n", valid_split['target'].value_counts(normalize=True).sort_index())


#val_split_seed_fixed(train_df)
#val_split_with_rare_class(train_df, min_samples=1)
val_split_with_meta(train_df, meta_df, min_samples=1)
#val_split_baseline(train_df, meta_df)