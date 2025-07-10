import pandas as pd
import os

# ✅ [ver1] seed 고정 + 무작위 셔플 기반 테스트셋 생성
def create_test_split_ver1(sample_submission_path, output_dir, seed=42):
    """
    [ver1] sample_submission.csv을 무작위 셔플하여 test용으로 저장
    - seed 고정으로 재현 가능성 확보

    Args:
        sample_submission_path (str): sample_submission.csv 경로
        output_dir (str): 결과 저장 폴더
        seed (int): random_state 고정

    Saves:
        test_ver1_seed_fixed.csv
    """
    df = pd.read_csv(sample_submission_path)
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # 무작위 셔플
    save_path = os.path.join(output_dir, "test_ver1_seed_fixed.csv")
    df_shuffled.to_csv(save_path, index=False)

# ✅ [ver2] meta.csv 병합 후 rare class 최소 1장 포함
def create_test_split_ver2(sample_submission_path, meta_path, output_dir):
    """
    [ver2] meta.csv에서 class_name을 기준으로 최소 1장 포함되도록 구성

    Args:
        sample_submission_path (str): sample_submission.csv 경로
        meta_path (str): meta.csv 경로
        output_dir (str): 결과 저장 폴더

    Saves:
        test_ver2_rare_class.csv
    """
    df = pd.read_csv(sample_submission_path)
    meta = pd.read_csv(meta_path)
    merged = df.merge(meta, on="image", how="left")  # class_name 포함

    rare_sample = merged.groupby("class_name").head(1)  # 각 class_name 최소 1장
    remaining = merged.drop(rare_sample.index).sample(frac=1, random_state=42)
    combined = pd.concat([rare_sample, remaining]).reset_index(drop=True)

    save_path = os.path.join(output_dir, "test_ver2_rare_class.csv")
    combined.to_csv(save_path, index=False)

# ✅ [ver3] meta 활용 + rare class 포함 + 시드 고정 셔플
def create_test_split_ver3(sample_submission_path, meta_path, output_dir, seed=42):
    """
    [ver3] meta 정보 + rare 보장 + 무작위성까지 포함된 완전한 test split 버전

    Args:
        sample_submission_path (str): sample_submission.csv 경로
        meta_path (str): meta.csv 경로
        output_dir (str): 결과 저장 폴더
        seed (int): 무작위 셔플 seed

    Saves:
        test_ver3_meta_rare.csv
    """
    df = pd.read_csv(sample_submission_path)
    meta = pd.read_csv(meta_path)
    merged = df.merge(meta, on="image", how="left")

    rare_sample = merged.groupby("class_name").head(1)
    remaining = merged.drop(rare_sample.index).sample(frac=1, random_state=seed)
    combined = pd.concat([rare_sample, remaining]).reset_index(drop=True)

    save_path = os.path.join(output_dir, "test_ver3_meta_rare.csv")
    combined.to_csv(save_path, index=False)

# ✅ 실행 구문
sample_csv = r"C:\Users\재형띠\Desktop\코딩친구들\컴퓨터비전 프로젝트\datasets_folder\data\sample_submission.csv"
meta_csv = r"C:\Users\재형띠\Desktop\코딩친구들\컴퓨터비전 프로젝트\datasets_folder\data\meta.csv"
save_dir = r"C:\Users\재형띠\Desktop\코딩친구들\컴퓨터비전 프로젝트\vaildation_set_test"

create_test_split_ver1(sample_csv, save_dir)
#create_test_split_ver2(sample_csv, meta_csv, save_dir)
#create_test_split_ver3(sample_csv, meta_csv, save_dir)
