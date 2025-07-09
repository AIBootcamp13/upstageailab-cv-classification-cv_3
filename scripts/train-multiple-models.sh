#!/bin/bash

# 로그 디렉토리 생성
LOG_DIR="../log"
mkdir -p $LOG_DIR

echo "RTX 3090 실험 시작: $(date)"
echo "========================================"

# RTX 3090 메모리 체크 함수
check_gpu_memory() {
    echo "GPU 메모리 상태:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
}

check_gpu_memory

# ConvNeXtV2 Tiny 실험들
echo "[1/8] ConvNeXtV2 Tiny - 빠른 베이스라인 (개선)"
python train.py \
    --model-name convnextv2_tiny \
    --learning-rate 1.5e-4 \
    --batch-size 64 \
    --epochs 50 \
    --weight-decay 0.05 \
    --drop-rate 0.1 \
    --drop-path-rate 0.1 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.2 \
    --warmup-epochs 5 \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/tiny_baseline_improved.log"

echo "[2/8] ConvNeXtV2 Tiny - 강한 정규화 (수정)"
python train.py \
    --model-name convnextv2_tiny \
    --learning-rate 1e-4 \
    --batch-size 48 \
    --epochs 70 \
    --weight-decay 0.08 \
    --drop-rate 0.2 \
    --drop-path-rate 0.15 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.3 \
    --cutmix-alpha 0.3 \
    --warmup-epochs 7 \
    --seed 123 \
    2>&1 | tee "$LOG_DIR/tiny_strong_reg_fixed.log"

echo "[3/8] ConvNeXtV2 Tiny - 효율적 대용량 배치 (수정)"
python train.py \
    --model-name convnextv2_tiny \
    --learning-rate 2e-4 \
    --batch-size 80 \
    --epochs 60 \
    --weight-decay 0.05 \
    --drop-rate 0.1 \
    --drop-path-rate 0.1 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.15 \
    --scheduler cosine_warm_restarts \
    --warmup-epochs 5 \
    --seed 999 \
    2>&1 | tee "$LOG_DIR/tiny_efficient_batch.log"

echo "[4/8] ConvNeXtV2 Tiny - 증강 비교 베이스라인"
python train.py \
    --model-name convnextv2_tiny \
    --learning-rate 1.2e-4 \
    --batch-size 64 \
    --epochs 60 \
    --weight-decay 0.05 \
    --drop-rate 0.15 \
    --drop-path-rate 0.1 \
    --label-smoothing 0.05 \
    --mixup-alpha 0.0 \
    --cutmix-alpha 0.0 \
    --warmup-epochs 5 \
    --seed 2024 \
    2>&1 | tee "$LOG_DIR/tiny_minimal_aug.log"

# ConvNeXtV2 Base 실험들 (RTX 3090 최적화)
echo "[5/8] ConvNeXtV2 Base - RTX 3090 최적화"
python train.py \
    --model-name convnextv2_base \
    --learning-rate 8e-5 \
    --batch-size 20 \
    --epochs 80 \
    --weight-decay 0.08 \
    --drop-rate 0.15 \
    --drop-path-rate 0.2 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.2 \
    --cutmix-alpha 0.2 \
    --warmup-epochs 8 \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/base_rtx3090_optimized.log"

echo "[6/8] ConvNeXtV2 Base - 안전한 최대 배치 (수정)"
python train.py \
    --model-name convnextv2_base \
    --learning-rate 6e-5 \
    --batch-size 24 \
    --epochs 100 \
    --weight-decay 0.1 \
    --drop-rate 0.2 \
    --drop-path-rate 0.25 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.3 \
    --cutmix-alpha 0.3 \
    --warmup-epochs 10 \
    --seed 123 \
    2>&1 | tee "$LOG_DIR/base_safe_max.log"

echo "[7/8] ConvNeXtV2 Base - 고성능 긴 학습 (개선)"
python train.py \
    --model-name convnextv2_base \
    --learning-rate 4e-5 \
    --batch-size 16 \
    --epochs 120 \
    --weight-decay 0.1 \
    --drop-rate 0.25 \
    --drop-path-rate 0.3 \
    --label-smoothing 0.15 \
    --mixup-alpha 0.4 \
    --cutmix-alpha 0.4 \
    --scheduler cosine \
    --warmup-epochs 12 \
    --seed 999 \
    2>&1 | tee "$LOG_DIR/base_high_performance.log"

echo "[8/8] ConvNeXtV2 Base - 균형잡힌 Step LR (개선)"
python train.py \
    --model-name convnextv2_base \
    --learning-rate 8e-5 \
    --batch-size 18 \
    --epochs 80 \
    --weight-decay 0.06 \
    --drop-rate 0.15 \
    --drop-path-rate 0.2 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.25 \
    --cutmix-alpha 0.25 \
    --scheduler step \
    --warmup-epochs 5 \
    --seed 2024 \
    2>&1 | tee "$LOG_DIR/base_balanced_step.log"

echo "========================================"
echo "모든 실험 완료: $(date)"

# GPU 메모리 최종 상태
echo "최종 GPU 메모리 상태:"
check_gpu_memory

# 향상된 결과 요약 스크립트
cat > "$LOG_DIR/analyze_results.py" << 'EOF'
import os
import re
import pandas as pd
from pathlib import Path

def extract_metrics(log_file):
    """로그 파일에서 최고 성능 메트릭 추출"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # val_f1 점수들 추출
        f1_scores = re.findall(r'val_f1.*?(\d+\.\d+)', content)
        f1_scores = [float(score) for score in f1_scores]

        # val_accuracy 추출
        acc_scores = re.findall(r'val_accuracy.*?(\d+\.\d+)', content)
        acc_scores = [float(score) for score in acc_scores]

        # val_loss 추출
        loss_scores = re.findall(r'val_loss.*?(\d+\.\d+)', content)
        loss_scores = [float(score) for score in loss_scores]

        return {
            'best_f1': max(f1_scores) if f1_scores else 0,
            'best_accuracy': max(acc_scores) if acc_scores else 0,
            'best_loss': min(loss_scores) if loss_scores else float('inf'),
            'final_f1': f1_scores[-1] if f1_scores else 0,
        }
    except:
        return {
            'best_f1': 0,
            'best_accuracy': 0,
            'best_loss': float('inf'),
            'final_f1': 0,
        }

def analyze_experiments():
    """실험 결과 분석"""
    log_dir = Path('.')
    results = []

    for log_file in log_dir.glob('*.log'):
        exp_name = log_file.stem
        metrics = extract_metrics(log_file)

        # 설정 정보 추출 (로그에서)
        with open(log_file, 'r') as f:
            first_lines = f.read()[:2000]  # 첫 부분만

        # 배치 크기, 학습률 등 추출
        batch_size = re.search(r'batch-size (\d+)', first_lines)
        learning_rate = re.search(r'learning-rate ([\d.e-]+)', first_lines)

        results.append({
            'experiment': exp_name,
            'best_f1': metrics['best_f1'],
            'best_accuracy': metrics['best_accuracy'],
            'final_f1': metrics['final_f1'],
            'best_loss': metrics['best_loss'],
            'batch_size': int(batch_size.group(1)) if batch_size else 'N/A',
            'learning_rate': learning_rate.group(1) if learning_rate else 'N/A',
        })

    # DataFrame으로 변환 및 정렬
    df = pd.DataFrame(results)
    df = df.sort_values('best_f1', ascending=False)

    print("🏆 실험 결과 랭킹 (Best F1 기준)")
    print("=" * 80)
    print(df.to_string(index=False, float_format='%.4f'))

    # Top 3 추천
    print("\n🥇 Top 3 추천 설정:")
    print("-" * 40)
    for i, row in df.head(3).iterrows():
        print(f"{i+1}. {row['experiment']}: F1={row['best_f1']:.4f}")

    # 결과 저장
    df.to_csv('experiment_results.csv', index=False)
    print(f"\n📊 상세 결과: experiment_results.csv")

if __name__ == "__main__":
    analyze_experiments()
EOF

# 결과 분석 실행
cd $LOG_DIR
python analyze_results.py
cd ..

echo "✅ 실험 완료 및 결과 분석 완료!"
echo "📁 로그 위치: $LOG_DIR"
echo "📊 결과 파일: $LOG_DIR/experiment_results.csv"
