#!/bin/bash

# 로그 디렉토리 생성
LOG_DIR="../log"
mkdir -p $LOG_DIR

echo "ConvNeXtV2 Tiny 최적화 실험 시작: $(date)"
echo "========================================"

# GPU 메모리 체크 함수
check_gpu_memory() {
    echo "GPU 메모리 상태:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
}

check_gpu_memory

# 💡 모델 설정 개선 (더 높은 LR)
echo "[1/6] ConvNeXtV2 Tiny - High LR Variant"
python src/script/train.py \
    --model-name convnextv2_tiny \
    --learning-rate 1.5e-4 \
    --batch-size 64 \
    --epochs 80 \
    --weight-decay 0.05 \
    --drop-rate 0.1 \
    --drop-path-rate 0.1 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.2 \
    --cutmix-alpha 0.2 \
    --scheduler cosine \
    --warmup-epochs 5 \
    --seed 456 \
    2>&1 | tee "$LOG_DIR/tiny_high_lr.log"

# 🔧 MixUp/CutMix 비율 조정
echo "[2/6] ConvNeXtV2 Tiny - Lower Augmentation"
python src/script/train.py \
    --model-name convnextv2_tiny \
    --learning-rate 1e-4 \
    --batch-size 48 \
    --epochs 100 \
    --weight-decay 0.08 \
    --drop-rate 0.15 \
    --drop-path-rate 0.15 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.15 \
    --cutmix-alpha 0.15 \
    --scheduler cosine \
    --warmup-epochs 7 \
    --seed 789 \
    2>&1 | tee "$LOG_DIR/tiny_lower_aug.log"

# 🚀 대용량 배치 실험
echo "[3/6] ConvNeXtV2 Tiny - Large Batch"
python src/script/train.py \
    --model-name convnextv2_tiny \
    --learning-rate 1.5e-4 \
    --batch-size 96 \
    --epochs 80 \
    --weight-decay 0.08 \
    --drop-rate 0.2 \
    --drop-path-rate 0.15 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.3 \
    --cutmix-alpha 0.3 \
    --scheduler cosine \
    --warmup-epochs 10 \
    --seed 999 \
    2>&1 | tee "$LOG_DIR/tiny_large_batch.log"

# 🎯 CosineAnnealingWarmRestarts 실험
echo "[4/6] ConvNeXtV2 Tiny - Warm Restarts"
python src/script/train.py \
    --model-name convnextv2_tiny \
    --learning-rate 1e-4 \
    --batch-size 48 \
    --epochs 120 \
    --weight-decay 0.08 \
    --drop-rate 0.2 \
    --drop-path-rate 0.15 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.3 \
    --cutmix-alpha 0.3 \
    --scheduler cosine_warm_restarts \
    --warmup-epochs 7 \
    --seed 1024 \
    2>&1 | tee "$LOG_DIR/tiny_warm_restarts.log"

# 🔍 정규화 감소 실험
echo "[5/6] ConvNeXtV2 Tiny - Less Regularization"
python src/script/train.py \
    --model-name convnextv2_tiny \
    --learning-rate 1.2e-4 \
    --batch-size 64 \
    --epochs 100 \
    --weight-decay 0.05 \
    --drop-rate 0.1 \
    --drop-path-rate 0.1 \
    --label-smoothing 0.05 \
    --mixup-alpha 0.1 \
    --cutmix-alpha 0.1 \
    --scheduler cosine \
    --warmup-epochs 5 \
    --seed 2024 \
    2>&1 | tee "$LOG_DIR/tiny_less_reg.log"

# 🎲 앙상블용 다른 시드
echo "[6/6] ConvNeXtV2 Tiny - Ensemble Seed"
python src/script/train.py \
    --model-name convnextv2_tiny \
    --learning-rate 1e-4 \
    --batch-size 48 \
    --epochs 100 \
    --weight-decay 0.08 \
    --drop-rate 0.2 \
    --drop-path-rate 0.15 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.3 \
    --cutmix-alpha 0.3 \
    --scheduler cosine \
    --warmup-epochs 7 \
    --seed 31415 \
    2>&1 | tee "$LOG_DIR/tiny_ensemble_seed.log"

echo "========================================"
echo "모든 실험 완료: $(date)"

# GPU 메모리 최종 상태
echo "최종 GPU 메모리 상태:"
check_gpu_memory

echo "✅ Tiny 모델 최적화 실험 완료!"
echo "📁 로그 위치: $LOG_DIR"
echo "📊 결과 파일: $LOG_DIR/tiny_experiment_results.csv"

# 추가: 최고 성능 모델 자동 선택
echo ""
echo "🔍 최고 성능 체크포인트 찾기..."
find ../checkpoints -name "*.ckpt" -type f -exec ls -la {} \; | sort -k5 -nr | head -5
