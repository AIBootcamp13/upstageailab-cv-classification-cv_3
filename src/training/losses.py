# src/training/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# 파일이 잘 불러와졌는지 확인용 로그 출력 (디버깅에 유용)
print("✅ losses.py 불러와짐")

# ----------------------------
# ✅ Label Smoothing Cross Entropy Loss 클래스 정의
# ----------------------------
class LabelSmoothingCrossEntropy(nn.Module):
    # 초기화 함수: smoothing 하이퍼파라미터를 받음
    def __init__(self, smoothing=0.1):
        super().__init__()  # 부모 클래스 초기화
        self.smoothing = smoothing  # smoothing 값 저장 (보통 0.1 ~ 0.2)

    # 실제 손실 계산을 수행하는 forward 함수
    def forward(self, input, target):
        # input: 모델이 예측한 로짓 (batch_size, num_classes)
        # target: 실제 정답 인덱스 (batch_size)

        # 클래스 수 추출 (num_classes)
        num_classes = input.size(1)

        # 로짓에 log_softmax를 적용하여 로그 확률로 변환
        log_probs = F.log_softmax(input, dim=1)

        # 정답 분포를 스무딩하여 생성 (gradient 계산 제외)
        with torch.no_grad():
            # log_probs와 동일한 shape의 0으로 채워진 텐서 생성
            true_dist = torch.zeros_like(log_probs)

            # 정답이 아닌 클래스에 대해 smoothing 분배
            true_dist.fill_(self.smoothing / (num_classes - 1))

            # 정답 클래스 위치에는 1 - smoothing 값을 넣음
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            # 예: smoothing=0.1, 정답 class=2일 때: [0.033, 0.033, 0.9, 0.033, ...]

        # KL divergence를 유도하는 CrossEntropy 계산
        # (-스무딩된 정답 * log_probs)의 총합 → 평균
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

# ----------------------------
# ✅ Focal Loss 클래스 정의
# ----------------------------
class FocalLoss(nn.Module):
    # gamma는 "얼마나 어려운 샘플에 더 집중할 것인가"를 결정하는 하이퍼파라미터
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()  # 부모 클래스 초기화
        self.gamma = gamma  # 감쇠 지수 저장
        self.ce = nn.CrossEntropyLoss(weight=weight)  # 기본 CrossEntropy 손실을 내부에서 사용

    def forward(self, input, target):
        # input: 모델이 예측한 로짓 (batch_size, num_classes)
        # target: 실제 정답 인덱스 (batch_size)

        # 로짓을 log_softmax로 변환하여 로그 확률 계산
        logpt = F.log_softmax(input, dim=1)

        # 로그 확률을 지수화하여 실제 확률 pt 얻음
        pt = torch.exp(logpt)  # shape: (batch_size, num_classes)

        # focal_term 계산: 예측 확률이 높을수록 작아짐 (쉽게 맞춘 샘플은 loss 작게)
        focal_term = (1 - pt).pow(self.gamma)  # shape: (batch_size, num_classes)

        # CrossEntropyLoss는 내부적으로 softmax + log + loss까지 계산
        # 따라서 여기에 focal_term의 정답 클래스에 해당하는 값만 곱해줌
        loss = self.ce(input, target) * focal_term[range(len(target)), target]

        # 전체 배치의 평균 loss 반환
        return loss.mean()

# 클래스를 정상적으로 불러왔는지 확인하기 위한 출력
print("✅ LabelSmoothingCrossEntropy = ", LabelSmoothingCrossEntropy)