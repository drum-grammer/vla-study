"""실습 5 — 공동학습과 파국적 망각: L = L_fast + L_slow 가 필요한 이유 (가이드 2.6)

무엇을 느끼는가
  System 1이 System 2의 마지막 블록을 "공유"하므로, 확산 손실(L_fast)만으로 학습하면 그 블록은
  노이즈 예측기로 변해 버리고 원래 하던 일(다음 토큰 예측 = 추론)을 잊는다.
  L_slow(교차엔트로피)를 함께 흘리면 두 일을 다 한다. 논문 4.2절 (4): L_slow 제거 시 0.69 → 0.62.

  저장소 대응: training/strategies/base_strategy.py
        loss, output = self.vlm(..., use_diff=True)      # L_fast (fisvla.py forward)
        if ar_diff_loss: loss = loss + output.loss       # + L_slow (LLM의 next-token cross-entropy)
  train.sh: AR_DIFF_LOSS=true

모형: 공유 몸통(블록 31·32 역할) + 머리 A(분류 = "이산 행동 토큰 맞히기", 교차엔트로피)
      + 머리 B(노이즈 예측 = 확산, MSE). 1단계: A로 사전학습. 2단계: B를 배우되 A 손실을 (i) 빼거나 (ii) 함께 둔다.

실행: python3 lab5_cotraining_forgetting.py   (약 10~20초)
"""
import numpy as np
from tinynn import MLP, Adam, softmax

rng = np.random.default_rng(5)
D_IN, N_CLS = 6, 4
W_true = rng.normal(0, 1, (D_IN, N_CLS))          # 과제 A: 어느 "이산 행동 토큰"인가 (숨은 규칙)
W_reg = rng.normal(0, 1, (D_IN, 2))               # 과제 B: 예측할 연속 노이즈의 숨은 규칙


def batch_A(n):
    x = rng.normal(0, 1, (n, D_IN))
    y = np.argmax(np.tanh(x @ W_true) + 0.3 * np.sin(x[:, :N_CLS]), 1)
    return x, y


def batch_B(n):
    x = rng.normal(0, 1, (n, D_IN))
    target = np.tanh(x @ W_reg) * 0.8 + 0.2 * np.cos(x[:, :2])
    return x, target


def make_model():
    trunk = MLP([D_IN, 32, 32], rng)      # 공유 몸통 (마지막 2블록). 작게 잡아 두 일이 자리를 다투게 한다
    head_a = MLP([32, N_CLS], rng)        # 자기회귀 머리 (lm_head)
    head_b = MLP([32, 2], rng)            # 확산 머리 (final_layer)
    return trunk, head_a, head_b


def acc_A(trunk, head_a, n=4000):
    x, y = batch_A(n)
    logits = head_a.forward(trunk.forward(x))
    return (logits.argmax(1) == y).mean()


def mse_B(trunk, head_b, n=4000):
    x, tg = batch_B(n)
    return ((head_b.forward(trunk.forward(x)) - tg) ** 2).mean()


def train_step(trunk, heads, opt, w_slow, w_fast):
    head_a, head_b = heads
    xa, ya = batch_A(128)
    xb, tb = batch_B(128)
    x = np.vstack([xa, xb])
    h = trunk.forward(x)
    dh = np.zeros_like(h)
    grads_a = grads_b = None
    if w_slow > 0:                                     # L_slow: 교차엔트로피 (앞 절반 배치)
        logits = head_a.forward(h[:128])
        p = softmax(logits)
        p[np.arange(128), ya] -= 1
        grads_a, dha = head_a.backward(w_slow * p / 128)
        dh[:128] += dha
    if w_fast > 0:                                     # L_fast: 노이즈 예측 MSE (뒤 절반 배치)
        pred = head_b.forward(h[128:])
        grads_b, dhb = head_b.backward(w_fast * 2 * (pred - tb) / tb.size)
        dh[128:] += dhb
    grads_t, _ = trunk.backward(dh)
    opt["trunk"].step(grads_t)
    if grads_a is not None: opt["a"].step(grads_a)
    if grads_b is not None: opt["b"].step(grads_b)


def run(mode):
    rng_state = np.random.get_state()
    trunk, head_a, head_b = make_model()
    opt = {"trunk": Adam(trunk.params(), 3e-3), "a": Adam(head_a.params(), 3e-3), "b": Adam(head_b.params(), 3e-3)}
    # 1단계: 사전학습 — System 2의 추론(다음 토큰) 능력만 갖춘 상태
    for _ in range(1500):
        train_step(trunk, (head_a, head_b), opt, w_slow=1.0, w_fast=0.0)
    acc0 = acc_A(trunk, head_a)
    # 2단계: 행동 생성(L_fast) 학습
    w_slow = 1.0 if mode == "co-training" else 0.0
    for k in ("trunk", "a", "b"):
        opt[k].lr = 6e-3                                   # 파인튜닝 학습률 (크면 망각이 빨라진다)
    for _ in range(4000):
        train_step(trunk, (head_a, head_b), opt, w_slow=w_slow, w_fast=1.0)
    return acc0, acc_A(trunk, head_a), mse_B(trunk, head_b)


print("== 1단계(사전학습) 후 → 2단계(행동 학습) 후, 과제 A 정확도 변화 ==")
print(f"  {'방식':<22}{'A 정확도(전)':>12}{'A 정확도(후)':>12}{'B 오차(후)':>12}")
for mode in ["L_fast only", "co-training"]:
    acc0, acc1, mse = run(mode)
    tag = "  ← 추론 능력을 잊었다 (파국적 망각)" if mode == "L_fast only" else "  ← 둘 다 한다 (dual-aware co-training)"
    print(f"  {mode:<22}{acc0*100:11.1f}%{acc1*100:11.1f}%{mse:12.4f}{tag}")
print("\n※ 논문: L_slow 없이 학습하면 RLBench 평균 0.69 → 0.62. 저장소: base_strategy.py 'loss = loss + output.loss'")
