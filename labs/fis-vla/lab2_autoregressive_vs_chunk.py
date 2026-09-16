"""실습 2 — 자기회귀는 왜 느리고, 이산화는 왜 끊기는가 (가이드 1.4 · 1.7)

무엇을 느끼는가
  (1) 어텐션 비용은 토큰 수의 제곱으로 는다.
  (2) 행동 7개를 토큰으로 하나씩 뽑으면(자기회귀) 모델을 7번 돌린다. 한 번에 연속값 7개를 내면 1번이다.
  (3) 연속 행동을 256 구간으로 자르면(OpenVLA 방식) 궤적이 계단이 된다 — 논문 2절 "action discontinuities".

실행: python3 lab2_autoregressive_vs_chunk.py   (out/lab2_quantize.svg 생성)
"""
import os, time
import numpy as np
from tinynn import TransformerBlock, run_blocks, SVG

rng = np.random.default_rng(1)
d, n_blocks = 64, 4
blocks = [TransformerBlock(d, rng) for _ in range(n_blocks)]
os.makedirs("out", exist_ok=True)


def timeit(fn, reps=5):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best


# ── (1) 토큰 수와 비용 ────────────────────────────────────────────────────
print("== (1) 토큰 수 T에 따른 블록 4개 통과 시간 (어텐션은 T² 비례) ==")
base = None
for T in [64, 128, 256, 512]:
    x = rng.normal(0, 1, (T, d))
    t = timeit(lambda: run_blocks(blocks, x))
    base = base or t
    print(f"  T={T:4d}: {t*1000:7.2f} ms   (T=64 대비 ×{t/base:5.1f})")

# ── (2) 자기회귀 7번 vs 한 번에 7개 ──────────────────────────────────────
print("\n== (2) 7-DoF 행동을 내는 두 방식 ==")
T0 = 300  # 언어 + 이미지 패치 토큰 수 (실제 FiS-VLA도 수백 개)
ctx = rng.normal(0, 1, (T0, d))


def autoregressive_7_tokens():
    x = ctx
    for _ in range(7):                       # 토큰 하나 뽑고 → 뒤에 붙이고 → 다시 전체 통과
        out = run_blocks(blocks, x)
        new_tok = out[-1:]                   # 마지막 자리의 출력이 "다음 토큰"
        x = np.vstack([x, new_tok])
    return x


def one_shot_7_values():
    x = np.vstack([ctx, rng.normal(0, 1, (1, d))])   # 노이즈 행동 토큰 1개 추가
    out = run_blocks(blocks, x)
    return out[-1, :7]                       # 마지막 자리에서 연속값 7개를 한 번에

t_ar = timeit(autoregressive_7_tokens)
t_os = timeit(one_shot_7_values)
print(f"  자기회귀(토큰 7개 순차)   : {t_ar*1000:7.2f} ms → {1/t_ar:6.1f} Hz")
print(f"  한 번에 연속값 7개(확산 1회): {t_os*1000:7.2f} ms → {1/t_os:6.1f} Hz   (×{t_ar/t_os:.1f} 빠름)")
print("  ※ 실제 확산은 디노이징을 여러 번(저장소 --ddim-steps 4) 반복하지만, 토큰 7개를 순차로 뽑는 것과는 다른 축의 비용이다")
print("  ※ 논문 표 1: OpenVLA(토큰 자기회귀) 6.3 Hz vs FiS-VLA 21.9 Hz")

# ── (3) 이산화의 계단 ─────────────────────────────────────────────────────
print("\n== (3) 연속 행동을 구간으로 자르면 ==")
t = np.linspace(0, 1, 200)
smooth = 0.6 * np.sin(2 * np.pi * t) * np.exp(-t)     # 부드러운 Δx 궤적 (범위 -1~1)


def quantize(a, bins):
    edges = np.linspace(-1, 1, bins + 1)
    idx = np.clip(np.digitize(a, edges) - 1, 0, bins - 1)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers[idx]

for bins in [256, 16]:
    q = quantize(smooth, bins)
    print(f"  {bins:3d} 구간: 최대 오차 {np.abs(q-smooth).max():.4f}, 서로 다른 값 {len(np.unique(q))}개 (연속은 200개)")

svg = SVG(xlim=(0, 1), ylim=(-0.7, 0.7), title="연속 행동 vs 256/16 구간 이산화")
svg.line(list(zip(t, smooth)), "#2F3E9E", 2)
svg.line(list(zip(t, quantize(smooth, 16))), "#D9531E", 1.5)
svg.line(list(zip(t, quantize(smooth, 256))), "#7a7a7a", 1, dash="3 3")
svg.legend([("#2F3E9E", "연속 (확산 헤드가 내는 것)"), ("#D9531E", "16 구간 (계단이 보인다)"), ("#7a7a7a", "256 구간 (OpenVLA 방식, 확대하면 계단)")])
path = svg.save("out/lab2_quantize.svg")
print(f"  그림 저장: {path}  (브라우저로 열어 보기)")
