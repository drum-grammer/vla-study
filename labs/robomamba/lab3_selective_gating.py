"""실습 3 · '선택적' SSM이 무엇을 선택하는가 — 가이드 1.5 · 2.2 · 2.4

논문 1절: "Mamba introduces the innovative selective State Space Model (SSM),
promoting context-aware reasoning while maintaining linear complexity."

이 '선택'이 없으면 정확히 무엇을 못 하는지 보인다.
과제는 선택적 복사(selective copy): 토큰 열 중 표시(marker)가 붙은 토큰의
값 하나만 끝까지 기억해 마지막 위치에서 내놓아야 한다. 나머지는 방해물이다.

  모델 A (비선택적, S4 계열): Δ가 입력과 무관한 상수
  모델 B (선택적,  S6 = Mamba): Δ가 입력의 함수 ← 논문이 쓰는 것

두 모델에 **똑같은 크기의 상태**와 **각자에게 최적인 선형 readout**을 준다.
readout은 최소제곱으로 닫힌 해를 구하므로 "학습이 덜 됐다"는 변명이 없다.
Δ도 격자탐색으로 각 모델의 최적값을 찾아 준다. 그래도 격차가 남으면
그것은 구조의 한계다.

실행: python3 lab3_selective_gating.py    (약 12초, out/lab3_selective.svg 생성)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tinyssm as T

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

L, N, NTRAIN, NTEST = 32, 8, 600, 600
A = -np.arange(1, N + 1, dtype=np.float64)   # 저장소의 S4D-real 초기값 A = −(1..N)

print(T.header(3, "'선택적' SSM이 무엇을 선택하는가", "1.5 · 2.2 · 2.4"))


def make(n, seed):
    """값 열 (n,L), 표시 마스크 (n,L), 정답 (n,)"""
    g = np.random.default_rng(seed)
    vals = g.uniform(-1, 1, (n, L))
    mk = g.integers(4, L - 4, n)             # 표시 위치는 예제마다 다르다
    mask = np.zeros((n, L))
    mask[np.arange(n), mk] = 1.0
    return vals, mask, vals[np.arange(n), mk]


def final_state(vals, delta):
    """tinyssm.selective_scan과 같은 순환식(B=C=1, 채널 1개, 상태 N개)의 최종 상태.
       h_t = exp(ΔA)·h_{t−1} + Δ·x_t   ← 논문 식 (2)(4)"""
    n = vals.shape[0]
    h = np.zeros((n, N))
    for t in range(L):
        d = delta[:, t][:, None]
        h = np.exp(d * A[None]) * h + d * vals[:, t][:, None]
    return h


def best_linear_readout(Htr, ytr, Hte, yte):
    """상태에서 정답으로 가는 최적 선형 사상을 최소제곱으로 구하고 테스트 MSE."""
    Xtr = np.concatenate([Htr, np.ones((len(Htr), 1))], axis=1)
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    Xte = np.concatenate([Hte, np.ones((len(Hte), 1))], axis=1)
    return float(np.mean((Xte @ w - yte) ** 2)), w


vtr, mtr, ytr = make(NTRAIN, 1)
vte, mte, yte = make(NTEST, 2)
var = float(np.var(yte))

print(f"""
과제  토큰 {L}개 중 '표시된' 토큰의 값 하나를 마지막 위치에서 내놓기
상태  두 모델 모두 숫자 {N}개로 고정 (A = −(1..{N}), 저장소 초기값과 같음)
평가  테스트 {NTEST}개의 MSE. 정답 분산 {var:.4f}이 '아무것도 못 배움' 기준선""")

# ── 모델 A: 비선택적. Δ 상수를 40개 후보에서 최적 선택 ──────────────────────
cands_a = np.logspace(-3, 1, 40)
curve_a = []
best_a = (float("inf"), None)
for d0 in cands_a:
    e, _ = best_linear_readout(final_state(vtr, np.full((NTRAIN, L), d0)), ytr,
                               final_state(vte, np.full((NTEST, L), d0)), yte)
    curve_a.append(e)
    if e < best_a[0]:
        best_a = (e, d0)

# ── 모델 B: 선택적. Δ_t = δ_lo + (δ_hi − δ_lo)·표시_t, 10×10 격자 ───────────
lo_grid, hi_grid = np.logspace(-3, -0.7, 10), np.logspace(-1, 1.2, 10)
best_b = (float("inf"), None, None)
for dlo in lo_grid:
    for dhi in hi_grid:
        e, _ = best_linear_readout(final_state(vtr, dlo + (dhi - dlo) * mtr), ytr,
                                   final_state(vte, dlo + (dhi - dlo) * mte), yte)
        if e < best_b[0]:
            best_b = (e, dlo, dhi)

rows = [
    ["A · 비선택적 (Δ 상수)", f"{len(cands_a)}개 후보 중 최적",
     f"Δ={best_a[1]:.4f}", f"{best_a[0]:.4f}", f"{best_a[0] / var * 100:5.1f}%"],
    ["B · 선택적 (Δ=f(표시))", f"{len(lo_grid) * len(hi_grid)}개 후보 중 최적",
     f"Δ_lo={best_b[1]:.4f} / Δ_hi={best_b[2]:.2f}",
     f"{best_b[0]:.4f}", f"{best_b[0] / var * 100:5.1f}%"],
]
print()
print(T.table(["모델", "Δ 탐색", "최적 Δ", "테스트 MSE", "정답 분산 대비"],
              rows, highlight={1}))

print(f"""
왜 비선택적 모델은 원리적으로 못 하는가
────────────────────────────────────────────────────────────────
Δ가 상수면 최종 상태는 h[j] = Σ_t Δ·exp(ΔA_j(L−1−t))·v_t 다.
즉 **위치에만 의존하는 고정 가중치로 모든 값을 더한 것** {N}개다.
선형 readout이 할 수 있는 일은 이 {N}개를 다시 섞는 것뿐이라,
결국 "v_t들의 고정 가중합"밖에 못 만든다. 그런데 정답은 예제마다
다른 위치의 값이다. 고정 가중합으로는 원리적으로 불가능하다
→ 최적값을 다 뒤져도 분산의 {best_a[0] / var * 100:.0f}%가 남는다.

선택적 모델은 표시 토큰에서만 Δ를 {best_b[2] / best_b[1]:.0f}배 키운다.
그 순간 exp(ΔA)≈0이 되어 **과거를 지우고**(경쟁자 제거) Δ·v를 크게 쓴다.
나머지 토큰에서는 Δ_lo={best_b[1]:.4f}로 작아 exp(ΔA)≈1, 즉 **그대로 보존**하며
새 입력도 거의 안 받는다. 실습 1 (3)의 '기억 반경'을 토큰마다 바꾼 것이다.
같은 상태 {N}개로 MSE {best_b[0]:.4f} — 분산의 {best_b[0] / var * 100:.1f}%.

이것이 논문이 Mamba를 고른 단 하나의 이유다. 로봇 문맥에서 그대로 읽으면
"last 20 steps: 1- open the drawer ..." 같은 긴 이력에서 지금 판단에
필요한 몇 토큰만 상태에 남기는 능력이고, 고정 크기 상태로 그걸 하니
문맥이 길어져도 비용이 늘지 않는다(실습 2).

저장소 대응 ({T.REPO})
────────────────────────────────────────────────────────────────
  Δ를 입력에서 만드는 경로 : modeling_mamba.py
      x_dbl = x_proj(conv1d_out) → dt, B, C로 split → dt_proj(dt) → softplus
  선택성이 붙는 축         : B, C가 (batch, N, L)로 토큰마다 다름
                             (S4는 (D, N) 하나로 고정 — 그것이 모델 A)
  LoRA가 건드리는 모듈     : vlm.py LinearVLM.lora()
      target_modules=["in_proj","dt_proj","x_proj","out_proj"]
      → 파인튜닝으로 '선택 규칙'을 바꾸는 지점이 정확히 dt_proj·x_proj다

주의 (이 실습의 한계)
  · 진짜 Mamba는 Δ를 표시 비트가 아니라 학습된 x_proj·dt_proj로 만든다.
    여기서는 최적 Δ 정책을 격자탐색으로 대신 찾아 '구조의 상한'을 비교했다.
  · 채널 1개·B=C=1로 줄였다. 실제는 d_inner=5120 채널이 각자 Δ를 갖는다.

해 볼 것
  · L을 64, 128로 늘리면 두 모델의 격차가 어떻게 되는가?
  · 표시를 두 개 주고 '두 번째 표시의 값'을 물으면 상태 {N}개로 되는가?
  · Δ_hi를 아주 크게(100) 주면 왜 다시 나빠지는가? (exp(ΔA)=0 → 이전 것 전멸)""")

# ─────────────────────────────────────────────────────────────────────────────
# 그림
# ─────────────────────────────────────────────────────────────────────────────
X0, Y0, X1, Y1 = 80, 54, 420, 268
body = T.axes(X0, Y0, X1, Y1, "상수 Δ (log)", "테스트 MSE")
lx = np.log10(cands_a)
ymax = var * 1.15
body += T.polyline([(X0 + (lx[i] - lx[0]) / (lx[-1] - lx[0]) * (X1 - X0),
                     Y1 - min(curve_a[i], ymax) / ymax * (Y1 - Y0))
                    for i in range(len(cands_a))], "#a8443f", 2)
yv = Y1 - var / ymax * (Y1 - Y0)
body += (f'<line x1="{X0}" y1="{yv:.0f}" x2="{X1}" y2="{yv:.0f}" stroke="#8a8a82" '
         f'stroke-dasharray="4 3"/><text x="{X0 + 6}" y="{yv - 6:.0f}" font-size="10" '
         f'fill="#55554f">정답 분산 = 아무것도 못 배움</text>')
yb = Y1 - best_b[0] / ymax * (Y1 - Y0)
body += (f'<line x1="{X0}" y1="{yb:.0f}" x2="{X1}" y2="{yb:.0f}" stroke="#2f6f4f" '
         f'stroke-width="2"/><text x="{X0 + 6}" y="{yb - 6:.0f}" font-size="10" '
         f'fill="#2f6f4f">선택적 모델 (MSE {best_b[0]:.4f})</text>')
body += (f'<text x="{X0 + 6}" y="{Y0 + 14}" font-size="11" fill="#a8443f">'
         f'■ 비선택적: Δ를 어떻게 골라도 분산 근처</text>')

# 오른쪽: 최적 선택적 Δ 프로파일
P0, P1 = 500, 840
dprof = (best_b[1] + (best_b[2] - best_b[1]) * mte)[0]
mpos = int(np.argmax(mte[0]))
body += T.axes(P0, Y0, P1, Y1, "토큰 위치", "Δ_t (선택적 모델 최적해)")
dmax = dprof.max() * 1.18
body += T.polyline([(P0 + t / (L - 1) * (P1 - P0), Y1 - dprof[t] / dmax * (Y1 - Y0))
                    for t in range(L)], "#2f6f4f", 2)
mx = P0 + mpos / (L - 1) * (P1 - P0)
body += (f'<line x1="{mx:.0f}" y1="{Y0}" x2="{mx:.0f}" y2="{Y1}" stroke="#a8443f" '
         f'stroke-dasharray="3 3"/><text x="{mx + 5:.0f}" y="{Y0 + 14}" font-size="10" '
         f'fill="#a8443f">표시 토큰 (Δ ×{best_b[2] / best_b[1]:.0f})</text>')
p = T.save_svg(os.path.join(OUT, "lab3_selective.svg"), 890, 310, body,
               "선택성이 있어야 고정 크기 상태로 '그 토큰 하나'를 기억한다")
print(f"\n그림 저장: {p}")
