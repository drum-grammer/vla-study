"""실습 1 · 상태공간모델을 손으로 굴려 보기 — 가이드 1.4 · 1.5 · 2.4

논문 3.1절의 식 (1)~(4)가 실제로 무엇을 계산하는지 숫자로 확인한다.
    h'(t) = A h(t) + B x(t);  y(t) = C h(t)      ... 식 (1)  연속
    Ā = exp(ΔA)                                   ... 식 (2)  이산화
    B̄ = (ΔA)^-1 (exp(ΔA) − I) · ΔB               ... 식 (3)  이산화
    h_t = Ā h_{t-1} + B̄ x_t;  y_t = C h_t        ... 식 (4)  순환

실행: python3 lab1_ssm_recurrence.py     (약 1초, out/lab1_state.svg 생성)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tinyssm as T

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

print(T.header(1, "상태공간모델을 손으로 굴려 보기", "1.4 · 1.5 · 2.4"))

# ─────────────────────────────────────────────────────────────────────────────
# (1) 이산화 — 식 (2)(3)이 하는 일
# ─────────────────────────────────────────────────────────────────────────────
print("""
(1) 이산화: 연속 시스템을 '한 토큰 = 한 스텝'으로 바꾸기
────────────────────────────────────────────────────────────────
식 (1)은 시간이 연속인 미분방정식이다. 토큰은 띄엄띄엄 오니까
'Δ만큼의 시간 동안 x가 일정했다'고 보고(zero-order hold) 적분해 둔다.
그 결과가 식 (2)(3)의 Ā, B̄이고, 식 (4)는 그냥 for 문이다.""")

A_scalar, B_scalar = -1.0, 1.0     # 상태 1개짜리 최소 예제
rows = []
for d in (0.001, 0.01, 0.1, 1.0, 5.0):
    Abar = np.exp(d * A_scalar)                             # 식 (2)
    Bbar_exact = (1.0 / (d * A_scalar)) * (Abar - 1.0) * d * B_scalar  # 식 (3)
    Bbar_code = d * B_scalar                                # 저장소가 쓰는 근사
    rows.append([f"{d:g}", f"{Abar:.4f}", f"{Bbar_exact:.4f}", f"{Bbar_code:.4f}",
                 f"{abs(Bbar_exact - Bbar_code) / abs(Bbar_exact) * 100:5.1f}%"])
print()
print(T.table(["Δ", "Ā=exp(ΔA)", "B̄ 식(3) 정확", "B̄ 저장소 근사(ΔB)", "차이"], rows))
print("""
읽는 법 (A = −1로 고정했을 때)
  · Ā는 '과거를 얼마나 남기나'. Δ가 작으면 1에 가깝고(거의 그대로 보존),
    Δ가 크면 0에 가깝다(과거를 지운다). A가 음수라서 항상 0~1 사이다.
  · B̄는 '지금 입력을 얼마나 받나'. Δ가 크면 크다.
  · 즉 Δ 하나가 "지금 것을 받아들이고 과거를 지운다"(Δ 큼) ↔
    "지금 것을 무시하고 과거를 지킨다"(Δ 작음)를 동시에 조절한다.
  · 저장소는 식 (3)을 정확히 계산하지 않고 B̄ ≈ Δ·B로 쓴다.
    Δ가 작은 영역(0.001~0.1, 실제 학습된 범위)에서 오차가 5% 미만이라
    Mamba 원 논문부터 이 근사를 쓴다. 저장소 확인 위치:
    modeling_mamba.py selective_scan_ref() 중 deltaB_u = einsum('bdl,bnl,bdl->bdln')""")

# ─────────────────────────────────────────────────────────────────────────────
# (2) 상태는 고정 크기 — 여기서 '선형 복잡도'가 나온다
# ─────────────────────────────────────────────────────────────────────────────
print("""
(2) 시퀀스가 길어져도 상태 크기는 그대로
────────────────────────────────────────────────────────────────""")
d_model = 32
W = T.mamba_params(d_model, seed=1)
rows = []
for L in (8, 64, 256, 1024):
    x = rng.normal(0, 1, (L, d_model))
    y, h = T.mamba_block(x, W)
    kv = 2 * L * d_model            # 트랜스포머가 들고 있어야 하는 K·V 캐시 크기
    rows.append([L, f"{y.shape}", f"{h.shape}", h.size, kv, f"{kv / h.size:.1f}×"])
print(T.table(["L(토큰 수)", "출력 모양", "상태 h 모양", "상태 숫자 개수",
               "트랜스포머 KV 캐시", "배"], rows))
print(f"""
상태 h는 (d_inner={W['d_inner']}, N={W['d_state']}) = {W['d_inner'] * W['d_state']}개 숫자로
L과 무관하게 고정이다. 토큰을 하나 더 넣는 비용도 항상 같다 → O(L).
트랜스포머는 토큰이 늘면 KV 캐시가 같이 늘고, 어텐션이 L²에 비례한다.
논문 초록의 "linear inference complexity"가 이 표의 4·5열 '배' 변화다.
(토큰이 아주 적으면 SSM 상태가 오히려 더 크다. 중요한 건 절대 크기가 아니라
 L이 늘 때 한쪽은 그대로, 다른 쪽은 비례해 커진다는 점이다.)
실제 값: d_model=2560 → 상태 5120×16 = 81,920개 (블록당), 64블록 = 524만개 고정.""")

# ─────────────────────────────────────────────────────────────────────────────
# (3) Δ가 '선택'하는 것을 눈으로 — 같은 입력, 다른 Δ
# ─────────────────────────────────────────────────────────────────────────────
print("""
(3) 같은 입력에 Δ만 바꿔 보기 — 기억이 남는 길이가 달라진다
────────────────────────────────────────────────────────────────""")
L, N = 60, 1
u = np.zeros((L, 1))
u[5] = 1.0          # 5번 토큰에만 정보 한 방울
A = np.array([[-1.0]])
Bc = np.ones((L, N))
Cc = np.ones((L, N))

curves, rows = {}, []
for d_val in (0.02, 0.1, 0.5, 2.0):
    delta = np.full((L, 1), d_val)
    y, _, states = T.selective_scan(u, delta, A, Bc, Cc, return_states=True)
    tr = states[:, 0, 0]
    peak = tr.max()
    # 신호가 최고값의 10%까지 줄어드는 데 걸린 토큰 수 = '기억 반경'
    after = tr[5:]
    life = int(np.argmax(after < peak * 0.1)) if (after < peak * 0.1).any() else L - 5
    curves[d_val] = tr
    rows.append([f"{d_val:g}", f"{np.exp(d_val * -1.0):.4f}", f"{peak:.4f}", life])
print(T.table(["Δ", "Ā=exp(ΔA)", "5번 토큰 직후 상태", "10%까지 남는 토큰 수"], rows))
print("""
Δ=0.02면 정보가 100토큰 넘게 살아 있고(장기 기억), Δ=2.0이면 2토큰 만에 사라진다.
Mamba의 S6는 이 Δ를 **입력이 정하게** 만든 것이다(Δ = softplus(dt_proj(x_proj(x)))).
"지금 토큰이 중요하다" → Δ를 키워 상태를 갈아 끼우고,
"관계없는 토큰이다" → Δ를 줄여 지나 보낸다. 논문 3.1절이 말하는
'content-aware reasoning'이 이것이고, 실습 3에서 학습으로 확인한다.""")

# ─────────────────────────────────────────────────────────────────────────────
# (4) 저장소 대응
# ─────────────────────────────────────────────────────────────────────────────
print(f"""
(4) 저장소 대응 — {T.REPO}
────────────────────────────────────────────────────────────────
  논문 식 (2) Ā=exp(ΔA)  → modeling_mamba.py  deltaA = torch.exp(einsum('bdl,dn->bdln', delta, A))
  논문 식 (3) B̄          → 같은 함수          deltaB_u = einsum('bdl,bnl,bdl->bdln', delta, B, u)
  논문 식 (4) 순환        → 같은 함수          x = deltaA[:,:,i] * x + deltaB_u[:,:,i]
  A가 음수인 근거         → MyMamba.forward()  A = -torch.exp(self.A_log.float())
  A의 초기값 (S4D-real)   → MyMamba.__init__() A_log = log(arange(1, d_state+1)) → A = −(1..16)
  Δ를 입력에서 만드는 곳  → MyMamba.__init__() x_proj(d_inner → dt_rank+2N), dt_proj(dt_rank → d_inner)

해 볼 것
  · A의 초기값을 −1..−16이 아니라 전부 −1로 바꾸면 (3)의 '기억 반경'이
    16개 채널에서 다 같아진다. 왜 굳이 1..16으로 흩어 놓았을까?
  · Δ를 (L,1) 상수가 아니라 u에 비례하게 주면 무슨 일이 생기는가?""")

# ─────────────────────────────────────────────────────────────────────────────
# 그림
# ─────────────────────────────────────────────────────────────────────────────
X0, Y0, X1, Y1 = 70, 50, 700, 300
body = T.axes(X0, Y0, X1, Y1, "토큰 위치 t (5번 토큰에만 입력 1.0)", "상태 h_t")
body += (f'<line x1="{X0 + 5 / L * (X1 - X0):.0f}" y1="{Y0}" '
         f'x2="{X0 + 5 / L * (X1 - X0):.0f}" y2="{Y1}" stroke="#c9c9c0" '
         f'stroke-dasharray="3 3"/>')
colors = {0.02: "#2f6f4f", 0.1: "#3f6fa8", 0.5: "#b8863f", 2.0: "#a8443f"}
ymax = max(c.max() for c in curves.values()) * 1.1
for i, (d_val, tr) in enumerate(curves.items()):
    pts = [(X0 + t / L * (X1 - X0), Y1 - tr[t] / ymax * (Y1 - Y0)) for t in range(L)]
    body += T.polyline(pts, colors[d_val], 2)
    body += (f'<text x="{X1 + 6}" y="{Y1 - tr[6] / ymax * (Y1 - Y0) + 4}" font-size="11" '
             f'fill="{colors[d_val]}">Δ={d_val:g}</text>')
p = T.save_svg(os.path.join(OUT, "lab1_state.svg"), 800, 340, body,
               "Δ 하나가 기억의 길이를 정한다 — h_t = exp(ΔA)·h_{t−1} + ΔB·x_t")
print(f"\n그림 저장: {p}")
