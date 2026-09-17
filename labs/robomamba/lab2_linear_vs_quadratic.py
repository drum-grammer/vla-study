"""실습 2 · 어텐션은 왜 느리고 SSM은 왜 빠른가 — 가이드 1.3 · 1.5 · 2.6

논문의 속도 주장(그림 1: RoboMamba 9.0 Hz vs OpenVLA 3.4 Hz vs ManipLLM 1.1 Hz,
4.2절: "7 times faster than LLaMA-AdapterV2")이 어디서 오는지 직접 재 본다.

실행: python3 lab2_linear_vs_quadratic.py   (약 15초, out/lab2_scaling.svg 생성)
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tinyssm as T

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

print(T.header(2, "어텐션은 왜 느리고 SSM은 왜 빠른가", "1.3 · 1.5 · 2.6"))

# ─────────────────────────────────────────────────────────────────────────────
# (1) 토큰 수를 두 배씩 늘리며 한 블록의 시간을 잰다
# ─────────────────────────────────────────────────────────────────────────────
D = 64
Wq, Wk, Wv, Wo = (rng.normal(0, D ** -0.5, (D, D)) for _ in range(4))
Wm = T.mamba_params(D, seed=1)

print("""
(1) 블록 하나를 통과하는 시간 (numpy 단일 스레드, 상대 비교용)
────────────────────────────────────────────────────────────────""")
rows, meas = [], {}
base_att = base_ssm = None
for L in (64, 128, 256, 512, 1024, 2048):
    x = rng.normal(0, 1, (L, D))
    t_att = T.timeit(lambda: T.causal_attention(x, Wq, Wk, Wv, Wo))
    t_ssm = T.timeit(lambda: T.mamba_block(x, Wm))
    if base_att is None:
        base_att, base_ssm = t_att, t_ssm
    meas[L] = (t_att, t_ssm)
    rows.append([L, f"{t_att:8.2f}", f"×{t_att / base_att:5.1f}",
                 f"{t_ssm:8.2f}", f"×{t_ssm / base_ssm:5.1f}",
                 f"{L * L:,}", f"{L:,}"])
print(T.table(["L", "어텐션 ms", "L=64 대비", "SSM ms", "L=64 대비",
               "L² (이론)", "L (이론)"], rows))
print("""
어텐션은 L을 32배 늘리면 시간이 수백 배로 뛴다(L² 항). SSM은 거의 32배다.
여기서 SSM의 절대 시간이 더 큰 것은 numpy for 문 때문이고(실제로는 CUDA
병렬 스캔 커널이 담당), 비교해야 할 것은 '증가율'이다.
로봇에서 이 차이가 중요한 이유: 이미지 한 장이 CLIP ViT-L/14@224에서
패치 토큰 256개를 만든다. 손목 카메라를 더하거나 과거 프레임을 쌓으면
L은 금방 1000을 넘고, 그때 어텐션은 L²로 벌을 받는다.""")

# ─────────────────────────────────────────────────────────────────────────────
# (2) 생성 단계: 자기회귀로 한 토큰씩 뽑을 때
# ─────────────────────────────────────────────────────────────────────────────
print("""
(2) 자기회귀 생성 — 토큰을 하나 더 뽑는 비용
────────────────────────────────────────────────────────────────
로봇 계획("다음 5스텝?")은 문장을 생성한다. 트랜스포머는 KV 캐시를 써도
새 토큰이 이전 t개 토큰 모두와 내적을 해야 하므로 t에 비례해 느려진다.
SSM은 고정 크기 상태 하나만 갱신하므로 언제나 같은 비용이다.""")

d_inner, N = Wm["d_inner"], Wm["d_state"]
rows = []
for ctx in (256, 512, 1024, 2048):
    # 트랜스포머: 새 토큰 q(1,D)와 캐시 K,V(ctx,D)
    Kc, Vc = rng.normal(0, 1, (ctx, D)), rng.normal(0, 1, (ctx, D))
    q = rng.normal(0, 1, (1, D))

    def step_attn():
        s = q @ Kc.T / math.sqrt(D)
        return (T.softmax(s) @ Vc) @ Wo

    # SSM: 고정 크기 상태 (d_inner, N) 하나 갱신
    h = rng.normal(0, 1, (d_inner, N))
    dA = np.exp(rng.normal(0, 0.1, (d_inner, N)))
    dBu = rng.normal(0, 1, (d_inner, N))
    Cc = rng.normal(0, 1, N)

    def step_ssm():
        hh = dA * h + dBu
        return (hh * Cc[None, :]).sum(-1)

    t_a = T.timeit(step_attn, repeat=5)
    t_s = T.timeit(step_ssm, repeat=5)
    rows.append([ctx, f"{ctx * D * 2:,}", f"{t_a * 1000:7.1f}",
                 f"{d_inner * N * 2:,}", f"{t_s * 1000:7.1f}"])
print(T.table(["문맥 길이", "어텐션이 읽는 숫자", "어텐션 µs",
               "SSM이 읽는 숫자", "SSM µs"], rows))
print("""
'읽는 숫자' 열이 핵심이다. 어텐션은 문맥이 길어질수록 매 스텝 더 많은
메모리를 훑고(그래서 실제 GPU에서도 메모리 대역폭에 묶인다), SSM은 항상
d_inner×N = 5120×16 = 81,920개만 본다. 논문 그림 1의 9.0 Hz vs 1.1 Hz는
이 성질 + 7B→2.8B 크기 축소가 함께 만든 결과다.""")

# ─────────────────────────────────────────────────────────────────────────────
# (3) 논문의 속도 수치를 제어 주기로 옮겨 읽기
# ─────────────────────────────────────────────────────────────────────────────
print("""
(3) 논문 그림 1의 Hz를 '한 번 판단에 걸리는 시간'으로
────────────────────────────────────────────────────────────────""")
rows = [
    ["ManipLLM (7B, 트랜스포머)", "1.1", f"{1000 / 1.1:6.0f}", "41.3M (0.5%)", "×1.0"],
    ["OpenVLA (7B, 트랜스포머)", "3.4", f"{1000 / 3.4:6.0f}", "7.0B (100%)", "×3.1"],
    ["RoboMamba (2.8B, Mamba)", "9.0", f"{1000 / 9.0:6.0f}", "3.7M (0.1%)", "×8.2"],
]
print(T.table(["모델", "추론 Hz", "1회 ms", "파인튜닝 파라미터", "ManipLLM 대비"],
              rows, highlight={2}))
print("""
· 값은 논문 그림 1 기준 (NVIDIA A100, 양자화·추론 가속 없음).
· 초록의 "3 times faster"는 OpenVLA(3.4 → 9.0, ×2.6)를 가리키고,
  4.2절의 "7 times faster"는 ManipLLM/LLaMA-AdapterV2(1.1 → 9.0, ×8.2)를
  가리킨다. 같은 논문에서 기준이 다른 두 배수를 쓰므로, 인용할 때
  '무엇 대비 몇 배'를 반드시 붙여야 한다.
· 1.1 Hz는 한 번 움직이는 판단에 900 ms다. 사람이 컵을 옮기는 동작
  하나가 1~2초인데 그 사이 한 번밖에 못 본다는 뜻이다.
· 단, RoboMamba의 시뮬 실험은 개루프(open-loop) 단발 포즈 예측이라
  9.0 Hz가 폐루프 제어 주기로 쓰인 적은 논문에 없다. FiS-VLA의 21.9 Hz와
  직접 비교하면 안 된다(그쪽은 행동 청크·폐루프 기준).""")

# ─────────────────────────────────────────────────────────────────────────────
# 그림 — 로그-로그 스케일링
# ─────────────────────────────────────────────────────────────────────────────
X0, Y0, X1, Y1 = 80, 50, 660, 300
Ls = sorted(meas)
body = T.axes(X0, Y0, X1, Y1, "토큰 수 L (log)", "시간 (log)")
lx = [math.log2(L) for L in Ls]
lxmin, lxmax = min(lx), max(lx)
vals = [v for pair in meas.values() for v in pair]
lymin, lymax = math.log10(min(vals)), math.log10(max(vals))


def pt(L, v):
    x = X0 + (math.log2(L) - lxmin) / (lxmax - lxmin) * (X1 - X0)
    y = Y1 - (math.log10(v) - lymin) / (lymax - lymin) * (Y1 - Y0)
    return x, y


body += T.polyline([pt(L, meas[L][0]) for L in Ls], "#a8443f", 2.5)
body += T.polyline([pt(L, meas[L][1]) for L in Ls], "#2f6f4f", 2.5)
# 기준선: 완전한 L² 와 완전한 L
body += T.polyline([pt(L, meas[Ls[0]][0] * (L / Ls[0]) ** 2) for L in Ls],
                   "#c9a9a5", 1.5, dash="4 3")
body += T.polyline([pt(L, meas[Ls[0]][1] * (L / Ls[0])) for L in Ls],
                   "#a5c9b5", 1.5, dash="4 3")
for L in Ls:
    x, _ = pt(L, meas[L][0])
    body += (f'<text x="{x:.0f}" y="{Y1 + 16}" text-anchor="middle" font-size="10" '
             f'fill="#55554f">{L}</text>')
body += (f'<text x="{X1 - 150}" y="{Y0 + 14}" font-size="11" fill="#a8443f">'
         f'■ 어텐션 (점선 = 이상적 L²)</text>'
         f'<text x="{X1 - 150}" y="{Y0 + 30}" font-size="11" fill="#2f6f4f">'
         f'■ SSM (점선 = 이상적 L)</text>')
p = T.save_svg(os.path.join(OUT, "lab2_scaling.svg"), 760, 340, body,
               "어텐션 O(L²) vs 선택적 스캔 O(L) — 측정값과 이론선")
print(f"\n그림 저장: {p}")
