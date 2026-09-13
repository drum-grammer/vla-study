"""실습 4 · 이미지를 Mamba에 넣는 방법과 '순서'의 대가 — 가이드 2.3 · 2.4

논문 3.2절: CLIP ViT-L로 f_v ∈ R^(B×N×1024)를 뽑고, MLP 연결기로
f_v^L ∈ R^(B×N×2560)으로 옮긴 뒤 텍스트 토큰과 이어 붙여 Mamba에 넣는다.

저장소 vlm.py LinearVLM.encode() 60행이 그 이어 붙이는 순서를 정한다.
    text_result[multi_modal] = torch.cat([vision_encoded_result, text_encoded], dim=1)
즉 **[이미지 토큰 … , 텍스트 토큰 …]** 순서다. 트랜스포머에서는 별 얘기가
아니지만 Mamba에서는 결과가 달라진다. 그 차이를 직접 확인한다.

실행: python3 lab4_vlm_token_order.py     (약 3초, out/lab4_order.svg 생성)
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

print(T.header(4, "이미지를 Mamba에 넣는 방법과 '순서'의 대가", "2.3 · 2.4"))

# ─────────────────────────────────────────────────────────────────────────────
# (1) 실제 차원으로 토큰 열 만들어 보기
# ─────────────────────────────────────────────────────────────────────────────
print("""
(1) 이미지 한 장이 토큰 몇 개가 되는가 (실제 차원)
────────────────────────────────────────────────────────────────""")
rows = []
for tag, res, patch in (("CLIP224 (논문 주 설정)", 224, 14), ("CLIP336", 336, 14),
                        ("SIGLIP256", 256, 16), ("SIGLIP384", 384, 16)):
    n_tok = (res // patch) ** 2
    rows.append([tag, f"{res}×{res}", patch, n_tok, 1024, f"{n_tok * 1024:,}"])
print(T.table(["vision.py의 encoder_type", "입력 해상도", "패치", "패치 토큰 수",
               "토큰 차원", "숫자 개수"], rows, highlight={0}))
print("""
저장소 vision.py의 vision_encoders 딕셔너리에 이 네 개가 있고,
test.sh는 --vision_encoder CLIP224를 쓴다. 논문 표 1의 'Res. 224' 행이 이것.

한 가지 더: vision.py 52행이 마지막 층이 아니라 **끝에서 두 번째 층**을 꺼낸다.
    x = self.model.get_intermediate_layers(x, n={len(self.model.blocks) - 2})[0]
마지막 층은 CLIP의 대조학습 목적(이미지 전체를 한 벡터로)에 맞춰 특화돼
국소 정보가 뭉개진다는 것이 LLaVA 계열의 경험이고, RoboMamba도 같은 선택을
했다. 논문 본문에는 없는, 코드에만 있는 사실이다(확인 2026-09-11).""")

# ─────────────────────────────────────────────────────────────────────────────
# (2) Projector — 논문의 "simple cross-modal connector"
# ─────────────────────────────────────────────────────────────────────────────
print("""
(2) 연결기(Projector)의 실제 모양과 크기
────────────────────────────────────────────────────────────────
저장소 vlm.py Projector: Linear(1024→4096) → GELU → Linear(4096→2560)
                          → GELU → Linear(2560→2560)""")
d_v, d_l = 1024, 2560
p1 = d_v * (d_v * 4) + d_v * 4
p2 = (d_v * 4) * d_l + d_l
p3 = d_l * d_l + d_l
print(T.table(["층", "모양", "파라미터"],
              [["Linear 1", f"{d_v} → {d_v * 4}", f"{p1:,}"],
               ["Linear 2", f"{d_v * 4} → {d_l}", f"{p2:,}"],
               ["Linear 3", f"{d_l} → {d_l}", f"{p3:,}"],
               ["합계", "", f"{p1 + p2 + p3:,} ({T.fmt_params(p1 + p2 + p3)})"]],
              highlight={3}))
print(f"""
논문은 이것을 그냥 "multilayer perceptron (MLP)"이라고만 쓴다. 실제로는
3층에 중간을 4배로 부풀린 {T.fmt_params(p1 + p2 + p3)} 모듈이고, Stage 1.1
정렬 사전학습에서 **이것만** 학습한다(비전 인코더·Mamba는 동결).
LLaVA의 연결기는 2층이라, 여기서 한 층 더 쓴 것은 저장소에서만 보이는 선택이다.""")

# ─────────────────────────────────────────────────────────────────────────────
# (3) 순서가 왜 문제인가 — 인과 순환의 결과
# ─────────────────────────────────────────────────────────────────────────────
print("""
(3) [이미지, 텍스트] 순서 vs [텍스트, 이미지] 순서
────────────────────────────────────────────────────────────────
Mamba는 인과(causal) 순환이다. t번째 토큰의 출력은 1..t번 토큰만 본다.
트랜스포머 디코더도 인과지만, 어텐션은 과거 토큰을 **직접 다시 읽는다**.
Mamba는 과거를 고정 크기 상태로 **압축해 들고 갈 뿐**이다. 그래서 순서가
'무엇이 무엇을 볼 수 있나'뿐 아니라 '얼마나 선명하게 보나'까지 바꾼다.""")

d_model = 48
W = T.mamba_params(d_model, seed=3)
n_img, n_txt = 16, 6
img = rng.normal(0, 1, (n_img, d_model))
txt = rng.normal(0, 1, (n_txt, d_model))

seq_it = np.concatenate([img, txt], axis=0)     # 저장소 순서
seq_ti = np.concatenate([txt, img], axis=0)     # 뒤집은 순서
y_it, _ = T.mamba_block(seq_it, W)
y_ti, _ = T.mamba_block(seq_ti, W)

# 마지막 토큰의 출력이 각 입력 토큰에 얼마나 민감한가 (수치 미분)
def sensitivity(seq, eps=1e-4):
    base, _ = T.mamba_block(seq, W)
    last = base[-1].copy()
    out = np.zeros(len(seq))
    for i in range(len(seq)):
        s = seq.copy()
        s[i] += eps
        y, _ = T.mamba_block(s, W)
        out[i] = np.abs(y[-1] - last).sum() / eps
    return out

s_it = sensitivity(seq_it)
s_ti = sensitivity(seq_ti)
img_it, txt_it = s_it[:n_img].mean(), s_it[n_img:].mean()
img_ti, txt_ti = s_ti[n_txt:].mean(), s_ti[:n_txt].mean()

print(T.table(["토큰 순서", "마지막 토큰", "이미지 토큰 민감도(평균)",
               "텍스트 토큰 민감도(평균)", "이미지/텍스트"],
              [[f"[이미지{n_img}, 텍스트{n_txt}] ← 저장소", "텍스트 끝",
                f"{img_it:.4f}", f"{txt_it:.4f}", f"{img_it / txt_it:.3f}"],
               [f"[텍스트{n_txt}, 이미지{n_img}]", "이미지 끝",
                f"{img_ti:.4f}", f"{txt_ti:.4f}", f"{img_ti / txt_ti:.3f}"]],
              highlight={0}))
print(f"""
읽는 법
  · 어느 순서든 **뒤에 온 토큰이 훨씬 큰 영향**을 준다. 앞쪽 토큰의 정보는
    exp(ΔA)를 여러 번 곱하며 지수적으로 옅어진다(실습 1의 기억 반경).
  · 저장소 순서([이미지, 텍스트])에서는 지시문이 상태에 가장 가깝게 남고,
    이미지 패치 {n_img}개는 그보다 멀어진다. 답을 만드는 마지막 위치에서 보면
    '지시는 선명하고, 이미지는 요약되어' 들어 있다.
  · 뒤집으면 반대가 된다. 이미지가 선명하고 지시가 옅어진다.
  · 트랜스포머라면 마지막 토큰이 어텐션으로 아무 패치나 다시 정확히 읽을 수
    있으므로 이 비대칭이 훨씬 약하다. **이것이 Mamba를 쓴 대가다.**

이 실습은 무작위 가중치의 축소 모형이라 절대 수치를 일반화하면 안 된다.
읽어야 할 것은 '순서에 따라 비대칭이 생긴다'는 방향성 하나다.

스터디에서 따져 볼 것
  · 논문 표 1에서 RoboMamba가 POPE(환각) 86.3·GQA 64.2로 좋은 반면
    MME(1297.2)·MMBench(60.9)는 LLaVA1.5(1510.7·64.3)보다 낮다.
    '이미지를 앞에 두고 압축해 통과시키는' 구조와 관계가 있을까?
  · 논문에 이미지/텍스트 순서를 바꾼 소거 실험은 **없다**(부록 C 확인).
    RoboMamba 후속인 FiS-VLA가 트랜스포머(LLaMA2)로 돌아간 이유 중 하나로
    이 비대칭을 꼽을 수 있는가? — 부스 질문 후보.""")

# ─────────────────────────────────────────────────────────────────────────────
# (4) 전체 입력 열 조립
# ─────────────────────────────────────────────────────────────────────────────
print(f"""
(4) 논문 주 설정의 실제 입력 열 (CLIP224 + mamba-2.8b)
────────────────────────────────────────────────────────────────
  이미지 224×224 → ViT-L/14 패치 16×16 = 256 토큰 × 1024차원
      ↓ Projector (1024→4096→2560→2560), {T.fmt_params(p1 + p2 + p3)}
  이미지 토큰 256 × 2560
  지시문 "Predict the contact point and orientation for pulling the {{object}}"
      ↓ Mamba 토크나이저 (GPT-NeoX 계열, vocab 50280)
  텍스트 토큰 약 15 × 2560
      ↓ torch.cat([vision, text], dim=1)      ← vlm.py 60행
  입력 열 약 271 × 2560  →  Mamba 블록 64개  →  출력 열 271 × 2560
      ↓ (추론) lm_head → 다음 토큰       ... 언어 답변 (reasoning)
      ↓ (조작)  정책 헤드                ... 실습 5로""")

# ─────────────────────────────────────────────────────────────────────────────
# 그림
# ─────────────────────────────────────────────────────────────────────────────
X0, Y0, X1 = 70, 60, 820
body = ""
for k, (lab, s, split, order) in enumerate((
        (f"[이미지 {n_img} → 텍스트 {n_txt}]  ← 저장소 순서", s_it, n_img, "it"),
        (f"[텍스트 {n_txt} → 이미지 {n_img}]", s_ti, n_txt, "ti"))):
    base = Y0 + k * 128
    w = (X1 - X0) / len(s)
    smax = max(s_it.max(), s_ti.max())
    body += (f'<text x="{X0}" y="{base - 10}" font-size="12" font-weight="600" '
             f'fill="#1a1a18">{lab}</text>')
    for i, v in enumerate(s):
        is_img = (i >= split) if order == "ti" else (i < split)
        col = "#3f6fa8" if is_img else "#b8863f"
        h = max(2.0, v / smax * 80)
        body += (f'<rect x="{X0 + i * w:.1f}" y="{base + 84 - h:.1f}" '
                 f'width="{w - 1.5:.1f}" height="{h:.1f}" fill="{col}" opacity="0.85"/>')
    body += (f'<line x1="{X0}" y1="{base + 84}" x2="{X1}" y2="{base + 84}" '
             f'stroke="#8a8a82"/>')
body += (f'<text x="{X0}" y="{Y0 + 262}" font-size="11" fill="#3f6fa8">'
         f'■ 이미지 토큰</text>'
         f'<text x="{X0 + 110}" y="{Y0 + 262}" font-size="11" fill="#b8863f">'
         f'■ 텍스트 토큰</text>'
         f'<text x="{X0 + 240}" y="{Y0 + 262}" font-size="11" fill="#55554f">'
         f'막대 높이 = 마지막 토큰 출력이 그 입력 토큰에 얼마나 민감한가</text>')
p = T.save_svg(os.path.join(OUT, "lab4_order.svg"), 880, 300, body,
               "Mamba는 '뒤에 온 토큰'을 선명하게 본다 — 순서가 설계 결정이 된다")
print(f"\n그림 저장: {p}")
