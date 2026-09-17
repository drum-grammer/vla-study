"""실습 6 · 논문의 '0.1%'를 직접 세어 보기 — 가이드 2.5 · 2.8

논문의 핵심 효율 주장은 숫자 하나다.
  "The fine-tuned policy head constitutes only 0.1% of the model parameters,
   which is 10 times smaller than existing robotic VLA approaches." (1절)
  부록 표 6: MLP×2 = 3.7M (0.11%) / MLP×1 = 1.8M (0.05%)
            / (SSM block+MLP)×2 = 45.2M (1.3%)

저장소 코드의 층 정의에서 파라미터를 직접 세어 이 숫자를 검증한다.
곱셈과 덧셈만 쓰므로 PyTorch도 체크포인트도 필요 없다.

실행: python3 lab6_param_audit.py     (1초 미만)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tinyssm as T

print(T.header(6, "논문의 '0.1%'를 직접 세어 보기", "2.5 · 2.8"))

# ─────────────────────────────────────────────────────────────────────────────
# mamba-2.8b 하이퍼파라미터 (state-spaces/mamba-2.8b-hf config.json, 확인 2026-09-11)
# ─────────────────────────────────────────────────────────────────────────────
D_MODEL, N_LAYER, D_STATE, D_CONV, EXPAND = 2560, 64, 16, 4, 2
DT_RANK, VOCAB = 160, 50280
D_INNER = EXPAND * D_MODEL          # 5120

print(f"""
(1) Mamba 블록 하나의 파라미터 — MyMamba.__init__()의 층을 그대로 센다
────────────────────────────────────────────────────────────────
d_model={D_MODEL}  d_inner={D_INNER}  d_state(N)={D_STATE}
d_conv={D_CONV}  dt_rank={DT_RANK}  n_layer={N_LAYER}  vocab={VOCAB}""")

blk = [
    ("in_proj", f"Linear({D_MODEL} → {2 * D_INNER}), bias=False",
     D_MODEL * 2 * D_INNER),
    ("conv1d", f"Conv1d({D_INNER}, groups={D_INNER}, k={D_CONV}) + bias",
     D_INNER * D_CONV + D_INNER),
    ("x_proj", f"Linear({D_INNER} → {DT_RANK + 2 * D_STATE}), bias=False",
     D_INNER * (DT_RANK + 2 * D_STATE)),
    ("dt_proj", f"Linear({DT_RANK} → {D_INNER}) + bias",
     DT_RANK * D_INNER + D_INNER),
    ("A_log", f"({D_INNER}, {D_STATE})", D_INNER * D_STATE),
    ("D", f"({D_INNER},)", D_INNER),
    ("out_proj", f"Linear({D_INNER} → {D_MODEL}), bias=False", D_INNER * D_MODEL),
    ("norm (RMSNorm)", f"({D_MODEL},)", D_MODEL),
]
blk_total = sum(n for _, _, n in blk)
rows = [[k, d, f"{n:,}", f"{n / blk_total * 100:5.1f}%"] for k, d, n in blk]
rows.append(["블록 합계", "", f"{blk_total:,}", "100.0%"])
print(T.table(["층", "모양", "파라미터", "블록 내 비중"], rows,
              highlight={len(rows) - 1}))
print(f"""
블록 하나 {T.fmt_params(blk_total)}. in_proj 하나가 블록의 {26214400 / blk_total * 100:.0f}%를 먹는다
(d_model → 2×d_inner = 4배 확장). 어텐션이 없으니 QKV도 없다.
실제 값 대조: transformers MambaForCausalLM(config)로 세면 블록당
41,244,160개 — 위 합계와 정확히 같다 (확인 2026-09-11).""")

# ─────────────────────────────────────────────────────────────────────────────
# (2) 전체 모델
# ─────────────────────────────────────────────────────────────────────────────
llm_total = blk_total * N_LAYER + VOCAB * D_MODEL + D_MODEL   # lm_head는 임베딩과 공유
clip = 303_179_776            # timm vit_large_patch14_clip_224.openai, num_classes=0
proj = (1024 * 4096 + 4096) + (4096 * 2560 + 2560) + (2560 * 2560 + 2560)
model_total = llm_total + clip + proj

print("""
(2) RoboMamba 전체 (CLIP224 + mamba-2.8b)
────────────────────────────────────────────────────────────────""")
print(T.table(["부품", "내용", "파라미터", "비중"],
              [["Mamba LLM", f"블록 {N_LAYER}개 + 임베딩 {VOCAB}×{D_MODEL} (lm_head 공유)",
                f"{llm_total:,}", f"{llm_total / model_total * 100:5.1f}%"],
               ["비전 인코더", "CLIP ViT-L/14@224 (동결, 학습 안 함)",
                f"{clip:,}", f"{clip / model_total * 100:5.1f}%"],
               ["Projector", "Linear(1024→4096→2560→2560)",
                f"{proj:,}", f"{proj / model_total * 100:5.1f}%"],
               ["합계", "", f"{model_total:,} ({T.fmt_params(model_total)})", "100.0%"]],
              highlight={3}))
print(f"""
논문 4절은 "RoboMamba, with only 3.2B parameters"라 쓴다. 위 계산은
{T.fmt_params(model_total)}. 표 1·2의 'LLM 2.7B'는 Mamba만, 그림 1의 '2.8B'는
체크포인트 이름(mamba-2.8b)이다. **같은 모델을 세는 세 가지 방식**이므로
인용할 때 무엇을 센 숫자인지 밝혀야 한다.""")

# ─────────────────────────────────────────────────────────────────────────────
# (3) 정책 헤드 — 논문 표 6 대조
# ─────────────────────────────────────────────────────────────────────────────
print("""
(3) 정책 헤드: 저장소 코드로 센 값 vs 논문 표 6
────────────────────────────────────────────────────────────────
저장소 manip.py SpecialMLP(inp, oup):
    fc1 Linear(inp,   inp//2)             + bias
    fc2 Linear(inp//2, inp//4)            + bias
    fc3 Linear(inp//4, oup, bias=False)""")


def special_mlp(inp, oup, div1=2, div2=4):
    h1, h2 = inp // div1, inp // div2
    return (inp * h1 + h1) + (h1 * h2 + h2) + (h2 * oup)


code_2 = special_mlp(D_MODEL, 2)
code_6 = special_mlp(D_MODEL, 6)
code_two = code_2 + code_6
code_one = special_mlp(D_MODEL, 8)
# 논문 수치와 맞는 대안 축소비: inp//4, inp//8
alt_2 = special_mlp(D_MODEL, 2, 4, 8)
alt_6 = special_mlp(D_MODEL, 6, 4, 8)
alt_two = alt_2 + alt_6
alt_one = special_mlp(D_MODEL, 8, 4, 8)

print()
print(T.table(["구성", "저장소 코드대로 센 값", "모델 대비", "논문 표 6", "논문 %"],
              [["MLP×2 (two_mlp, 본 모델)", f"{code_two:,} ({T.fmt_params(code_two)})",
                f"{code_two / model_total * 100:.3f}%", "3.7M", "0.11%"],
               ["MLP×1 (mlp, 출력 8개)", f"{code_one:,} ({T.fmt_params(code_one)})",
                f"{code_one / model_total * 100:.3f}%", "1.8M", "0.05%"],
               ["(SSM+MLP)×2", f"{2 * blk_total + code_two:,} "
                f"({T.fmt_params(2 * blk_total + code_two)})",
                f"{(2 * blk_total + code_two) / model_total * 100:.3f}%",
                "45.2M", "1.3%"]], highlight={0}))

print(f"""
→ **논문과 코드가 맞지 않는다.** two_mlp를 저장소 코드대로 세면
   {T.fmt_params(code_two)}인데 논문 표 6은 3.7M이라 적는다. {code_two / 3_700_000:.2f}배 차이.

가장 그럴듯한 설명 (가설, 미확인)
  SpecialMLP의 축소비를 //2, //4가 아니라 **//4, //8**로 두면
    출력 2개 → {alt_2:,} / 출력 6개 → {alt_6:,} / 합계 {alt_two:,} = {alt_two / 1e6:.2f}M ≈ 3.7M ✓
    출력 8개 하나만 → {alt_one:,} = {alt_one / 1e6:.2f}M ≈ 1.8M ✓
  즉 논문 표 6은 축소비가 한 단계 더 큰(=더 작은) 헤드로 잰 값이고,
  공개된 코드는 그보다 넓은 헤드다. 논문 제출 후 코드를 바꿨을 가능성.
  (SSM+MLP)×2의 45.2M도 'Mamba 블록 1개({T.fmt_params(blk_total)}) + 위 3.7M'
  = {(blk_total + alt_two) / 1e6:.1f}M에 가깝지만, 코드는 ssm1·ssm2 **두 개**를
  만든다(manip.py 56~57행) → {T.fmt_params(2 * blk_total + code_two)}. 여기도 상충.

이 불일치가 논문의 결론을 흔드는가? — 아니다. 오히려 강화한다.
  · {T.fmt_params(code_two)}든 3.7M이든 전체의 0.1~0.3%다. "10배 작다"는
    ManipLLM 41.3M(0.5%) 대비 주장은 두 값 모두에서 성립한다.
  · 표 6의 결론 자체가 "헤드 크기는 결과에 거의 영향이 없다"
    (63.7% / 62.1% / 63.2%)이므로 크기가 두 배여도 논지가 유지된다.
  · 다만 **논문 숫자를 그대로 인용해 재현하려 하면 맞지 않는다.**
    대외 문서에 쓸 때는 "논문 표 6 기준 3.7M(공개 코드로는 {T.fmt_params(code_two)},
    확인 2026-09-11)"처럼 두 값을 같이 적는 편이 안전하다.""")

# ─────────────────────────────────────────────────────────────────────────────
# (4) 베이스라인과 비교 — 논문 4.3절
# ─────────────────────────────────────────────────────────────────────────────
print("""
(4) "10배 작다"는 무엇과 비교한 것인가 (논문 4.3절)
────────────────────────────────────────────────────────────────""")
print(T.table(["모델", "파인튜닝하는 것", "파라미터", "모델 대비"],
              [["RoboFlamingo", "모델 일부 전체", "1.8B", "35.5%"],
               ["ManipLLM", "어댑터", "41.3M", "0.5%"],
               ["OpenVLA", "모델 전체 (그림 1)", "7.0B", "100%"],
               ["RoboMamba (논문)", "MLP 헤드 2개", "3.7M", "0.11%"],
               ["RoboMamba (공개 코드)", "MLP 헤드 2개",
                T.fmt_params(code_two), f"{code_two / model_total * 100:.2f}%"]],
              highlight={3, 4}))
print(f"""
비교의 성격을 정확히 보기
  · RoboFlamingo·ManipLLM은 백본까지 손대므로 원래 능력이 훼손될 수 있다
    (논문 3.4절: "breaks the inherent abilities of the pre-trained model").
    RoboMamba는 백본을 아예 동결하니 언어 추론 능력이 그대로 남는다.
    이것이 FiS-VLA가 L_slow(공동학습)로 해결하려 한 문제와 같은 문제인데,
    RoboMamba는 '건드리지 않음'으로, FiS-VLA는 '같이 학습함'으로 풀었다.
  · 단, 동결의 대가도 있다. 조작 학습이 백본의 표현을 개선할 수 없으므로
    백본이 못 담은 정보는 헤드가 만들어 낼 수 없다(실습 5 (3)).
  · 저장 용량으로는 헤드가 fp32에서 {code_two * 4 / 1e6:.1f} MB
    (논문은 "only a 7MB policy head" — 3.7M × 4바이트 ≈ 14.8MB가 아니라
     7MB라면 3.7M × 2바이트(fp16) 또는 1.8M × 4바이트에 해당. 미확인).

해 볼 것
  · D_MODEL을 1.4b 설정(d_model=2048, n_layer=48)으로 바꿔 전체를 다시 세 보라.
    논문 그림 3 a)의 'Ours-1.4B'가 몇 개짜리 모델인가?
  · in_proj가 블록의 64%를 먹는다. expand를 2에서 1로 줄이면 전체가 얼마나
    줄고, 실습 1의 상태 크기는 어떻게 되는가?""")
