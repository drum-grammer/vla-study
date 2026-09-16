"""실습 1 — 토큰 열, 어텐션, 블록 쌓기 (가이드 1.2 · 1.3 · 2.4)

무엇을 느끼는가
  (1) 종류가 다른 정보(언어·이미지 패치·로봇 상태·노이즈 행동)가 전부 "같은 차원의 벡터 행"이 되어
      한 행렬에 나란히 놓인다 — 논문 그림 2의 범례가 그리는 것.
  (2) 어텐션은 "질문과 비슷한 제목표를 가진 책의 내용을 더 많이 가져오는" 가중평균이다.
  (3) 트랜스포머 블록은 입력·출력 모양이 같아서 어디서든 중간 표현을 꺼낼 수 있다.
      블록 0~3을 System 2처럼 돌리고, 그 출력 뒤에 고주파 토큰을 붙여 블록 4~5를 System 1처럼 돌린다.
      → 저장소 models/vlms/prismatic.py forward()의 llm_layer_end=self.llm_middle_layer / llm_layer_start=self.llm_middle_layer 두 번 호출과 같은 구조.

실행: python3 lab1_tokens_blocks.py
"""
import numpy as np
from tinynn import TransformerBlock, run_blocks, attention

rng = np.random.default_rng(0)
d = 16  # 실제 LLaMA2-7B는 4096

# ── (1) 모든 입력을 같은 차원의 벡터로 ────────────────────────────────────
lang_words = ["place", "the", "wine", "on", "rack"]
lang = rng.normal(0, 1, (len(lang_words), d))

# 이미지 3×3 패치 9개. 4번 패치가 "병" 패치라고 가정하고 "wine" 임베딩과 비슷하게 만든다
img = rng.normal(0, 1, (9, d))
img[4] = lang[2] * 0.9 + rng.normal(0, 0.3, d)

state = rng.normal(0, 1, (1, d))          # 로봇 상태 토큰 (상태 인코더 MLP 출력에 해당)
noised_action = rng.normal(0, 1, (1, d))  # 노이즈 섞인 행동 토큰 (행동 인코더 MLP 출력에 해당)

print("== (1) 토큰 열 ==")
for name, arr in [("언어", lang), ("이미지 패치", img), ("로봇 상태", state), ("노이즈 행동", noised_action)]:
    print(f"  {name:8s} 토큰: {arr.shape}  ← (토큰 수, 차원)")
system2_input = np.vstack([lang, img])
print(f"  System 2 입력 = 언어 + 이미지 → {system2_input.shape}  (한 행렬에 나란히)")

# ── (2) 어텐션 = 비슷한 것끼리 가중평균 ───────────────────────────────────
print("\n== (2) 어텐션: 'wine' 토큰은 어느 이미지 패치를 보는가 ==")
I = np.eye(d)
_, A = attention(system2_input, I, I, I, causal=False)   # Q=K=V=x 그대로 → 유사도가 곧 가중치
wine_row = A[2]                                          # 'wine' 토큰(2번)이 다른 토큰에 준 가중치
patch_w = wine_row[len(lang):]
for i, w in enumerate(patch_w):
    bar = "#" * int(w * 60)
    tag = "  ← 병 패치" if i == 4 else ""
    print(f"  패치 {i}: {w:.3f} {bar}{tag}")
print(f"  가장 많이 본 패치 = {int(np.argmax(patch_w))} (기대: 4)")

# ── (3) 블록 쌓기와 중간 표현 꺼내기 ─────────────────────────────────────
print("\n== (3) 블록 6개를 System 2(0~3) / System 1(4~5)로 나눠 쓰기 ==")
blocks = [TransformerBlock(d, rng) for _ in range(6)]
MIDDLE = 4  # 저장소 train.sh의 LLM_MIDDLE_LAYER=30 에 해당 (32블록 중 30)

h_mid = run_blocks(blocks, system2_input, start=0, end=MIDDLE)
print(f"  블록 0~{MIDDLE-1} 통과 (System 2)   : {system2_input.shape} → {h_mid.shape}   ※ 모양이 같다")
print("   → 이 h_mid가 논문의 '중간 잠재 특징'. 저장소: output.hidden_states[-1] (prismatic.py)")

fast_tokens = np.vstack([img, state, noised_action])   # 고주파 입력: 이미지·상태·노이즈 행동
system1_input = np.vstack([h_mid, fast_tokens])
h_out = run_blocks(blocks, system1_input, start=MIDDLE, end=6)
print(f"  고주파 토큰 {fast_tokens.shape[0]}개를 뒤에 붙임         : {h_mid.shape} + {fast_tokens.shape} → {system1_input.shape}")
print(f"  블록 {MIDDLE}~5 통과 (System 1)     : {system1_input.shape} → {h_out.shape}")
action_hidden = h_out[-1]   # 마지막 행 = 노이즈 행동 토큰 자리 → final_layer를 거쳐 노이즈 예측이 된다
print(f"  노이즈 행동 토큰 자리의 출력 벡터 : {action_hidden.shape}  → FinalLayer(MLP) → 예측 노이즈 (실습 3)")

# 같은 h_mid를 여러 번 재사용 = 1:n 비동기
print("\n== (4) 같은 h_mid를 4번 재사용 (1:4 비동기) ==")
for step in range(4):
    fresh_state = rng.normal(0, 1, (1, d))
    x = np.vstack([h_mid, img, fresh_state, noised_action])
    out = run_blocks(blocks, x, start=MIDDLE, end=6)
    print(f"  스텝 {step}: System 2는 안 돌리고 h_mid 재사용, 새 상태 토큰만 바꿔 System 1 실행 → {out.shape}")
print("  → 저장소 scripts/sim.py: if slow_cnt % slow_fast_ratio == 0 일 때만 slow_system_forward()")
