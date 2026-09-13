"""실습 4 — 행동 청크와 1:n 비동기, 117.7 Hz의 뜻 (가이드 1.8 · 2.5 · 2.7)

무엇을 느끼는가
  (1) GPU 한 장에서 System 2(무겁다)와 System 1(가볍다)을 순차 실행하면, System 2를 n스텝에 한 번만
      돌릴수록 제어 주파수가 오른다. 대신 System 1은 "낡은 이해"로 움직인다 — 이미 끝난 하위 목표를
      System 2가 다시 볼 때까지 계속 붙들고 있다.
      → 너무 잦으면(1:1) 느리고, 너무 드물면(1:16) 헛도는 시간이 길다. 사이 어딘가가 최선 (논문 표 8: 1:4).
  (2) 행동 청크 H: 한 번 추론에 H스텝을 내면 명령 주파수는 H배가 되지만, 관측을 보는 횟수는 그대로다.
      → 논문 117.7 Hz(청크 8)와 21.9 Hz(청크 1)의 관계 (표 9).
  (3) 저장소 scripts/sim.py 의 루프가 정확히 이 구조다:
        if slow_cnt % slow_fast_ratio == 0: slow_system_forward()   # System 2, 가끔
        fast_system_forward(slow_latent_embedding, 현재 이미지·상태)   # System 1, 매 스텝
        for action in actions: env.step(action)                       # 청크 실행

모형: 2차원 평면의 점 로봇이 웨이포인트 수십 개를 순서대로 방문한다 ("그릇 → 물체 → 그릇 → 접시 …" 같은
      다단계 태스크. 논문 파인튜닝 데이터의 하위 태스크 언어 계획 lang_subgoals에 해당).
  - System 2 (비용 c_slow=90 ms): 지금 어느 단계인지 보고 다음 웨이포인트를 목표로 낸다. n스텝마다 한 번.
  - System 1 (비용 c_fast=30 ms): 자기 위치(항상 최신)와 마지막 목표로 한 스텝 이동량(최대 Δ)을 낸다.
  - 단계가 끝나도 System 1은 다음 System 2 갱신 전까지 옛 목표 자리에서 헛돈다.
  - 성공: 시간 예산 안에 모두 방문. 에피소드마다 개수(20~44)가 달라 성공률이 0/100이 아닌 값으로 나온다.
  ※ 청크 1·비율 1:4에서 관측 주파수가 논문 표 1의 21.9 Hz 근처가 되도록 비용을 잡았다.

실행: python3 lab4_chunk_async.py   (약 10초, out/lab4_ratio.svg 생성)
"""
import os
import numpy as np
from tinynn import SVG

os.makedirs("out", exist_ok=True)

C_SLOW, C_FAST = 0.090, 0.030     # 초. 7B VLM 32블록 전체 vs 마지막 2블록 (+ 인코더·확산 스텝)
STEP = 0.03                       # System 1이 한 스텝에 낼 수 있는 최대 이동량 (m)
EPS = 0.02                        # 웨이포인트 도달 반경 (m)
N_WAYPOINTS = (20, 44)            # 에피소드마다 하위 목표(키프레임) 개수를 이 범위에서 뽑는다 (평균 32)
BUDGET = 9.0                      # 에피소드 시간 예산 (초)
NOISE = 0.003                     # 매 명령 외란 (m)


def episode(ratio_n, chunk_h, seed):
    r = np.random.default_rng(seed)
    pos = np.zeros(2)
    # 웨이포인트: 이전 지점에서 0.06~0.13 m 떨어진 무작위 방향 (2~4 스텝 거리)
    wps = []
    p = pos.copy()
    n_wp = int(r.integers(N_WAYPOINTS[0], N_WAYPOINTS[1] + 1))
    for _ in range(n_wp):
        ang = r.uniform(0, 2 * np.pi); dist = r.uniform(0.06, 0.13)
        p = p + dist * np.array([np.cos(ang), np.sin(ang)]); wps.append(p.copy())
    phase = 0                     # 환경이 아는 "지금 단계"
    goal = None                   # System 1이 아는 목표 (System 2가 마지막으로 알려 준 것)
    t, step_i = 0.0, 0
    while t < BUDGET:
        if step_i % ratio_n == 0:                 # System 2: 지금 단계를 보고 목표 갱신 (비용 c_slow)
            goal = wps[phase]; t += C_SLOW
        t += C_FAST                               # System 1: 한 번 추론 (비용 c_fast), 청크 H개 계획
        plan_p = pos.copy(); deltas = []
        for _ in range(chunk_h):
            d = goal - plan_p
            mv = d if np.linalg.norm(d) < STEP else d / np.linalg.norm(d) * STEP
            deltas.append(mv); plan_p = plan_p + mv           # 청크 안은 관측 없는 개루프
        for mv in deltas:
            pos = pos + mv + r.normal(0, NOISE, 2)
            if np.linalg.norm(pos - wps[phase]) < EPS:        # 환경: 단계 완료
                phase += 1
                if phase == n_wp:
                    return True, t
                # System 1은 아직 모른다 → 다음 System 2 갱신까지 옛 목표(=현 위치)에서 헛돈다
        step_i += 1
    return False, t


def evaluate(ratio_n, chunk_h, n_ep=300):
    ok = [episode(ratio_n, chunk_h, s)[0] for s in range(n_ep)]
    per_cmd = (C_FAST + C_SLOW / ratio_n) / chunk_h
    obs_hz = 1 / (C_FAST + C_SLOW / ratio_n)          # 새 관측을 보는 빈도 (System 1 호출)
    slow_hz = obs_hz / ratio_n                        # System 2 갱신 빈도
    return np.mean(ok), 1 / per_cmd, obs_hz, slow_hz


print("== (1) 주파수 비율 1:n  (청크 1 고정)  — 논문 표 8과 비교 ==")
print(f"  {'비율':>6} {'성공률':>7} {'명령 Hz':>8} {'관측 Hz':>8} {'S2 Hz':>7}   단계 전환 시 최대 헛도는 시간")
res_ratio = []
for n in [1, 2, 4, 8, 16]:
    sr, cmd_hz, obs_hz, slow_hz = evaluate(n, 1)
    res_ratio.append((n, sr))
    stale = n * (C_FAST + C_SLOW / n)
    print(f"  1:{n:<4d} {sr*100:6.1f}% {cmd_hz:8.1f} {obs_hz:8.1f} {slow_hz:7.1f}   {stale*1000:5.0f} ms")
best = max(res_ratio, key=lambda x: x[1])
print(f"  → 최선 비율 1:{best[0]}  (논문 표 8: 1:1 0.60 · 1:2 0.63 · 1:4 0.69 · 1:8 0.61)")
print("  → 1:1은 System 2를 매번 돌려 느리고, 1:16은 빠르지만 단계가 바뀐 뒤 옛 목표를 오래 붙든다")

print("\n== (2) 행동 청크 H  (비율 1:4 고정) — 논문 표 9와 비교 ==")
print(f"  {'청크':>4} {'성공률':>7} {'명령 Hz':>8} {'관측 Hz':>8}")
for h in [1, 2, 4, 8]:
    sr, cmd_hz, obs_hz, _ = evaluate(4, h)
    print(f"  {h:4d} {sr*100:6.1f}% {cmd_hz:8.1f} {obs_hz:8.1f}")
print("  → 명령 Hz는 H에 비례해 오르지만 관측 Hz는 그대로. 논문의 117.7 Hz(청크 8)는 '한 번 추론에 8개'라는 뜻이고")
print("    새 관측을 보는 빈도는 21.9 Hz급 그대로다. 청크가 길면 외란을 못 보고 개루프로 가는 구간이 길어진다")

svg = SVG(xlim=(0, 5), ylim=(0, 1.05), title="주파수 비율 1:n 에 따른 성공률 (장난감 모형)")
pts = [(i + 0.5, sr) for i, (_, sr) in enumerate(res_ratio)]
svg.line(pts, "#2F3E9E", 2)
svg.points(pts, "#D9531E", 5)
for (n, sr), (x, y) in zip(res_ratio, pts):
    svg.label(x - 0.15, y + 0.06, f"1:{n}", "#222")
    svg.label(x - 0.2, y - 0.05, f"{sr*100:.0f}%", "#D9531E")
path = svg.save("out/lab4_ratio.svg")
print(f"\n  그림 저장: {path}")
print("※ 이 모형은 논문의 로봇·모델이 아니라 '순차 실행 비용'과 '이해가 낡는 비용' 두 힘만 뽑아낸 장난감이다.")
print("  C_SLOW/C_FAST, N_WAYPOINTS, BUDGET을 바꿔 최선 비율이 어디로 움직이는지 확인하라 (논문 5절 한계: 비율이 고정이다).")
