"""실습 3 — 확산으로 행동 만들기, 논문 식 (1) L_fast 직접 학습 (가이드 1.8 · 2.6)

무엇을 느끼는가
  (1) 시연 데이터는 보통 "여러 갈래"다. 장애물을 왼쪽으로 돌아도 되고 오른쪽으로 돌아도 된다.
      평균을 맞히는 회귀(MSE로 행동을 직접 예측)는 두 갈래의 가운데 = 장애물 위를 내놓는다.
  (2) 확산 모델은 "노이즈를 맞히는" 법을 배우고, 샘플링하면 두 갈래를 모두 만든다.
  (3) 학습 코드는 저장소 models/vlas/fisvla.py forward()의 네 줄과 같다:
        noise = randn_like(actions); timestep = randint(0, T); x = q_sample(actions, timestep, noise)
        loss = ((noise_pred - noise) ** 2).mean()
      노이즈 스케줄도 저장소와 같은 squaredcos_cap_v2, T=100.

실행: python3 lab3_diffusion_action.py   (약 20~40초, out/lab3_diffusion.svg 생성)
"""
import os
import numpy as np
from tinynn import MLP, Adam, make_schedule, q_sample, SVG

rng = np.random.default_rng(3)
os.makedirs("out", exist_ok=True)
T = 100
betas, alphas, alpha_bar = make_schedule(T)


# ── 시연 데이터: 조건 c(0 또는 1)에 따라 위/아래, 좌/우 두 갈래 ─────────────
def demo_actions(n):
    c = rng.integers(0, 2, n)                          # 조건: 0 = 물체가 위쪽, 1 = 아래쪽
    side = rng.choice([-1.0, 1.0], n)                  # 왼쪽 또는 오른쪽으로 돌아간다 (두 갈래)
    y = np.where(c == 0, 0.55, -0.55)
    a = np.stack([side * 0.65, y], 1) + rng.normal(0, 0.05, (n, 2))
    return a, c.astype(float)


def t_embed(t):
    """타임스텝 τ를 sin/cos로 (저장소 TimestepEmbedder의 축소판)"""
    f = t / T * 2 * np.pi
    return np.stack([np.sin(f), np.cos(f), np.sin(2 * f), np.cos(2 * f)], 1)


# ── (A) 회귀 베이스라인: 행동을 직접 MSE로 예측 ───────────────────────────
reg = MLP([1, 64, 64, 2], rng)
opt = Adam(reg.params(), lr=2e-3)
for step in range(1500):
    a, c = demo_actions(128)
    pred = reg.forward(c[:, None])
    grads, _ = reg.backward(2 * (pred - a) / len(a))
    opt.step(grads)

# ── (B) 확산: 노이즈 예측기 π_θf(x_τ, c, τ) 학습 = 논문 식 (1) ─────────────
eps_net = MLP([2 + 4 + 1, 128, 128, 2], rng)
opt = Adam(eps_net.params(), lr=2e-3)
print("== L_fast 학습 (노이즈 예측 MSE) ==")
for step in range(4000):
    a, c = demo_actions(128)
    noise = rng.normal(0, 1, a.shape)                    # noise = torch.randn_like(actions)
    t = rng.integers(0, T, len(a))                       # timestep = torch.randint(0, num_timesteps)
    x_t = q_sample(a, t, noise, alpha_bar)               # x = diffusion.q_sample(actions, timestep, noise)
    inp = np.concatenate([x_t, t_embed(t), c[:, None]], 1)
    noise_pred = eps_net.forward(inp)
    loss = ((noise_pred - noise) ** 2).mean()            # loss = ((noise_pred - noise) ** 2).mean()
    grads, _ = eps_net.backward(2 * (noise_pred - noise) / noise.size)
    opt.step(grads)
    if step % 500 == 0 or step == 3999:
        print(f"  step {step:4d}  L_fast = {loss:.4f}")


# ── 샘플링: 순수 노이즈에서 시작해 예측 노이즈를 빼 가며 행동으로 (DDPM) ─────
def sample(c_val, n):
    x = rng.normal(0, 1, (n, 2))
    c = np.full((n, 1), float(c_val))
    for t in reversed(range(T)):
        tt = np.full(n, t)
        eps = eps_net.forward(np.concatenate([x, t_embed(tt), c], 1))
        ab, a_t, b_t = alpha_bar[t], alphas[t], betas[t]
        mean = (x - b_t / np.sqrt(1 - ab) * eps) / np.sqrt(a_t)
        if t > 0:
            ab_prev = alpha_bar[t - 1]
            var = b_t * (1 - ab_prev) / (1 - ab)          # sigma_small=True 에 해당
            x = mean + np.sqrt(var) * rng.normal(0, 1, x.shape)
        else:
            x = mean
    return x


print("\n== 결과 (조건 c=0, 물체가 위쪽) ==")
data, cdat = demo_actions(400)
reg_pred = reg.forward(np.array([[0.0]]))[0]
samples = sample(0, 300)
left = (samples[:, 0] < 0).mean()
print(f"  회귀 예측 행동      : ({reg_pred[0]:+.2f}, {reg_pred[1]:+.2f})  ← 두 갈래의 가운데(x≈0) = 장애물 위")
print(f"  확산 샘플 300개     : 왼쪽 {left*100:.0f}% / 오른쪽 {(1-left)*100:.0f}%,  평균 y = {samples[:,1].mean():+.2f} (정답 +0.55)")
print(f"  확산 샘플 x 절댓값 평균 = {np.abs(samples[:,0]).mean():.2f} (정답 0.65) → 갈래를 지켰다")

svg = SVG(title="시연(회색) · 회귀 예측(빨강 ×) · 확산 샘플(주황)  — 조건 c=0")
svg.points([tuple(p) for p in data[cdat == 0]], "#9aa1ae", 2.2, 0.6)
svg.points([tuple(p) for p in samples], "#D9531E", 2.5, 0.8)
svg.label(reg_pred[0] - 0.03, reg_pred[1] + 0.02, "×", "#C23A2B", 26)
svg.label(-0.15, -0.05, "장애물", "#444")
svg.line([(-0.12, 0), (0.12, 0)], "#444", 6)
svg.legend([("#9aa1ae", "시연 데이터 (두 갈래)"), ("#D9531E", "확산 샘플 (두 갈래 유지)"), ("#C23A2B", "× 회귀 예측 (가운데로 붕괴)")])
path = svg.save("out/lab3_diffusion.svg")
print(f"  그림 저장: {path}")
print("\n※ 저장소 대응: models/vlas/fisvla.py forward() (noise/q_sample/MSE), models/diffusion/__init__.py create_diffusion(noise_schedule='squaredcos_cap_v2', diffusion_steps=100)")
