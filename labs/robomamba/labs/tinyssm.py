"""tinyssm — RoboMamba 실습용 공용 도구 (numpy만 사용)

RoboMamba 논문(arXiv 2406.04339v2)과 저장소 lmzpai/roboMamba(확인 2026-09-11)의
Mamba 블록을 numpy로 그대로 옮긴 축소판이다. 설계 원칙 세 개.

1. 저장소와 같은 식·같은 이름. selective_scan()은 modeling_mamba.py의
   selective_scan_ref()를, mamba_block()은 MyMamba.forward()를 따른다.
2. GPU·PyTorch 없이 1초 안에 돈다. 차원만 작게 줄였다.
3. 학습이 필요한 실습은 수동 역전파 + Adam으로 처리한다.

참고: 실제 하이퍼파라미터 (state-spaces/mamba-2.8b-hf config.json, 확인 2026-09-11)
  d_model=2560, n_layer=64, d_state(N)=16, d_conv=4, expand=2,
  d_inner=5120, dt_rank(time_step_rank)=160, vocab=50280
"""

import math
import time

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 활성 함수
# ─────────────────────────────────────────────────────────────────────────────


def softplus(x):
    """log(1+exp(x)). Δ를 항상 양수로 만드는 데 쓴다 (delta_softplus=True)."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def silu(x):
    """x * sigmoid(x). Mamba 블록의 활성 함수 (config의 hidden_act='silu')."""
    return x / (1.0 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0.0)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
# 선택적 스캔 (SSM의 심장)
# ─────────────────────────────────────────────────────────────────────────────


def selective_scan(u, delta, A, B, C, D=None, z=None, return_states=False):
    """저장소 modeling_mamba.py의 selective_scan_ref()를 numpy로 옮긴 것.

    논문 식 (2)(3)(4)에 해당한다.
        Ā = exp(ΔA)                    ... 식 (2)
        B̄ ≈ Δ·B                        ... 식 (3)의 1차 근사(저장소와 동일)
        h_t = Ā h_{t-1} + B̄ x_t        ... 식 (4)
        y_t = C h_t

    인자
        u     (L, Dm)  입력 시퀀스 (저장소의 conv+silu 통과 후 x)
        delta (L, Dm)  시각 간격 Δ. 토큰마다·채널마다 다르다 → 이것이 '선택성'
        A     (Dm, N)  상태 행렬. 저장소는 A = -exp(A_log)이라 항상 음수
        B     (L, N)   입력 행렬. 토큰마다 다르다 (S6의 핵심)
        C     (L, N)   출력 행렬. 토큰마다 다르다
        D     (Dm,)    잔차(skip) 항
        z     (L, Dm)  게이트. out = out * silu(z)

    반환
        y (L, Dm), 마지막 상태 h (Dm, N) [, 모든 상태 (L, Dm, N)]

    주의: 저장소는 einsum('bdl,dn->bdln')으로 배치를 함께 처리하지만
    여기서는 배치 1개만 다룬다. 수식은 완전히 같다.
    """
    L, Dm = u.shape
    N = A.shape[1]
    h = np.zeros((Dm, N), dtype=np.float64)
    y = np.zeros((L, Dm), dtype=np.float64)
    states = np.zeros((L, Dm, N), dtype=np.float64) if return_states else None

    for t in range(L):
        dA = np.exp(delta[t][:, None] * A)                     # (Dm, N)  식 (2)
        dBu = delta[t][:, None] * B[t][None, :] * u[t][:, None]  # (Dm, N)  식 (3)
        h = dA * h + dBu                                        # 식 (4) 앞부분
        y[t] = (h * C[t][None, :]).sum(axis=-1)                 # 식 (4) 뒷부분
        if return_states:
            states[t] = h

    if D is not None:
        y = y + u * D[None, :]
    if z is not None:
        y = y * silu(z)
    return (y, h, states) if return_states else (y, h)


def causal_depthwise_conv1d(x, w, b=None):
    """저장소 conv1d(groups=d_inner, kernel_size=4, padding=3)의 인과 버전.

    채널별로 독립인(depthwise) 1D 합성곱. 왼쪽만 0으로 채워 미래를 보지 않는다.
    Mamba가 '직전 3토큰'이라는 짧은 국소 문맥을 보는 유일한 장치다.
    """
    L, Dm = x.shape
    k = w.shape[1]
    pad = np.zeros((k - 1, Dm), dtype=x.dtype)
    xp = np.concatenate([pad, x], axis=0)
    out = np.zeros_like(x)
    for t in range(L):
        win = xp[t:t + k]                       # (k, Dm)
        out[t] = (win * w.T).sum(axis=0)
    if b is not None:
        out = out + b[None, :]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Mamba 블록
# ─────────────────────────────────────────────────────────────────────────────


def mamba_params(d_model, d_state=16, d_conv=4, expand=2, dt_rank=None, seed=0):
    """MyMamba.__init__()과 같은 모양·같은 초기화의 가중치 묶음을 만든다."""
    rng = np.random.default_rng(seed)
    d_inner = expand * d_model
    if dt_rank is None:
        dt_rank = math.ceil(d_model / 16)       # 저장소의 dt_rank='auto'

    # dt_proj 초기화: softplus(bias)가 [dt_min, dt_max]에 들어오게 (저장소와 동일)
    dt_min, dt_max, dt_floor = 0.001, 0.1, 1e-4
    dt = np.exp(rng.random(d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)).clip(min=dt_floor)
    inv_dt = dt + np.log(-np.expm1(-dt))        # softplus의 역함수
    dt_std = dt_rank ** -0.5

    return {
        "d_model": d_model, "d_inner": d_inner, "d_state": d_state,
        "d_conv": d_conv, "dt_rank": dt_rank,
        "in_proj": rng.normal(0, d_model ** -0.5, (d_model, 2 * d_inner)),
        "conv_w": rng.normal(0, 0.5, (d_inner, d_conv)),
        "conv_b": np.zeros(d_inner),
        "x_proj": rng.normal(0, d_inner ** -0.5, (d_inner, dt_rank + 2 * d_state)),
        "dt_proj_w": rng.uniform(-dt_std, dt_std, (dt_rank, d_inner)),
        "dt_proj_b": inv_dt,
        # S4D-real 초기화: A_log = log(1..N) → A = -exp(A_log) = -(1..N)
        "A_log": np.log(np.tile(np.arange(1, d_state + 1, dtype=np.float64),
                                (d_inner, 1))),
        "D": np.ones(d_inner),
        "out_proj": rng.normal(0, d_inner ** -0.5, (d_inner, d_model)),
    }


def mamba_block(x, W, return_states=False):
    """MyMamba.forward()의 numpy 축소판. x: (L, d_model) → (L, d_model)

    순서: in_proj로 x와 z로 갈라짐 → 인과 conv → SiLU → x_proj로 Δ·B·C를
    뽑음 → dt_proj+softplus로 Δ → selective_scan → out_proj.
    """
    N, dt_rank = W["d_state"], W["dt_rank"]
    xz = x @ W["in_proj"]                                   # (L, 2*d_inner)
    xin, z = np.split(xz, 2, axis=-1)
    xin = silu(causal_depthwise_conv1d(xin, W["conv_w"], W["conv_b"]))
    dbc = xin @ W["x_proj"]                                 # (L, dt_rank+2N)
    dt, Bm, Cm = np.split(dbc, [dt_rank, dt_rank + N], axis=-1)
    delta = softplus(dt @ W["dt_proj_w"] + W["dt_proj_b"])  # (L, d_inner)
    A = -np.exp(W["A_log"])                                 # 항상 음수 → 감쇠
    out = selective_scan(xin, delta, A, Bm, Cm, D=W["D"], z=z,
                         return_states=return_states)
    y = out[0]
    return (y @ W["out_proj"],) + out[1:]


# ─────────────────────────────────────────────────────────────────────────────
# 비교 대상: 어텐션 (O(L²))
# ─────────────────────────────────────────────────────────────────────────────


def causal_attention(x, Wq, Wk, Wv, Wo):
    """디코더 전용 트랜스포머 블록의 어텐션. 계산량이 L²에 비례한다."""
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    scores = q @ k.T / math.sqrt(q.shape[-1])               # (L, L) ← 여기가 L²
    mask = np.triu(np.ones_like(scores), k=1) * -1e9
    return (softmax(scores + mask) @ v) @ Wo


# ─────────────────────────────────────────────────────────────────────────────
# 6D 회전 표현 (저장소 manip.py의 loss_6d_rot)
# ─────────────────────────────────────────────────────────────────────────────


def gram_schmidt_6d(d6):
    """저장소 manip.py의 bgs(). 6개 숫자 → 정규직교 회전행렬 3×3.

    d6 (..., 6)을 두 벡터 a1, a2로 보고
      b1 = normalize(a1)
      b2 = normalize(a2 - (b1·a2) b1)
      b3 = b1 × b2
    결과 [b1 b2 b3]는 항상 SO(3) 원소다. 헤드가 9개 숫자를 그냥 뱉으면
    직교성이 깨지는데, 6D 표현은 어떤 6개 숫자가 나와도 유효한 회전이 된다.
    """
    d6 = np.asarray(d6, dtype=np.float64).reshape(-1, 2, 3)
    a1, a2 = d6[:, 0, :], d6[:, 1, :]
    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-12)
    a2p = a2 - (b1 * a2).sum(axis=1, keepdims=True) * b1
    b2 = a2p / (np.linalg.norm(a2p, axis=1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)                  # (M, 3, 3) 열이 축


def geodesic_loss(R_pred, R_gt):
    """논문 식 (6). arccos((tr(Rgt^T Rpred) − 1) / 2) = 두 회전 사이 각도(라디안).

    저장소 manip.py의 bgdR()과 같다. clamp로 arccos 정의역을 지킨다.
    """
    Rd = np.einsum("mij,mik->mjk", R_gt, R_pred)            # Rgt^T @ Rpred
    tr = np.trace(Rd, axis1=1, axis2=2)
    return np.arccos(np.clip(0.5 * (tr - 1.0), -1 + 1e-6, 1 - 1e-6))


# ─────────────────────────────────────────────────────────────────────────────
# 수동 역전파 MLP + Adam (학습이 필요한 실습용)
# ─────────────────────────────────────────────────────────────────────────────


class Linear:
    def __init__(self, inp, oup, rng, bias=True, xavier=True):
        # 저장소 manip.py는 정책 헤드를 xavier_uniform_ + bias=0으로 초기화한다
        if xavier:
            lim = math.sqrt(6.0 / (inp + oup))
            self.W = rng.uniform(-lim, lim, (inp, oup))
        else:
            self.W = rng.normal(0, inp ** -0.5, (inp, oup))
        self.b = np.zeros(oup) if bias else None
        self.gW = np.zeros_like(self.W)
        self.gb = np.zeros_like(self.b) if bias is not None and self.b is not None else None

    def __call__(self, x):
        self.x = x
        out = x @ self.W
        return out if self.b is None else out + self.b

    def backward(self, g):
        self.gW[...] = self.x.T @ g
        if self.b is not None:
            self.gb[...] = g.sum(axis=0)
        return g @ self.W.T

    def params(self):
        return [(self.W, self.gW)] + ([(self.b, self.gb)] if self.b is not None else [])

    def n_params(self):
        return self.W.size + (self.b.size if self.b is not None else 0)


class SpecialMLP:
    """저장소 manip.py의 SpecialMLP을 그대로 옮긴 것.

        fc1: Linear(inp, inp//2)   → ReLU
        fc2: Linear(inp//2, inp//4)→ ReLU
        fc3: Linear(inp//4, oup, bias=False)

    RoboMamba의 정책 헤드는 이 MLP 두 개다(head_type='two_mlp').
    head1: oup=2 (접촉점 x, y) / head2: oup=6 (6D 회전)
    """

    def __init__(self, inp, oup, rng):
        self.fc1 = Linear(inp, inp // 2, rng)
        self.fc2 = Linear(inp // 2, inp // 4, rng)
        self.fc3 = Linear(inp // 4, oup, rng, bias=False)

    def __call__(self, x):
        self.h1 = relu(self.fc1(x))
        self.h2 = relu(self.fc2(self.h1))
        return self.fc3(self.h2)

    def backward(self, g):
        g = self.fc3.backward(g)
        g = self.fc2.backward(g * (self.h2 > 0))
        return self.fc1.backward(g * (self.h1 > 0))

    def params(self):
        return self.fc1.params() + self.fc2.params() + self.fc3.params()

    def n_params(self):
        return self.fc1.n_params() + self.fc2.n_params() + self.fc3.n_params()


class Adam:
    """저장소가 쓰는 AdamW의 weight-decay 없는 버전 (논문 4.1절: AdamW, β=(0.9,0.999))."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.p = list(params)
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m = [np.zeros_like(w) for w, _ in self.p]
        self.v = [np.zeros_like(w) for w, _ in self.p]
        self.t = 0

    def step(self):
        self.t += 1
        for i, (w, g) in enumerate(self.p):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            w -= self.lr * mh / (np.sqrt(vh) + self.eps)

    def zero_grad(self):
        for _, g in self.p:
            g[...] = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 측정·출력 도구
# ─────────────────────────────────────────────────────────────────────────────


def timeit(fn, repeat=3):
    """가장 빠른 실행 시간(ms). 배경 잡음을 줄이려고 최소값을 쓴다."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def fmt_params(n):
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def table(headers, rows, highlight=None):
    """터미널용 표. highlight는 강조할 행 인덱스 집합."""
    cols = [max(len(str(h)), *(len(str(r[i])) for r in rows))
            for i, h in enumerate(headers)]
    line = "─".join("─" * c for c in cols)
    out = [" │ ".join(str(h).ljust(c) for h, c in zip(headers, cols)), line]
    for j, r in enumerate(rows):
        mark = " ◀" if highlight and j in highlight else ""
        out.append(" │ ".join(str(v).ljust(c) for v, c in zip(r, cols)) + mark)
    return "\n".join(out)


def save_svg(path, width, height, body, title=""):
    """의존성 없이 그림을 남긴다. 브라우저로 열어 본다."""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="ui-sans-serif,system-ui,sans-serif">'
        f'<rect width="{width}" height="{height}" fill="#fbfbf9"/>'
        + (f'<text x="{width/2}" y="24" text-anchor="middle" font-size="15" '
           f'font-weight="600" fill="#1a1a18">{title}</text>' if title else "")
        + body + "</svg>"
    )
    with open(path, "w") as f:
        f.write(svg)
    return path


def polyline(pts, color="#2f6f4f", w=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{w}"{d}/>'


def axes(x0, y0, x1, y1, xlabel="", ylabel=""):
    s = (f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#8a8a82"/>'
         f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#8a8a82"/>')
    if xlabel:
        s += (f'<text x="{(x0+x1)/2}" y="{y1+34}" text-anchor="middle" '
              f'font-size="11" fill="#55554f">{xlabel}</text>')
    if ylabel:
        s += (f'<text x="{x0-38}" y="{(y0+y1)/2}" text-anchor="middle" font-size="11" '
              f'fill="#55554f" transform="rotate(-90 {x0-38} {(y0+y1)/2})">{ylabel}</text>')
    return s


PAPER = "RoboMamba (arXiv 2406.04339v2, NeurIPS 2024)"
REPO = "lmzpai/roboMamba (main, 확인 2026-09-11)"


def header(n, title, guide):
    bar = "═" * 74
    return (f"\n{bar}\n  실습 {n} · {title}\n  가이드 {guide} · {PAPER}\n{bar}")
