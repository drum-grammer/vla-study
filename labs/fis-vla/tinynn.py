"""tinynn.py — 실습 공용 최소 신경망 (numpy만 사용)

FiS-VLA 실습에서 PyTorch 없이 학습을 보여 주기 위한 아주 작은 도구 상자.
  - MLP: 완전연결 신경망 (GELU), 수동 역전파
  - Adam: 최적화기
  - attention / transformer_block: 트랜스포머 블록 순전파 (학습은 안 함, 구조 관찰용)

논문·저장소 대응
  - MLP는 저장소의 ActionEmbedder / TimestepEmbedder / FinalLayer(models/diffusion/models.py)가 하는 일과 같다
  - transformer_block은 LLaMA2 블록 32개 중 하나의 뼈대를 흉내 낸다 (models/backbones/llm/llama2.py가 불러오는 것)

이 파일은 직접 실행하지 않고 lab 스크립트가 import한다.
"""
import numpy as np


# ── 활성 함수 ─────────────────────────────────────────────────────────────
def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def gelu_grad(x):
    t = np.tanh(0.7978845608 * (x + 0.044715 * x ** 3))
    dt = (1 - t ** 2) * 0.7978845608 * (1 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1 + t) + 0.5 * x * dt


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ── MLP (수동 역전파) ─────────────────────────────────────────────────────
class MLP:
    """sizes=[in, h1, h2, ..., out]. forward → backward(dout) → params/grads."""

    def __init__(self, sizes, rng, scale=None):
        self.W, self.b = [], []
        for a, c in zip(sizes[:-1], sizes[1:]):
            s = scale if scale is not None else np.sqrt(2.0 / a)
            self.W.append(rng.normal(0, s, (a, c)))
            self.b.append(np.zeros(c))
        self._cache = None

    def params(self):
        return self.W + self.b

    def forward(self, x):
        acts, pres = [x], []
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            pres.append(z)
            h = gelu(z) if i < len(self.W) - 1 else z
            acts.append(h)
        self._cache = (acts, pres)
        return h

    def backward(self, dout):
        """dout: dL/d(output). returns grads list aligned with params(), and dL/d(input)."""
        acts, pres = self._cache
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        d = dout
        for i in reversed(range(len(self.W))):
            if i < len(self.W) - 1:
                d = d * gelu_grad(pres[i])
            gW[i] = acts[i].T @ d
            gb[i] = d.sum(axis=0)
            d = d @ self.W[i].T
        return gW + gb, d


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ── 트랜스포머 블록 (순전파만) ───────────────────────────────────────────
def layer_norm(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def attention(x, Wq, Wk, Wv, causal=False):
    """x: (T, d). 반환: (출력 (T, d), 어텐션 가중치 (T, T))"""
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    if causal:
        T = x.shape[0]
        scores = np.where(np.tril(np.ones((T, T), bool)), scores, -1e9)
    A = softmax(scores, axis=-1)
    return A @ V, A


class TransformerBlock:
    """LLaMA 블록의 뼈대: x → x + Attn(LN(x)) → x + MLP(LN(x)). 입력·출력 모양이 같다."""

    def __init__(self, d, rng, d_ff=None):
        d_ff = d_ff or 4 * d
        s = 1.0 / np.sqrt(d)
        self.Wq, self.Wk, self.Wv, self.Wo = (rng.normal(0, s, (d, d)) for _ in range(4))
        self.W1 = rng.normal(0, s, (d, d_ff))
        self.W2 = rng.normal(0, 1.0 / np.sqrt(d_ff), (d_ff, d))
        self.last_attn = None

    def __call__(self, x, causal=True):
        a, A = attention(layer_norm(x), self.Wq, self.Wk, self.Wv, causal)
        self.last_attn = A
        x = x + a @ self.Wo
        x = x + gelu(layer_norm(x) @ self.W1) @ self.W2
        return x


def run_blocks(blocks, x, start=0, end=None, causal=True):
    """blocks[start:end]만 통과시킨다. 저장소의 llm_layer_start / llm_layer_end 인자와 같은 역할."""
    end = len(blocks) if end is None else end
    for blk in blocks[start:end]:
        x = blk(x, causal)
    return x


# ── 확산 스케줄 (저장소와 같은 squaredcos_cap_v2) ───────────────────────
def squaredcos_cap_v2_betas(T):
    """models/diffusion/models.py get_named_beta_schedule('squaredcos_cap_v2')와 같은 식."""
    def alpha_bar(t):
        return np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
    betas = []
    for i in range(T):
        t1, t2 = i / T, (i + 1) / T
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
    return np.array(betas)


def make_schedule(T):
    betas = squaredcos_cap_v2_betas(T)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    return betas, alphas, alpha_bar


def q_sample(x0, t, noise, alpha_bar):
    """논문 식 (1)의 √β_τ·ã + √(1−β_τ)·η. 저장소 diffusion.q_sample와 같은 일."""
    ab = alpha_bar[t][:, None]
    return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * noise


# ── 아주 단순한 SVG 그리기 (matplotlib 없이) ─────────────────────────────
class SVG:
    def __init__(self, w=520, h=360, pad=40, xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), title=""):
        self.w, self.h, self.pad, self.xlim, self.ylim = w, h, pad, xlim, ylim
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
                      f'font-family="system-ui, sans-serif" font-size="12">',
                      f'<rect width="{w}" height="{h}" fill="#fff"/>',
                      f'<rect x="{pad}" y="{pad}" width="{w-2*pad}" height="{h-2*pad}" fill="none" stroke="#bbb"/>']
        if title:
            self.parts.append(f'<text x="{pad}" y="{pad-14}" font-size="14" font-weight="700" fill="#222">{title}</text>')

    def _xy(self, x, y):
        (x0, x1), (y0, y1) = self.xlim, self.ylim
        px = self.pad + (x - x0) / (x1 - x0) * (self.w - 2 * self.pad)
        py = self.h - self.pad - (y - y0) / (y1 - y0) * (self.h - 2 * self.pad)
        return px, py

    def points(self, pts, color, r=2.5, opacity=0.8):
        for x, y in pts:
            px, py = self._xy(x, y)
            self.parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r}" fill="{color}" opacity="{opacity}"/>')

    def line(self, pts, color, width=1.5, dash=""):
        d = " ".join(f"{'M' if i == 0 else 'L'}{self._xy(x, y)[0]:.1f},{self._xy(x, y)[1]:.1f}" for i, (x, y) in enumerate(pts))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{da}/>')

    def label(self, x, y, text, color="#222", size=12):
        px, py = self._xy(x, y)
        self.parts.append(f'<text x="{px:.1f}" y="{py:.1f}" fill="{color}" font-size="{size}">{text}</text>')

    def legend(self, items, x=None, y=None):
        x = x if x is not None else self.pad + 8
        y = y if y is not None else self.pad + 16
        for i, (color, text) in enumerate(items):
            yy = y + i * 16
            self.parts.append(f'<rect x="{x}" y="{yy-9}" width="10" height="10" fill="{color}"/>')
            self.parts.append(f'<text x="{x+16}" y="{yy}" fill="#222">{text}</text>')

    def save(self, path):
        self.parts.append("</svg>")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.parts))
        return path
