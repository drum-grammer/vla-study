"""실습 5 · 정책 헤드 하나로 조작을 배우기 — 가이드 2.5 · 2.7

논문 3.4절의 주장을 그대로 재현한다.
  "once RoboMamba possesses sufficient reasoning capability, it can acquire
   pose prediction skills with minimal fine-tuning parameters and time."

저장소 manip.py의 구조를 그대로 쓴다.
  · SpecialMLP: Linear(h, h/2) → ReLU → Linear(h/2, h/4) → ReLU → Linear(h/4, out, bias=False)
  · head1 → 2개 (접촉점 x, y)          ... 논문 식 (5) L1 손실
  · head2 → 6개 (6D 회전 표현)          ... 논문 식 (6) 측지 손실
  · 백본은 전부 동결. 헤드만 학습.

세 가지를 확인한다.
  (1) 동결 백본 + 작은 헤드가 실제로 빠르게 학습되는가
  (2) 논문 그림 3 b)의 주장 — 백본의 '이해 수준'이 조작 정확도를 좌우하는가
  (3) 왜 회전을 9개 숫자가 아니라 6개로 내놓는가

실행: python3 lab5_policy_head_6d.py      (약 25초, out/lab5_head.svg 생성)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tinyssm as T

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

HID = 256          # 축소판 (실제 llm.hidden_size = 2560)
NTRAIN, NTEST = 1500, 600
STEPS, BATCH = 700, 64

print(T.header(5, "정책 헤드 하나로 조작을 배우기", "2.5 · 2.7"))

# ─────────────────────────────────────────────────────────────────────────────
# 장난감 '관절 물체 당기기' 데이터
# ─────────────────────────────────────────────────────────────────────────────


def make_scene(n, seed, y_rule="canonical"):
    """잠재 상태 z = [접촉점 x, y, 표면 법선 3개]와 정답 포즈.

    논문 4.1절의 데이터 수집 규칙:
      "randomly select a contact point p on the movable part and orient the
       end-effector's z-axis opposite to its normal vector, with a random
       y-axis direction"

    z축은 법선이 정해 주지만 **y축은 임의**라고 적혀 있다. 이 한 줄이
    회전 학습에 무슨 뜻인지가 이 실습 (2)의 주제다. 두 규칙을 다 만든다.
      y_rule="canonical" : y축을 월드 업벡터에서 결정 (장면의 함수)
      y_rule="random"    : 논문 문장 그대로 임의 (장면의 함수가 아님)
    """
    g = np.random.default_rng(seed)
    pos = g.uniform(0.15, 0.85, (n, 2))                  # 접촉점 (이미지 좌표)
    nrm = g.normal(0, 1, (n, 3))
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)    # 표면 법선
    z_axis = -nrm                                        # 그리퍼 z축 = 법선 반대
    if y_rule == "canonical":
        tmp = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))  # 월드 업
        deg = np.abs((tmp * z_axis).sum(1)) > 0.98        # z와 거의 평행하면
        tmp[deg] = np.array([0.0, 1.0, 0.0])              # 대체 기준축
        tmp = tmp + g.normal(0, 0.02, (n, 3))             # 수집 잡음
    else:
        tmp = g.normal(0, 1, (n, 3))
    y_axis = tmp - (tmp * z_axis).sum(1, keepdims=True) * z_axis
    y_axis /= np.linalg.norm(y_axis, axis=1, keepdims=True)
    x_axis = np.cross(y_axis, z_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=-1)      # (n,3,3) 열이 축
    return np.concatenate([pos, nrm], axis=1), pos, R


def backbone_features(z, quality, seed):
    """동결 백본이 내놓는 '전역 토큰'을 흉내 낸다.

    quality = 장면을 얼마나 이해했는가(= 논문이 말하는 reasoning ability).
      1.0 → 잠재 상태 5개가 모두 특징 안에 선형으로 들어 있다
      0.5 → 절반만 들어 있고 나머지 자리는 잡음
      0.0 → 전부 잡음 (장면을 전혀 못 읽음)
    """
    g = np.random.default_rng(seed)
    n, dz = z.shape
    keep = int(round(dz * quality))
    latent = np.concatenate([z[:, :keep], g.normal(0, 1, (n, dz - keep))], axis=1)
    M = np.random.default_rng(1234).normal(0, 1, (dz, HID)) / np.sqrt(dz)
    return latent @ M + g.normal(0, 0.05, (n, HID))


# ─────────────────────────────────────────────────────────────────────────────
# 헤드 (저장소 manip.py two_mlp와 같은 구조)
# ─────────────────────────────────────────────────────────────────────────────


def z_axis_error_deg(R_pred, R_gt):
    """그리퍼 z축(접근 방향)만의 각오차. 흡착 그리퍼는 z축 주위 회전이
    결과를 바꾸지 않으므로, 논문의 성공률에 실제로 걸리는 양은 이것이다."""
    zp, zg = R_pred[:, :, 2], R_gt[:, :, 2]
    return np.degrees(np.arccos(np.clip((zp * zg).sum(1), -1, 1)))


def geo_grad(d6, R_gt, eps=1e-5):
    """논문 식 (6)의 기울기. 6개 숫자에 대해서만 수치미분하고 나머지는 해석적.

    저장소 loss_6d_rot()를 그대로 손실로 쓰면서도 역전파가 가능한 방법이다.
    (6D → 3×3 Gram-Schmidt → arccos 경로를 손으로 미분하지 않아도 된다)
    """
    n = len(d6)
    base = T.geodesic_loss(T.gram_schmidt_6d(d6), R_gt)
    g = np.zeros_like(d6)
    for k in range(6):
        d = d6.copy()
        d[:, k] += eps
        g[:, k] = (T.geodesic_loss(T.gram_schmidt_6d(d), R_gt) - base) / eps
    return g / n, float(base.mean())


def train_heads(quality, seed=0, log=None, y_rule="canonical"):
    rng = np.random.default_rng(seed)
    ztr, ptr, Rtr = make_scene(NTRAIN, 10 + seed, y_rule)
    zte, pte, Rte = make_scene(NTEST, 500 + seed, y_rule)
    Ftr = backbone_features(ztr, quality, 20 + seed)
    Fte = backbone_features(zte, quality, 20 + seed)

    head1 = T.SpecialMLP(HID, 2, rng)      # 접촉점
    head2 = T.SpecialMLP(HID, 6, rng)      # 6D 회전
    opt = T.Adam(head1.params() + head2.params(), lr=3e-4)

    hist = []
    for s in range(STEPS):
        idx = rng.integers(0, NTRAIN, BATCH)
        f, pg, Rg = Ftr[idx], ptr[idx], Rtr[idx]

        # 식 (5) 위치: L1
        pred_p = head1(f)
        diff = pred_p - pg
        l_pos = float(np.abs(diff).mean())
        g_pos = np.sign(diff) / diff.size

        # 식 (6) 방향: 측지 거리
        pred_d = head2(f)
        g_dir, l_dir = geo_grad(pred_d, Rg)

        opt.zero_grad()
        head1.backward(g_pos)
        head2.backward(g_dir)
        opt.step()
        hist.append((l_pos, l_dir))
        if log and s % 175 == 0:
            print(f"    quality={quality:.1f}  step {s:4d}  "
                  f"L_pos {l_pos:.4f}  L_dir {np.degrees(l_dir):5.1f}°")

    # 테스트
    pp = head1(Fte)
    Rp = T.gram_schmidt_6d(head2(Fte))
    ang = np.degrees(T.geodesic_loss(Rp, Rte))
    zerr = z_axis_error_deg(Rp, Rte)
    perr = np.abs(pp - pte).mean(axis=1)
    succ = float(np.mean((perr < 0.1) & (zerr < 30.0)))
    return dict(quality=quality, hist=hist, pos_err=float(perr.mean()),
                ang=float(ang.mean()), zerr=float(zerr.mean()), succ=succ,
                y_rule=y_rule, n_params=head1.n_params() + head2.n_params())


# ─────────────────────────────────────────────────────────────────────────────
# (1) 동결 백본 + 작은 헤드
# ─────────────────────────────────────────────────────────────────────────────
print(f"""
(1) 백본 동결, 헤드만 학습 — 얼마나 빨리 배우는가
────────────────────────────────────────────────────────────────
헤드 구조는 저장소 manip.py와 동일(SpecialMLP ×2). 여기서는 hidden={HID}
축소판이고 실제는 2560이다. 학습 {STEPS}스텝 · 배치 {BATCH}.
손실은 논문 식 (5) L1(위치) + 식 (6) 측지거리(회전), 저장소와 같다.
""")
full = train_heads(1.0, log=True)
print(f"""
  최종  접촉점 평균오차   {full['pos_err']:.4f}   (이미지 좌표 0~1 기준)
        회전 전체 오차    {full['ang']:.2f}°   (식 6, 3×3 행렬 전체)
        그리퍼 z축 오차   {full['zerr']:.2f}°   (접근 방향만)
        성공률            {full['succ'] * 100:.1f}%   (위치<0.1 & z축<30°)
        학습 파라미터     {T.fmt_params(full['n_params'])}

논문 3.4절의 "a few dozen minutes on a single A100"이 이 구조 때문이다.
백본이 동결이면 역전파가 헤드 안에서만 돌고 옵티마이저 상태도 헤드 것만
들고 있으면 된다. 여기서는 노트북 CPU로 10초.""")

# ─────────────────────────────────────────────────────────────────────────────
# (2) 논문 4.1절의 'random y-axis' 한 줄이 뜻하는 것
# ─────────────────────────────────────────────────────────────────────────────
print("""
(2) 정답 회전이 애초에 예측 가능한가 — 논문 4.1절 데이터 수집 규칙 읽기
────────────────────────────────────────────────────────────────
논문 4.1절:
  "randomly select a contact point p on the movable part and orient the
   end-effector's z-axis opposite to its normal vector, **with a random
   y-axis direction** to interact with the object"

z축은 표면 법선이 정해 준다. 그런데 y축은 '임의'다. 그러면 정답 3×3 행렬은
장면의 함수가 아니고, z축 주위 회전각만큼 예측 불가능한 성분을 담는다.
식 (6)의 측지 손실은 그 성분까지 맞히라고 요구한다. 무슨 일이 생기는가.""")
rand_y = train_heads(1.0, seed=0, y_rule="random")
rows = [
    ["y축을 장면이 결정 (canonical)", f"{full['pos_err']:.4f}",
     f"{full['ang']:6.2f}°", f"{full['zerr']:6.2f}°", f"{full['succ'] * 100:5.1f}%"],
    ["y축을 임의로 (논문 문장 그대로)", f"{rand_y['pos_err']:.4f}",
     f"{rand_y['ang']:6.2f}°", f"{rand_y['zerr']:6.2f}°",
     f"{rand_y['succ'] * 100:5.1f}%"],
]
print(T.table(["데이터 수집 규칙", "접촉점 오차", "회전 전체 오차(식 6)",
               "z축 오차", "성공률"], rows, highlight={1}))
g0 = np.random.default_rng(3)
_, _, Ra = make_scene(600, 77, "random")
_, _, Rb = make_scene(600, 78, "random")
chance = float(np.degrees(T.geodesic_loss(Ra, Rb)).mean())
print(f"""
읽는 법
  · 임의 y축에서는 회전 전체 오차가 {rand_y['ang']:.1f}°에 멈춘다. 무작위 회전
    두 개 사이의 평균 거리가 {chance:.1f}°이니 **거의 학습이 안 된 수준**이다.
    당연하다. 정답의 그 성분은 장면에 정보가 없다. 학습은 잡음과 싸운다.
  · 그런데 **z축 오차는 {rand_y['zerr']:.1f}°까지 내려간다.** 예측 가능한 부분은
    제대로 배운 것이다. 성공률도 {rand_y['succ'] * 100:.0f}%로 거의 안 떨어진다.
  · 왜 성공률이 안 떨어지는가: 논문의 액추에이터는 **흡착 그리퍼**(4.1절,
    부록 D는 실기에서 양면테이프를 붙여 흡착으로 바꿨다고 적는다)다.
    빨판은 z축 주위로 돌려도 결과가 같다. 그리고 논문의 성공 판정은
    "물체 관절 상태 변화가 0.1 m 초과"(4.1절)이므로 z축만 맞으면 된다.

여기서 배울 것 (스터디용)
  · 논문이 회전 오차(도)를 표로 내지 않고 성공률만 보고하는 데는 이유가 있다.
    이 설정에서 식 (6)을 지표로 쓰면 상한이 막혀 있어 모델 비교가 안 된다.
  · 반대로, 정밀한 자세가 필요한 과제(2지 그리퍼로 손잡이를 잡는 등)로
    옮기면 이 데이터 수집 규칙 자체를 바꿔야 한다.
  · 논문 5절 한계에 이 얘기는 없다. '밝히지 않은 한계' 후보다.""")

# ─────────────────────────────────────────────────────────────────────────────
# (3) 논문 그림 3 b) 재현 — 백본의 이해 수준이 조작 정확도를 좌우한다
# ─────────────────────────────────────────────────────────────────────────────
print("""
(3) 백본이 장면을 얼마나 읽었나에 따라 (논문 그림 3 b) 구조 재현)
────────────────────────────────────────────────────────────────""")
runs = [full] + [train_heads(q, seed=i + 1) for i, q in enumerate((0.6, 0.4, 0.0))]
runs.sort(key=lambda r: -r["quality"])
rows = [[f"{r['quality']:.1f}", f"{r['pos_err']:.4f}", f"{r['zerr']:6.2f}°",
         f"{r['succ'] * 100:5.1f}%", T.fmt_params(r["n_params"])] for r in runs]
print(T.table(["백본 이해 수준", "접촉점 오차", "z축 오차", "성공률",
               "학습 파라미터"], rows, highlight={0}))
print("""
헤드 크기는 네 줄 모두 같다. 달라진 것은 **백본이 장면을 얼마나 담고
있는가**뿐인데 성공률이 그에 따라 움직인다. 논문 그림 3 b)와 같은 모양이다.
  OpenFlamingo 0.26 / LLaMA-AdapterV2 0.46 / Ours-1.4B 0.39
  / Ours-2.7B(co-training 없음) 0.61 / Ours-2.7B 0.63     (seen 성공률)
논문의 결론 문장 그대로: "fine-tuning an MLLM to learn robot skills does not
require extensive resources; it only requires that the MLLM possesses strong
robotic-related reasoning abilities."

주의: 논문 그림 3 b)는 서로 다른 실제 MLLM 네 개를 비교한 것이고 여기서는
같은 백본의 '정보량'만 인공적으로 줄였다. 인과 방향(이해 → 조작)을 보여 주는
장난감이지 논문 수치를 재현한 것이 아니다.""")

# ─────────────────────────────────────────────────────────────────────────────
# (4) 왜 6개 숫자인가
# ─────────────────────────────────────────────────────────────────────────────
print("""
(4) 회전을 9개가 아니라 6개로 내놓는 이유
────────────────────────────────────────────────────────────────
논문 식 (6)은 a_dir ∈ R^(3×3) 회전행렬을 쓴다고 적었지만, 저장소
manip.py의 head2는 **6개**를 내놓고 bgs()(Gram-Schmidt)로 3×3을 만든다.
신경망이 9개 숫자를 그냥 뱉으면 그것이 회전행렬일 보장이 없다.""")
g = np.random.default_rng(0)
raw9 = g.normal(0, 1, (2000, 9)).reshape(-1, 3, 3)
err9 = np.abs(np.einsum("mij,mik->mjk", raw9, raw9)
              - np.eye(3)[None]).max(axis=(1, 2))
det9 = np.linalg.det(raw9)
raw6 = g.normal(0, 1, (2000, 6))
R6 = T.gram_schmidt_6d(raw6)
err6 = np.abs(np.einsum("mij,mik->mjk", R6, R6) - np.eye(3)[None]).max(axis=(1, 2))
det6 = np.linalg.det(R6)
print(T.table(["표현", "직교성 오차 max|RᵀR−I| 평균", "det(R) 평균",
               "유효한 회전 비율"],
              [["9개 숫자 그대로", f"{err9.mean():.4f}", f"{det9.mean():+.4f}",
                f"{np.mean(err9 < 1e-6) * 100:.1f}%"],
               ["6개 + Gram-Schmidt", f"{err6.mean():.2e}", f"{det6.mean():+.4f}",
                f"{np.mean(err6 < 1e-6) * 100:.1f}%"]], highlight={1}))
print(f"""
무작위 6개 숫자를 Gram-Schmidt에 넣으면 **항상** det=+1인 정규직교행렬이
나온다. 9개를 그냥 쓰면 회전이 아닌 행렬이 나와 식 (6)의 arccos((tr−1)/2)가
정의역을 벗어난다. 저장소가 clamp(−1+1e−6, 1−1e−6)를 두는 이유이기도 하다.
부작용: 완벽히 맞혀도 손실이 0이 아니라 {np.degrees(np.arccos(1 - 1e-6)):.3f}°에서 바닥을 친다.

저장소 대응 ({T.REPO})
  SpecialMLP            : manip.py 18~30행  Linear(h,h/2)→ReLU→Linear(h/2,h/4)→ReLU→Linear(h/4,out,bias=False)
  head1/head2 (two_mlp) : manip.py 42~43행, xavier_uniform_ 초기화 44~53행
  전역 토큰 만드는 곳   : manip.py 82행
      res = (res[:, vision_encoded.shape[1]] + res[:, -1]) / 2
      → 논문 그림 2 설명은 "global token ... generated through a pooling
        operation from the language output tokens"라 쓰지만, 실제 two_mlp
        경로는 **이미지 토큰 직후 토큰과 마지막 토큰 딱 두 개의 평균**이다.
        AdaptiveAvgPool1d(1)는 ssm+mlp 경로에서만 쓰인다. → 논문·코드 상충
  lm_head 우회          : manip.py 64~66행  self.llm.mamba.lm_head = nn.Identity()
      → .logits이 어휘 점수(50280)가 아니라 2560차원 은닉상태가 되게 하는 기법
  6D → 회전행렬         : manip.py 116~122행 bgs()
  측지 손실 식 (6)      : manip.py 125~130행 bgdR()
  위치 출력이 2개인 근거: manip.py 42행 SpecialMLP(hidden, 2)
      → 논문 3.4절 "RoboMamba only predicts the 2D position (x, y) of the
        contact pixel, which is then translated into 3D space using depth"

해 볼 것
  · head1/head2를 하나로 합치면(저장소 head_type='mlp', 출력 8개) 어떻게 되는가?
    논문 표 6은 63.7% → 62.1%로 거의 같다고 한다.
  · 위치 손실을 L1(식 5)에서 MSE로 바꾸면 접촉점 오차가 어떻게 변하는가?
  · y_rule="random"에서 손실을 z축 각오차만으로 바꾸면 수렴이 빨라지는가?""")

# ─────────────────────────────────────────────────────────────────────────────
# 그림
# ─────────────────────────────────────────────────────────────────────────────
X0, Y0, X1, Y1 = 80, 56, 400, 270
body = T.axes(X0, Y0, X1, Y1, "학습 스텝", "회전 전체 오차 (도, 식 6)")
amax = 180.0
for r, col, lab in ((full, "#2f6f4f", "y축 = 장면이 결정"),
                    (rand_y, "#a8443f", "y축 = 임의 (논문 규칙)")):
    pts = [(X0 + i / STEPS * (X1 - X0),
            Y1 - min(np.degrees(v[1]), amax) / amax * (Y1 - Y0))
           for i, v in enumerate(r["hist"])]
    body += T.polyline(pts, col, 1.8)
yc = Y1 - chance / amax * (Y1 - Y0)
body += (f'<line x1="{X0}" y1="{yc:.0f}" x2="{X1}" y2="{yc:.0f}" stroke="#8a8a82" '
         f'stroke-dasharray="4 3"/><text x="{X0 + 6}" y="{yc - 6:.0f}" font-size="10" '
         f'fill="#55554f">무작위 회전 사이 평균 거리 {chance:.0f}°</text>')
body += (f'<text x="{X0 + 8}" y="{Y0 + 14}" font-size="11" fill="#2f6f4f">'
         f'■ y축 = 장면이 결정 → {full["ang"]:.1f}°</text>'
         f'<text x="{X0 + 8}" y="{Y0 + 29}" font-size="11" fill="#a8443f">'
         f'■ y축 = 임의 → {rand_y["ang"]:.1f}° (바닥)</text>')

P0, P1 = 490, 840
body += T.axes(P0, Y0, P1, Y1, "백본 이해 수준", "성공률")
w = (P1 - P0 - 40) / len(runs) * 0.55
for i, r in enumerate(sorted(runs, key=lambda x: x["quality"])):
    h = r["succ"] * (Y1 - Y0)
    x = P0 + 24 + i * (P1 - P0 - 44) / len(runs)
    body += (f'<rect x="{x:.0f}" y="{Y1 - h:.0f}" width="{w:.0f}" height="{h:.0f}" '
             f'fill="#2f6f4f" opacity="{0.35 + 0.16 * i:.2f}"/>'
             f'<text x="{x + w / 2:.0f}" y="{Y1 - h - 6:.0f}" text-anchor="middle" '
             f'font-size="10" fill="#1a1a18">{r["succ"] * 100:.0f}%</text>'
             f'<text x="{x + w / 2:.0f}" y="{Y1 + 16:.0f}" text-anchor="middle" '
             f'font-size="10" fill="#55554f">{r["quality"]:.1f}</text>')
p = T.save_svg(os.path.join(OUT, "lab5_head.svg"), 890, 315, body,
               "왼쪽: 정답에 예측 불가 성분이 있으면 손실은 바닥에 멈춘다 · 오른쪽: 논문 그림 3 b) 구조")
print(f"\n그림 저장: {p}")
