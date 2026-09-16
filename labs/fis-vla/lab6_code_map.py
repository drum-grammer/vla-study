"""실습 6 — 코드 지도: 논문 문장을 저장소 줄 번호로 (가이드 2.4 · 2.5 · 2.6 · 3.4)

저장소를 통째로 받지 않고(LIFT3D 포함 700MB) 핵심 파일 5개만 GitHub에서 내려받아,
논문 개념이 구현된 줄을 앵커 문자열로 찾아 앞뒤 몇 줄과 함께 출력한다. 줄 번호가 바뀌어도 찾는다.

실행: python3 lab6_code_map.py            (네트워크 필요, 약 150KB 다운로드)
      python3 lab6_code_map.py --offline  (이미 받은 out/src/ 사용)
"""
import os, sys, urllib.request

REPO = "https://raw.githubusercontent.com/CHEN-H01/Fast-in-Slow/main/"
FILES = ["train.sh", "scripts/sim.py", "models/vlas/fisvla.py", "models/vlms/prismatic.py",
         "training/strategies/base_strategy.py"]

# (파일, 앵커 문자열, 논문 대응 설명, 앞뒤 줄 수)
ANCHORS = [
    ("train.sh", "LLM_MIDDLE_LAYER=", "System 1 = 블록 30 이후. 32블록 중 마지막 2개 (논문 4.2절 (1) '2블록에서 포화')", 1),
    ("train.sh", "SLOW_FAST_RATIO=", "주파수 비율 1:4 (논문 3.3절, 표 8)", 1),
    ("train.sh", "TRAINING_MODE=", "'async' = 비동기 학습 (논문 3.3절 asynchronous sampling)", 0),
    ("train.sh", "AR_DIFF_LOSS=", "자기회귀 손실을 함께 쓴다 = L_slow (논문 식 2·3)", 1),
    ("train.sh", "POINTCLOUD_POS=", "포인트클라우드를 System 1(fast) 쪽에 넣는다 (논문 3.3절 이종 입력, 부록 B.2 변형 실험)", 1),
    ("train.sh", "REPEATED_DIFFUSION_STEPS=", "한 배치를 4번 복제해 서로 다른 τ로 학습 (fisvla.py _repeat_tensor)", 0),
    ("models/vlas/fisvla.py", "noise = torch.randn_like(actions)", "논문 식 (1): 가우시안 노이즈 η", 3),
    ("models/vlas/fisvla.py", "x = self.diffusion.q_sample(actions, timestep, noise)", "논문 식 (1): √β·ã + √(1−β)·η", 1),
    ("models/vlas/fisvla.py", "loss = ((noise_pred - noise) ** 2).mean()", "논문 식 (1) L_fast = ‖η − π(…)‖²", 2),
    ("models/vlas/fisvla.py", "noise_schedule = 'squaredcos_cap_v2'", "노이즈 스케줄과 T=100 (논문 3.4절 T=100)", 1),
    ("models/vlas/fisvla.py", "def slow_system_forward", "System 2 단독 추론 → 중간 잠재 특징 (그림 2 왼쪽)", 2),
    ("models/vlas/fisvla.py", "def fast_system_forward", "System 1 단독 추론: 중간 특징 + 고주파 입력 → 행동 (그림 2 오른쪽)", 2),
    ("models/vlms/prismatic.py", "llm_layer_end=self.llm_middle_layer,", "LLM을 블록 0~30 (System 2)까지만 통과", 3),
    ("models/vlms/prismatic.py", "llm_layer_start=self.llm_middle_layer,", "블록 30~32 (System 1) 통과. 중간 특징 뒤에 고주파 토큰을 붙인 뒤", 3),
    ("models/vlms/prismatic.py", "if self.training_mode == 'sync': fast_projected_patch_embeddings = None", "동기 모드면 System 1에 별도 이미지를 주지 않는다 (비동기와의 차이)", 1),
    ("models/vlms/prismatic.py", "('head_slow_', 'slow'),", "카메라 이미지를 slow(System 2용)·fast(System 1용)로 나눠 인코딩 (그림 2의 Shared Encoder 두 개)", 6),
    ("training/strategies/base_strategy.py", "loss = loss + output.loss", "논문 식 (3): L = L_fast + L_slow (dual-aware co-training)", 3),
    ("scripts/sim.py", "if slow_cnt % int(args.slow_fast_ratio) == 0:", "n스텝마다 한 번만 System 2 실행, 나머지는 slow_latent_embedding 재사용 (1:4 비동기 추론)", 5),
    ("scripts/sim.py", "for action in actions:", "행동 청크를 순서대로 실행 (논문 부록 B.1, 청크 8 → 117.7 Hz)", 3),
    ("scripts/sim.py", "--ddim-steps", "확산 디노이징 반복 수 (test_rlbench.sh는 4). 청크와 별개인 속도 축", 0),
]


def fetch_all(dst):
    os.makedirs(dst, exist_ok=True)
    for f in FILES:
        path = os.path.join(dst, f.replace("/", "__"))
        if os.path.exists(path):
            continue
        print(f"  받는 중: {f}")
        urllib.request.urlretrieve(REPO + f, path)


def show(dst):
    for fname, anchor, note, ctx in ANCHORS:
        path = os.path.join(dst, fname.replace("/", "__"))
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
        hits = [i for i, l in enumerate(lines) if anchor in l]
        print("\n" + "─" * 100)
        print(f"[{fname}]  {note}")
        if not hits:
            print(f"  (앵커를 찾지 못함: {anchor!r} — 저장소가 바뀌었을 수 있다)")
            continue
        i = hits[0]
        for j in range(max(0, i - ctx), min(len(lines), i + ctx + 1)):
            mark = "▶" if j == i else " "
            print(f"  {mark} {j+1:5d} | {lines[j][:110]}")
        if len(hits) > 1:
            print(f"    (같은 앵커가 {len(hits)}곳: 줄 {', '.join(str(h+1) for h in hits)})")


if __name__ == "__main__":
    dst = os.path.join("out", "src")
    if "--offline" not in sys.argv:
        print("== GitHub에서 핵심 파일 5개만 내려받기 ==")
        fetch_all(dst)
    show(dst)
    print("\n※ 읽는 순서 제안: train.sh(설정) → sim.py(추론 루프) → fisvla.py(손실) → prismatic.py(블록 분기) → base_strategy.py(손실 합산)")
