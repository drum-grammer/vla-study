"""실습 7 · 논문 문장을 저장소 줄 번호로 — 가이드 2.4 · 2.5 · 3.4

논문의 주장 하나하나가 코드의 어느 줄인지 직접 찾아 보여 준다.
줄 번호는 바뀌므로 **앵커 문자열**로 찾는다. 앵커가 사라지면 '찾지 못함'을
출력하고 계속 진행한다.

처음 실행하면 GitHub에서 파일 6개(약 200KB)를 받아 out/src/에 저장한다.
두 번째부터는 --offline로 네트워크 없이 돌린다.

실행: python3 lab7_code_map.py            (처음, 약 5초)
      python3 lab7_code_map.py --offline   (이후)
"""

import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tinyssm as T

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "out", "src")
BASE = "https://raw.githubusercontent.com/lmzpai/roboMamba/main"
FILES = ["readme.md", "requirements.txt", "src/script/test.sh", "src/model/vision.py",
         "src/model/vlm.py", "src/model/manip.py", "src/model/llm.py",
         "src/model/create.py"]
OFFLINE = "--offline" in sys.argv

# (파일, 앵커, 논문에서의 위치, 무엇을 확인하는가)
ANCHORS = [
    ("src/model/vision.py", "vision_encoders = {", "표 1 'Res.' 열 · 부록 표 4",
     "고를 수 있는 비전 인코더 네 개. test.sh는 CLIP224를 쓴다"),
    ("src/model/vision.py", "get_intermediate_layers", "3.2절 (논문에 없는 사실)",
     "마지막 층이 아니라 끝에서 두 번째 층을 꺼낸다"),
    ("src/model/llm.py", "mamba_dict = {", "4.1절 구현 상세 · 그림 3 a)",
     "mamba-2.8b / 1.4b / 790m / 370m. 논문 표는 2.7B, 그림 3은 1.4B도 비교"),
    ("src/model/vlm.py", "class Projector", "3.2절 'simple cross-modal connector'",
     "논문은 MLP라고만 쓴 연결기의 실제 모양 (3층, 중간 4배 확장)"),
    ("src/model/vlm.py", "vision_encoded_result, text_encoded", "3.2절 cat(f_v^L, f_t)",
     "이어 붙이는 순서가 [이미지, 텍스트]. Mamba라 순서가 결과를 바꾼다(실습 4)"),
    ("src/model/vlm.py", "target_modules=", "3.3절 (논문에 없는 사실)",
     "LoRA를 쓸 때 건드리는 Mamba 모듈 네 개"),
    ("src/model/manip.py", "class SpecialMLP", "3.4절 'simple policy head' · 부록 표 6",
     "정책 헤드의 실제 층 구성. 실습 6에서 파라미터를 센 대상"),
    ("src/model/manip.py", "self.action_head1 = SpecialMLP", "3.4절 'two types of MLPs'",
     "위치 2개 / 방향 6개로 나뉜 두 헤드 (head_type='two_mlp')"),
    ("src/model/manip.py", "self.llm.mamba.lm_head = nn.Identity()", "3.4절 (논문에 없는 사실)",
     "lm_head를 지워 .logits이 2560차원 은닉상태가 되게 하는 기법"),
    ("src/model/manip.py", "res = (res[:,vision_encoded.shape[1]]", "그림 2 'pooling operation'",
     "논문은 pooling이라 쓰지만 실제로는 토큰 두 개의 평균 → 상충"),
    ("src/model/manip.py", "def bgs(d6s)", "3.4절 a_dir ∈ R^(3×3)",
     "6개 숫자 → Gram-Schmidt → 회전행렬. 논문 본문에 6D 얘기는 없다"),
    ("src/model/manip.py", "theta = torch.clamp(0.5 * (Rt - 1)", "식 (6) 방향 손실",
     "arccos((tr−1)/2)와 정의역 보호 clamp"),
    ("src/model/manip.py", "self.ssm1 = MambaBlock", "부록 표 6 '(SSM block+MLP)×2'",
     "SSM 블록을 두 개 만든다. 45.2M과 맞지 않는 지점(실습 6)"),
    ("src/script/test.sh", "--llm_name mamba-2.8b", "4.1절 구현 상세",
     "평가 스크립트의 기본 설정. run_type VLM = 추론 평가, dataset robovqa"),
    ("readme.md", "The checkpoints are shown in the test branch", "공개 범위",
     "학습 코드는 공개되지 않았다 — 저자에게 메일로 요청해야 한다"),
    ("requirements.txt", "mamba_ssm", "재현 환경",
     "CUDA 커널 패키지. 이것 때문에 Apple Silicon에서는 저장소 원본이 안 돈다"),
]

print(T.header(7, "논문 문장을 저장소 줄 번호로", "2.4 · 2.5 · 3.4"))
print(f"\n저장소: https://github.com/lmzpai/roboMamba  (main, 확인 2026-09-11)")

os.makedirs(SRC, exist_ok=True)
for f in FILES:
    dst = os.path.join(SRC, f)
    if os.path.exists(dst):
        continue
    if OFFLINE:
        print(f"  [offline] 없음: {f}")
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        with urllib.request.urlopen(f"{BASE}/{f}", timeout=20) as r:
            data = r.read()
        with open(dst, "wb") as g:
            g.write(data)
        print(f"  받음 {f}  ({len(data):,} B)")
    except Exception as e:
        print(f"  실패 {f}: {e}")

cache = {}
for f in FILES:
    p = os.path.join(SRC, f)
    if os.path.exists(p):
        cache[f] = open(p, encoding="utf-8", errors="replace").read().splitlines()

found = missing = 0
cur_file = None
for f, anchor, where, why in ANCHORS:
    if f != cur_file:
        cur_file = f
        print(f"\n{'━' * 74}\n  {f}\n{'━' * 74}")
    lines = cache.get(f)
    if lines is None:
        print(f"  · 파일 없음 (offline?)  앵커 {anchor!r}")
        missing += 1
        continue
    hit = next((i for i, ln in enumerate(lines)
                if anchor.replace(" ", "") in ln.replace(" ", "")), None)
    if hit is None:
        print(f"  · 찾지 못함: {anchor!r}  ← 저장소가 바뀌었을 수 있다")
        missing += 1
        continue
    found += 1
    print(f"\n  ▸ {why}")
    print(f"    논문: {where}")
    lo, hi = max(0, hit - 1), min(len(lines), hit + 4)
    for i in range(lo, hi):
        mark = "▸" if i == hit else " "
        print(f"    {mark} {i + 1:4d} │ {lines[i][:96]}")

print(f"""
{'━' * 74}
앵커 {found}개 확인, {missing}개 실패.

이 표를 들고 확인할 것 (스터디 질문 후보)
  1. 논문 그림 2는 '전역 토큰을 pooling으로 만든다'고 하는데 코드는 토큰
     두 개의 평균이다. 어느 쪽이 표 2의 63%를 낸 설정인가?
  2. 부록 표 6의 3.7M과 코드의 8.20M 중 어느 쪽이 체크포인트와 맞는가?
     (test 브랜치 체크포인트를 열어 헤드 텐서 모양을 보면 판정된다)
  3. 학습 코드가 없으므로 Stage 1.1 → 1.2 → 2 파이프라인은 논문 서술만으로
     재현해야 한다. 협업 논의라면 여기가 가장 먼저 물어볼 지점이다.

저장소에 **없는** 것 (확인 2026-09-11)
  · 학습 코드 (readme: 메일로 요청)
  · SAPIEN 데이터 수집 스크립트, RoboVQA 전처리
  · 조작 평가 스크립트 (test.sh는 run_type VLM = 추론 평가만)
  · 체크포인트 (main 아님, test 브랜치)
  → FiS-VLA 저장소(학습·평가 스크립트 모두 공개)와 공개 범위가 크게 다르다.
    같은 회사가 공저자인데 왜 다른지도 부스 질문 후보다.""")
