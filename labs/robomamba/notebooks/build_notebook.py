#!/usr/bin/env python3
"""상위 폴더(lab/)의 lab*.py 에서 자기완결형 주피터 노트북을 생성한다.

왜 생성하는가: 실습 스크립트가 정본이고 노트북은 그 사본이다. 손으로 두 벌을
관리하면 반드시 어긋난다. 이 스크립트를 다시 돌리면 노트북이 항상 최신이 된다.

왜 '자기완결형'인가: 코랩에서는 저장소 파일이 없다. tinyssm.py를 첫 셀에서
디스크에 써 두면 로컬 주피터와 코랩이 **같은 .ipynb 하나**로 동작한다.

실행: python3 build_notebook.py
결과: robomamba_lab.ipynb  (로컬 주피터 / 코랩 공용)
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LABS = os.path.dirname(HERE)          # 실습 스크립트는 상위 폴더(lab/)에 있다
OUT = os.path.join(HERE, "robomamba_lab.ipynb")

LAB_FILES = [
    ("lab1_ssm_recurrence.py", "상태공간모델을 손으로 굴려 보기", "lab1_state.svg"),
    ("lab2_linear_vs_quadratic.py", "어텐션은 왜 느리고 SSM은 왜 빠른가", "lab2_scaling.svg"),
    ("lab3_selective_gating.py", "'선택적' SSM이 무엇을 선택하는가", "lab3_selective.svg"),
    ("lab4_vlm_token_order.py", "이미지를 Mamba에 넣는 방법과 '순서'의 대가", "lab4_order.svg"),
    ("lab5_policy_head_6d.py", "정책 헤드 하나로 조작을 배우기", "lab5_head.svg"),
    ("lab6_param_audit.py", "논문의 '0.1%'를 직접 세어 보기", None),
    ("lab7_code_map.py", "논문 문장을 저장소 줄 번호로", None),
]


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def notebook_safe(src):
    """스크립트를 노트북 셀에서 그대로 돌 수 있게 고친다."""
    src = re.sub(r"^sys\.path\.insert\(0, os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)\n",
                 "", src, flags=re.M)
    src = src.replace('os.path.dirname(os.path.abspath(__file__))', '"."')
    src = src.replace('HERE = "."', 'HERE = "."')
    src = src.replace('"--offline" in sys.argv', 'os.path.exists(os.path.join(SRC, "readme.md"))')
    return src


cells = [
    md("# RoboMamba 실습 노트북\n",
       "\n",
       "논문 **RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning "
       "and Manipulation** (arXiv [2406.04339](https://arxiv.org/abs/2406.04339) v2, "
       "NeurIPS 2024) 을 손으로 굴려 보는 실습 7개.\n",
       "\n",
       "저장소: [lmzpai/roboMamba](https://github.com/lmzpai/roboMamba) · "
       "프로젝트: [robomamba-web](https://sites.google.com/view/robomamba-web)\n",
       "\n",
       "## 이 노트북을 어디서 돌리나\n",
       "\n",
       "| | 로컬 주피터 | 구글 코랩 |\n",
       "|---|---|---|\n",
       "| 준비물 | Python 3.9+ 와 numpy | 브라우저만 |\n",
       "| 설치 | 없음 (아래 셀이 확인) | 없음 |\n",
       "| 실습 1~7 | 전부 동작 | 전부 동작 |\n",
       "| 부록 (실제 Mamba 모델) | Apple Silicon은 MPS로 동작 | T4 GPU로 동작 |\n",
       "\n",
       "**코랩에서 여는 법**: 코랩 → `파일 > 노트북 업로드` 에 이 `.ipynb`를 "
       "끌어다 놓으면 된다. 아래 첫 셀이 필요한 파일을 스스로 만들기 때문에 "
       "저장소를 클론하지 않아도 된다.\n",
       "\n",
       "**순서**: 위에서 아래로 실행하면 된다. 실습 7만 네트워크가 필요하다 "
       "(GitHub에서 소스 6개, 약 200KB).\n",
       "\n",
       "---\n"),
    md("## 0. 부트스트랩 — 이 셀을 먼저 실행\n",
       "\n",
       "환경을 확인하고, 실습 공용 도구 `tinyssm.py`를 현재 디렉터리에 만든다. "
       "로컬에서 저장소 안에 있으면 기존 파일을 그대로 쓴다.\n"),
]

boot = '''import os, sys, platform

print("Python     :", sys.version.split()[0])
print("플랫폼      :", platform.platform())
IN_COLAB = "google.colab" in sys.modules
print("실행 환경   :", "구글 코랩" if IN_COLAB else "로컬 (주피터/IPython)")

try:
    import numpy as np
    print("numpy      :", np.__version__)
except ImportError:
    print("numpy 가 없다 →  %pip install numpy  를 실행하고 이 셀을 다시 돌려라")
    raise

os.makedirs("out", exist_ok=True)
print("\\n실습 1~7은 numpy만 쓴다. 추가 설치는 필요 없다.")
'''
cells.append(code(boot))

# tinyssm 원본을 셀에 심는다 (로컬 저장소에 있으면 덮어쓰지 않는다)
tiny = open(os.path.join(LABS, "tinyssm.py"), encoding="utf-8").read()
cells.append(md("### 0-1. 공용 도구 `tinyssm.py` 준비\n",
                "\n",
                "저장소 `tinyssm.py`와 **같은 내용**이다. 이미 같은 폴더에 "
                "있으면 이 셀은 건너뛴다(로컬에서 저장소를 클론한 경우).\n"))
cells.append(code(
    'import os\n'
    'if os.path.exists("tinyssm.py"):\n'
    '    print("tinyssm.py 가 이미 있다 — 그것을 쓴다")\n'
    'else:\n'
    '    src = _TINYSSM_SOURCE\n'
    '    open("tinyssm.py", "w", encoding="utf-8").write(src)\n'
    '    print(f"tinyssm.py 생성 ({len(src):,} 바이트)")\n'
    'import tinyssm as T\n'
    'print("import OK ·", T.PAPER)\n'))
# 원본 문자열을 별도 셀로 (가독성: 접어 두고 볼 수 있게)
cells.insert(len(cells) - 1, code(
    "# tinyssm.py 원본 (저장소 lab/tinyssm.py와 동일) — 내용을 읽고 싶으면 펼쳐 보라\n"
    "_TINYSSM_SOURCE = r'''" + tiny.replace("'''", "\\'\\'\\'") + "'''\n"
    "print(f'tinyssm 원본 {len(_TINYSSM_SOURCE):,} 바이트 준비')\n"))

for i, (fn, title, svg) in enumerate(LAB_FILES, start=1):
    src = open(os.path.join(LABS, fn), encoding="utf-8").read()
    doc = src.split('"""')[1].strip() if src.startswith('"""') else ""
    first = doc.splitlines()[0] if doc else title
    body = "\n".join(doc.splitlines()[1:]).strip()
    cells.append(md(f"---\n", f"\n## 실습 {i} · {title}\n", "\n",
                    f"```\n{body}\n```\n" if body else "\n",
                    f"\n원본 스크립트: `{fn}`\n"))
    cells.append(code(notebook_safe(src)))
    if svg:
        cells.append(code(
            f'from IPython.display import SVG, display\n'
            f'display(SVG(filename="out/{svg}"))\n'))

cells.append(md(
    "---\n",
    "\n## 부록 · 진짜 Mamba 모델을 돌려 보기 (선택)\n",
    "\n",
    "실습 1~7은 numpy 축소판이다. 여기서는 **실제 사전학습된 Mamba LLM**을 "
    "불러 RoboMamba의 백본이 무엇인지 눈으로 본다. RoboMamba 자체의 "
    "체크포인트는 저장소 `test` 브랜치에 있고 학습 코드는 비공개이므로, "
    "백본인 `state-spaces/mamba-*-hf`까지만 확인한다.\n",
    "\n",
    "| 환경 | 설치 | 130m 생성 속도 (측정값) |\n",
    "|---|---|---|\n",
    "| 코랩 (T4 GPU) | 기본 torch 사용 | 빠름 |\n",
    "| Mac (Apple Silicon, MPS) | `pip install torch transformers` | **89~145 tok/s** |\n",
    "| Mac (CPU) | 같음 | 0.83 tok/s — 쓰지 말 것 |\n",
    "\n",
    "`mamba_ssm` / `causal_conv1d`(저장소 requirements.txt)는 **CUDA 전용**이라 "
    "Mac에서는 빌드되지 않는다. 하지만 transformers가 순수 PyTorch 대체 경로를 "
    "갖고 있어(\"falling back to its reference PyTorch implementation... "
    "This is correct but much slower\") 결과는 같다. 측정: M2 Pro / "
    "torch 2.9.1 / transformers 5.1.1, 2026-09-11.\n"))
cells.append(code('''# 필요할 때만 실행 (약 1.5GB 다운로드). 실습 1~7은 이것 없이도 전부 동작한다.
# %pip install -q torch transformers

import time

try:
    import torch
    from transformers import AutoTokenizer, MambaForCausalLM
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    print("torch/transformers 가 없다. 위 %pip 줄의 주석을 풀고 이 셀을 다시 실행하라.")
    print("이 셀은 선택 사항이다 — 실습 1~7은 numpy만으로 모두 동작한다.")


def run_real_mamba(name="state-spaces/mamba-130m-hf"):
    if torch.cuda.is_available():
        dev = "cuda"                  # 코랩 T4
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = "mps"                   # Apple Silicon — CPU보다 100배 빠르다
    else:
        dev = "cpu"                   # 0.83 tok/s. 권하지 않는다
    print("장치:", dev)

    tok = AutoTokenizer.from_pretrained(name)
    m = MambaForCausalLM.from_pretrained(name).to(dev).eval()
    c = m.config
    print(f"{name}: {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M 파라미터")
    print(f"  hidden={c.hidden_size}  layers={c.num_hidden_layers}  "
          f"d_state={c.state_size}  d_conv={c.conv_kernel}  "
          f"expand={c.expand}  dt_rank={c.time_step_rank}")
    print("  → RoboMamba가 쓰는 2.8b는 hidden=2560, layers=64 (실습 6에서 센 값)")

    # 실습 1에서 '저장소 초기값은 A = −(1..16)'이라 했다. 학습 후에는 어떻게 됐나?
    A = -torch.exp(m.backbone.layers[0].mixer.A_log.float()).detach().cpu()
    print(f"\\n블록 0의 A (학습 후): 모양 {tuple(A.shape)}")
    print(f"  최솟값 {A.min():.2f}   최댓값 {A.max():.2f}")
    print(f"  채널 0의 {A.shape[1]}개 값: {[round(v, 2) for v in A[0].tolist()]}")
    print("  → S4D 초기값 −(1..16)에서 크게 벗어나 있다. 절댓값이 작은 채널")
    print("     (−0.1 근처)은 수백 토큰을 기억하고, 큰 채널(−200 이하)은 거의")
    print("     직전 토큰만 본다. 실습 1 (3)의 '기억 반경'이 채널마다 학습으로")
    print("     분화한 결과다 — 사람이 정해 준 것이 아니다.")

    ids = tok("The robot should first", return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        m.generate(ids, max_new_tokens=3, do_sample=False)          # 워밍업
        t0 = time.time()
        out = m.generate(ids, max_new_tokens=30, do_sample=False)
    dt = time.time() - t0
    print(f"\\n생성 30토큰 {dt:.2f}초 ({30 / dt:.1f} tok/s)")
    print("→", tok.decode(out[0], skip_special_tokens=True))
    print("\\n(130m은 로봇 데이터를 본 적이 없다. 답이 엉성한 것이 정상이고,")
    print(" 논문 Stage 1.2 instruction co-training이 무엇을 하는지 역으로 보여 준다.)")


if HAVE_TORCH:
    run_real_mamba()
'''))

cells.append(md(
    "---\n",
    "\n## 다음 단계\n",
    "\n",
    "- 실습이 가리키는 논문 절은 **RoboMamba 정독 가이드**를 함께 보라\n",
    "- 저장소 원본(`bash script/test.sh`)을 돌리려면 CUDA GPU + `test` 브랜치 "
    "체크포인트가 필요하다. 학습 코드는 비공개(저자 메일 요청)\n",
    "- 같은 스터디 1주차의 다른 논문 **FiS-VLA**(arXiv 2506.01953)는 별도 실습 "
    "폴더가 있다\n"))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.9"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"생성: {OUT}")
print(f"  셀 {len(cells)}개 ({sum(1 for c in cells if c['cell_type'] == 'code')} 코드 / "
      f"{sum(1 for c in cells if c['cell_type'] == 'markdown')} 마크다운)")
print(f"  크기 {os.path.getsize(OUT):,} 바이트")
