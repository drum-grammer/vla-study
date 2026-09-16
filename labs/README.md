# 실습 코드 — VLA 스터디

https://drum-grammer.github.io/vla-study/ 의 실습 세션이 가리키는 코드다. **GPU 없이, PyTorch 없이, numpy만으로** 돈다.

| 폴더 | 논문 | 안내 페이지 |
|---|---|---|
| `fis-vla/` | FiS-VLA (arXiv 2506.01953) — 스크립트 6개 + `tinynn.py` + `notebooks/` | https://drum-grammer.github.io/vla-study/paper-2-fis-vla.html |
| `robomamba/` | RoboMamba (arXiv 2406.04339) — 스크립트 7개 + `tinyssm.py` + `notebooks/robomamba_lab.ipynb` | https://drum-grammer.github.io/vla-study/paper-1-robomamba.html |

```bash
git clone https://github.com/drum-grammer/vla-study
cd vla-study/labs/fis-vla && python3 lab1_tokens_blocks.py
cd ../robomamba/labs   && python3 lab1_ssm_recurrence.py
```

코랩: `https://colab.research.google.com/github/drum-grammer/vla-study/blob/main/labs/<경로>.ipynb`

원본은 비공개 저장소에 있고 배포 때 여기로 복사된다. 여기서 직접 고치지 말 것.
