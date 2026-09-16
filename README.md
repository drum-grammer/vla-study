# VLA 논문 6편 읽기 (AI² Robotics) — 스터디 사이트

https://drum-grammer.github.io/vla-study/

AI² Robotics 계열 VLA 논문 6편을 3주에 읽는 스터디의 **메인 문서와 논문별 스터디 페이지**.
AI·트랜스포머 배경 없이 읽히도록 썼고, 실습은 GPU 없이 numpy만으로 돈다.

| 페이지 | 내용 |
|---|---|
| [index.html](https://drum-grammer.github.io/vla-study/) | 메인 문서 — 논문 6편·계보·3주 일정·읽는 법·부스 질문·자료 지도 |
| [paper-1-robomamba.html](https://drum-grammer.github.io/vla-study/paper-1-robomamba.html) | RoboMamba (arXiv 2406.04339) 스터디 페이지 — 개념 + 실습 7개 + 공유회 준비 |
| [paper-2-fis-vla.html](https://drum-grammer.github.io/vla-study/paper-2-fis-vla.html) | FiS-VLA (arXiv 2506.01953) 스터디 페이지 — 개념 + 실습 6개 + 공유회 준비 |
| `fis-vla-guide.html` · `robomamba-guide.html` | 정독 가이드 (심화) |
| `fis-vla-lab.html` · `robomamba-lab.html` | 실습 세션 상세 (결과 그림) |
| `lab-environment.html` | 실습 환경 (로컬 numpy · 코랩 · Apple Silicon) |
| [`labs/`](labs/) | 실습 코드 — `fis-vla/`, `robomamba/`. 파이썬 3.9+ 와 numpy만 |

```bash
git clone https://github.com/drum-grammer/vla-study
cd vla-study/labs/fis-vla && python3 lab1_tokens_blocks.py
```

정적 사이트다 — 서버도 데이터베이스도 없다. 이 저장소는 **생성물만** 담는다. 원본(마크다운 정본·HTML 소스·실습 코드)은
비공개 저장소 `drum-grammer/ai2-vla-paper-study`에 있고, 그 저장소의 GitHub Actions(`deploy-pages`)가 `main` push마다 여기로 밀어 넣는다.
여기서 직접 고치지 말 것 — 다음 배포에 덮어써진다.
