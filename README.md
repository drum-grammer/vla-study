# 피지컬AI 논문 읽기 — 스터디 사이트

https://drum-grammer.github.io/physical-ai-study/

피지컬AI 논문을 주제별로 분류하고, 논문마다 **스터디 페이지(개념 + numpy 실습 + 공유회 준비) → 정독 가이드 → 실습 상세**를 붙인다.
AI·트랜스포머 배경 없이 읽히도록 썼고, 실습은 GPU 없이 numpy만으로 돈다.

| 주소 | 내용 |
|---|---|
| [`/`](https://drum-grammer.github.io/physical-ai-study/) | 홈 — 주제별 논문 목록, 컬렉션, 공부하는 법 |
| `/papers/<slug>/` | 논문 스터디 페이지. `guide.html`(정독 가이드) · `lab.html`(실습 상세)이 곁에 있다 |
| `/collections/<이름>/` | 여러 편을 묶어 읽는 스터디 — 일정 · 계보 · 질문 |
| [`/lab-environment.html`](https://drum-grammer.github.io/physical-ai-study/lab-environment.html) | 실습 환경 (로컬 numpy · 코랩 · Apple Silicon) |
| [`labs/`](labs/) | 실습 코드 — 논문별 폴더. 파이썬 3.9+ 와 numpy만 |

지금 있는 것: [RoboMamba](https://drum-grammer.github.io/physical-ai-study/papers/robomamba/) · [FiS-VLA](https://drum-grammer.github.io/physical-ai-study/papers/fis-vla/) · 컬렉션 [AI² Robotics VLA 논문 6편](https://drum-grammer.github.io/physical-ai-study/collections/ai2-robotics-vla-6/)

```bash
git clone https://github.com/drum-grammer/physical-ai-study
cd physical-ai-study/labs/fis-vla && python3 lab1_tokens_blocks.py
```

정적 사이트다 — 서버도 데이터베이스도 없다. 이 저장소는 **생성물만** 담는다. 원본(마크다운 정본·HTML 소스·실습 코드)은
비공개 저장소 `drum-grammer/physical-ai-papers`에 있고, 그 저장소의 GitHub Actions(`deploy-pages`)가 `main` push마다 여기로 밀어 넣는다.
여기서 직접 고치지 말 것 — 다음 배포에 덮어써진다.
