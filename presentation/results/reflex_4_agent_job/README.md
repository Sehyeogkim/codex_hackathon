# Actual Reflex 4-Agent Run — `demo-91f-reflex`

이 디렉터리는 하나의 91프레임 영상을 네 개의 실제 Reflex Agent Session에서
처리한 감사 가능한 실행 증거다. 네 Agent는 각자 전용 Runloop Devbox에서
실행됐고, 완료 후 Runloop API로 아래 artifact를 로컬에 회수했다.

공통 입력은 `data/demo2/do_as_i_do_pick_place_preview.mp4`이며 SHA256은
`7ccb44d5e0115f61a401cc79466ed83703541ab9be65ace93d766fe9b58ff4d8`다.
로봇은 Franka Panda, grasp/release frame은 30/70이다.

| Agent | 실제 입력 | 실제 출력 | 측정값 | Gate |
|---|---|---|---|---|
| Reconstruction | 91-frame RGB MP4 | `vision.json`, report | 손 검출 83/91 = 91.2%; index/timestamp valid | PASS |
| Retargeting | locally bootstrapped `vision.json`, config | canonical EE + Panda trajectory | 91→91 frames; invalid IK 0; joint violations 0 | PASS |
| Physical Validation | locally bootstrapped Panda trajectory, config | validation JSON + 187-frame MP4 | task success; collision-free; target error 0.383 mm | PASS |
| Data Scaling | validated Panda seed, config | 3 validated episodes | 5 attempted; 3 accepted; 2 rejected; pass rate 60% | PASS |

## Reflex Session evidence

| Agent | Reflex Agent ID | Runloop Devbox ID | Final turn |
|---|---|---|---|
| Reconstruction | `agt_35oqb8Ykk56YROp4eDFQsh` | `dbx_34Fp60gRXLf6zGmZpjOyo` | completed, seq 18014 |
| Retargeting | `agt_2dxNkwPXFt98QlFWG0AnPj` | `dbx_34Fp61K7AYs7SrmAm86eF` | completed, seq 14393 |
| Physical Validation | `agt_5vvglwdDfNZ0VattM4CGFW` | `dbx_34Fp62BjP8Bh3q2R4YY2o` | completed, seq 10051 |
| Data Scaling | `agt_04NQTlMSwiUmPsaZVVtm2m` | `dbx_34Fp62vwBnFBrOBZ72zfn` | completed, seq 7629 |

## Recovered artifacts

- `remote_artifacts/reconstruction/`: MediaPipe observations and Agent reports
- `remote_artifacts/retargeting/`: canonical EE and Franka joint trajectories
- `remote_artifacts/physical_validation/`: MuJoCo validation JSON and rollout MP4
- `remote_artifacts/scaling/`: three accepted episodes plus two reported rejections
- `run_summary.json`: machine-readable Session, I/O, metric, hash, and Gate summary

## 정확한 해석

이 실행은 **네 실제 Reflex Session이 각 역할의 코드를 수행하고 입출력을 남긴
증거**다. 아직 cross-Devbox artifact handoff가 구현되지 않았기 때문에 순차
Agent-to-Agent 전달은 아니며, Retargeting·Validation·Scaling Agent는 동일한
원본 영상으로 필요한 upstream을 각 Devbox에서 재생성했다. 모든 report에서
bootstrap과 owned output을 분리했다.

Session은 코드 정리 이전에 생성되어 당시 tracked 경로인 `src.reflex_stages`와
`config/demo_config.json`을 사용했다. 현재 제출 브랜치의 대응 경로는
`persona.reflex_stages`와 `dataminer/config/demo_config.json`이다. 입력 영상의
SHA256은 동일하다.

Scaling의 60%는 5회의 데이터 증강 물리검증 중 3개가 통과했다는 뜻이며,
별도 RunPod 학습의 held-out 12/20 결과와 혼동하면 안 된다.
