# Final 20-Minute Submission Runbook

## 지금 할 일

1. `presentation/index.html`을 Chrome에서 열고 `F`로 전체화면을 확인한다.
2. Demo 1, Demo 2 성공, Demo 2 실패 영상이 자동 재생되는지 한 번만 확인한다.
3. GitHub 제출 링크는 `codex/demo-runloop-reflex` 브랜치로 제출한다.
4. 라이브 학습이나 신규 설치는 하지 않는다. 저장된 영상과 완료된 Reflex Session만 보여준다.

## 90초 핵심 발표 흐름

1. **문제:** 로봇별 demonstration data는 사람이 teleoperation으로 수집해 비싸고 확장이 어렵다.
2. **해결:** 고객이 로봇·태스크·성공 조건을 주면, 3인칭 사람 영상에서 robot-specific trajectory와 검증된 episode를 생성한다.
3. **Demo 1:** 91프레임 RGB → 손 검출 83/91 → Franka IK 91/91 → MuJoCo pick-and-place 성공, 목표 오차 0.383mm.
4. **Demo 2:** DexYCB RGB seed 2개에서 500개 검증 episode와 100,360 transitions 생성 → RunPod RTX 4090에서 6개 후보를 300 epochs 학습 → held-out 12/20 성공.
5. **운영:** Reflex의 실제 Agent 4개와 전용 Devbox 4개가 Reconstruction, Retargeting, Validation, Scaling 역할을 실행했고 모든 Gate를 통과했다.
6. **사업:** 로봇 회사가 스펙과 태스크를 주면 촬영 운영부터 검증된 학습 데이터 납품까지 제공한다.

## 영상 재생 순서

1. `video_demo/01_demo1_human_to_franka.mp4`
2. `video_demo/02_demo2_success_rollout.mp4`
3. `video_demo/03_demo2_failure_rollout.mp4`
4. 질문이 나오면 `results/reflex_4_agent_job/remote_artifacts/physical_validation/physics_rollout.mp4`를 실제 Reflex 회수본으로 보여준다.

## 반드시 정확히 말할 것

- `60%`는 전체 데이터의 성공률이 아니라 **학습과 분리된 held-out 환경 20개 중 12개 성공**이다.
- Scaling Agent의 `60%`는 별도로 **5회 증강 시도 중 3개 통과**한 작은 Agent I/O 증명이다.
- DexYCB seed는 3개 요청 중 실제 검증된 **2개만 사용**했다.
- 성공 rollout만 고르지 않고 실패 rollout도 함께 공개한다.
- 현재 4개 Reflex Agent는 실제 실행됐지만 **cross-Devbox 자동 artifact handoff는 아직 없다**. 후단 Agent는 같은 원본 영상에서 upstream을 재생성했다.
- 현재 프로토타입은 MediaPipe 기반 RGB pipeline이다. 범용 VLM이나 완전한 metric human-motion reconstruction이라고 과장하지 않는다.

## 제출 링크

- Repository branch: `https://github.com/Sehyeogkim/codex_hackathon/tree/codex/demo-runloop-reflex`
- Reflex evidence commit: `9a05682`
- Reflex evidence: `presentation/results/reflex_4_agent_job/`
