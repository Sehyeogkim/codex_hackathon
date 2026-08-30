# 데모 계획

## 목표

사람의 작업 영상을 특정 로봇의 학습 데이터로 변환하고, 생성된 데이터로 가상 로봇을 학습할 수 있음을 보여준다.

## 공통 설정

- 고객 로봇: Franka Panda 7-DoF 로봇팔
- 태스크: 머스터드 병을 A에서 집어 B 위치로 이동하고 유지하기
- 촬영: 고정된 3인칭 카메라, 정면 yaw 0°·하향 35°
- 출력: 시간별 end-effector pose, 7개 관절각, gripper 상태

## 데모 1: 직접 촬영한 영상 변환

1. 촬영 가이드에 따라 발표자가 병을 옮기는 영상을 촬영한다.
2. 영상에서 손과 병의 움직임을 추출한다.
3. 움직임을 Franka의 작업 공간에 맞게 변환한다.
4. MuJoCo IK로 Franka joint trajectory를 생성한다.
5. MuJoCo에서 trajectory를 재생한다.
6. 결과를 JSON 또는 CSV로 내보낸다.

### 보여줄 결과

- 원본 사람 영상과 가상 Franka 동작 비교
- 시간에 따른 7개 관절각과 gripper 값
- joint limit, 충돌, trajectory 부드러움 검사

## 데모 2: 공개 영상으로 데이터 생성 및 학습

1. DexYCB의 고정 3인칭 pick-up 영상을 사용한다.
2. 모든 영상을 Franka joint trajectory로 일괄 변환한다.
3. 시작점과 목표점을 바꾼 시뮬레이션 episode를 추가 생성한다.
4. 시뮬레이터에서 observation과 action을 함께 기록한다.
5. RunPod GPU에서 간단한 Behavior Cloning 정책을 학습한다.
6. 새로운 병 위치에서 학습된 정책을 평가한다.
7. 성공한 rollout을 영상으로 저장한다.

### 보여줄 결과

- 원본 영상 수와 생성된 episode 수
- 학습 데이터 샘플
- 학습 과정과 성공률
- 학습된 로봇의 새로운 환경 수행 영상

## 구현 구조

```text
사람 영상
→ MediaPipe 손 landmark
→ 카메라 좌표를 테이블 좌표로 변환
→ 손 경로와 gripper 상태 생성
→ trajectory smoothing
→ MuJoCo Franka IK
→ joint trajectory 데이터셋
→ Behavior Cloning 학습
→ 가상 로봇 평가
```

## Runloop·Reflex 에이전트 데모

- 실제 계산: 기존 Python 파이프라인이 수행한다.
- Runloop: 고객별 작업을 격리된 Devbox에서 실행하고 결과 파일을 회수한다.
- Reflex: 작업 요청, 진행 이벤트, 결과 요약과 재실행 기록을 보여준다.
- RunPod: GPU 기반 학습이나 Isaac Sim이 필요할 때만 사용한다.

```text
고객 영상 + 로봇 설정 + 태스크
→ Reflex Robot Data Agent 요청
→ Runloop Devbox 생성
→ 영상 → 손 추출 → retargeting → Panda IK
→ joint trajectory JSON 반환
→ Reflex에서 결과와 실행 기록 확인
```

로컬 검증:

```bash
.venv/bin/python -m dataminer \
  data/demo2/do_as_i_do_pick_place_preview.mp4 \
  --config dataminer/config/demo_config.json \
  --output-dir artifacts/demo1_local \
  --grasp-frame 30 --release-frame 70
```

Runloop 원격 실행:

```bash
export RUNLOOP_API_KEY='...'
.venv/bin/python -m dataminer.product_runloop \
  dataminer/config/demo_request.json \
  --output-dir artifacts/runloop_demo
```

제품 백엔드는 위 명령으로 Devbox를 만들고, Reflex Agent에서는 Reflex가
이미 Devbox를 만들기 때문에 `dataminer.product_request`를 직접 실행한다. Reflex에서는
[`../persona/robot_data_agent_prompt.md`](../persona/robot_data_agent_prompt.md)를
사용한다. MP4는 Reflex 첨부 제한 때문에
직접 첨부하지 않고 저장소 또는 다운로드 URL로 전달한다.

`Do as I Do` 전체 코드는 사용하지 않는다. 해당 코드는 Sharpa 손과 dual UR3e를 대상으로 하며 GPU와 설치 요구사항이 크다. Reconstruction과 retargeting의 단계 구성 및 검증 방식만 참고한다.

## 실행 전략

- Runloop: 고객 영상 변환과 MuJoCo 물리 검증을 격리 Devbox에서 실행한다.
- RunPod: subject-07 다운로드, 요청한 3개 중 검증 가능한 DexYCB RGB sequence
  2개를 변환하고, 500개 검증 episode 생성, phase-conditioned BC 학습과 20회
  평가를 한 Secure GPU Pod에서 실행한다.
- Reflex: GitHub 브랜치를 clone하고 두 작업의 로그·실패 원인·artifact를 추적한다.
- Isaac Sim은 사용하지 않는다. 모든 물리 검증과 발표 영상은 MuJoCo로 만든다.

RunPod dry-run:

```bash
.venv/bin/python -m runpod_workflow_train
```

실제 실행은 `.env`의 `RUNPOD_API_KEY`와 RunPod에 등록된 SSH 공개키가 있을 때만 한다.

```bash
set -a; source .env; set +a
.venv/bin/python -m runpod_workflow_train \
  runpod_workflow_train/config/training_request.json \
  --execute --ssh-key ~/.ssh/id_ed25519_runpod \
  --output-dir artifacts/runpod_final
```

Pod는 RTX 4090 → A40 → L4 순서로 요청하며, 성공·실패 모두 `finally`에서 삭제한다.

## 데이터 전략

- 정확성 검증: DexYCB subject-07의 오른손 머스터드 병 sequence를 3개
  요청했으나 조건을 통과한 2개만 사용하며, 이 제한을 selection provenance에 기록
- 확장성 설명: Something-Something V2의 220,847개 hand-object 영상
- 한계: 대규모 3인칭 영상과 정확한 3D pose를 동시에 제공하는 공개 데이터는 없음
- DexYCB에는 place가 없으므로 RGB 기반 pickup과 생성 carry/place/release를 명시해 결합한다.
- DexYCB GT pose/depth는 trajectory 입력으로 사용하지 않는다.
- 현재 RGB XY는 데모 homography, Z는 phase prior다. metric human-motion reconstruction이
  아니라 고객 장면의 물체·목표 anchor로 다시 컴파일되는 robot seed다.
- DexYCB는 CC BY-NC 4.0이므로 연구·비상업 데모 용도로만 사용한다.

## 발표 순서

1. 로봇 데이터 수집 비용과 텔레옵의 한계를 설명한다.
2. 직접 촬영한 영상을 Franka trajectory로 변환한다.
3. 공개 영상 여러 개를 데이터셋으로 변환한 결과를 보여준다.
4. 생성 데이터로 학습한 가상 Franka의 rollout MP4를 재생한다.
5. 고객이 로봇 사양과 태스크를 주면 같은 방식으로 데이터를 납품할 수 있음을 설명한다.

## 성공 기준

- 손 검출률 70% 이상, 사람 영상 한 개가 유효한 Franka episode로 변환된다.
- 생성된 모든 관절값이 joint limit 안에 있다.
- 시뮬레이션에서 충돌 없이 병을 옮기고 목표 거리가 7cm 미만이다.
- DexYCB에서 요청한 3개와 실제 검증·선택된 2개의 수를 모두 기록하고,
  선택된 2개 sequence의 provenance와 human/generated 구간을 보존한다.
- 검증 통과 episode를 정확히 500개 생성한다.
- 학습과 분리된 고정 seed 20개 중 10개 이상 성공해야 학습 데모를 통과시킨다.
- 기준 미달이면 실제 수치를 공개하고 성공 처리하지 않는다.

## 현재 검증 결과

- 로컬: 91/91 IK, 손 검출률 93.4%, `task_success=true`, 목표 거리 0.29mm
- Runloop 실제 Devbox: 91/91 IK, `task_success=true`, 충돌 없음, 목표 거리 0.38mm
- 테스트: 80개 통과
- RunPod: subject-07 sequence 3개 요청 중 2개를 검증·선택했고, 물리 검증
  episode 500개(총 100,360 step)를 생성했다. 실제 학습 성공은 회수된
  summary/checkpoint와 분리된 20회 평가가 있을 때만 주장한다.

## 핵심 메시지

사람 영상 하나를 로봇 학습용 episode 하나로 변환할 수 있다. 이 과정을 여러 작업자와 영상에 적용하면 비싼 로봇 텔레옵을 줄이고 Physical AI 학습 데이터를 대규모로 생성할 수 있다.
