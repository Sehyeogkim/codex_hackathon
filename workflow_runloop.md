# Runloop·Reflex 4-Agent 데모 워커 풀

## 구조

데모는 역할이 고정된 Reflex Agent 4개와 각 Agent의 전용 Runloop
Devbox를 사용한다.

```text
Reconstruction Agent    / Devbox R
Retargeting Agent       / Devbox T
Physical Validation     / Devbox V
Data Scaling Agent      / Devbox S
```

- 영상마다 Agent를 새로 만들지 않는다.
- 동일한 4개 role session을 유지하고, 요청을 `job_id`로 구분해 순차
  처리한다.
- 각 Devbox에서는 `/workspace/jobs/<job_id>/` 아래 경로만 사용한다.
- Reflex Agent 안에서 별도의 Runloop Devbox를 생성하지 않는다.
- 동시에 영상이 여러 개면 role별 큐에 넣어 pipeline 단계를 겹쳐
  실행한다. 하나의 role session은 한 시점에 job 하나만 수정한다.

```text
video-001: R → T → V → S
video-002:     R → T → V → S
video-003:         R → T → V → S
```

## 역할과 Gate

| Agent | 입력 | 출력 | 통과 Gate |
|---|---|---|---|
| Reconstruction | `input.mp4`, camera/request metadata | `vision.json`, reconstruction report | 프레임 있음, 오른손 coverage 기준 통과 |
| Retargeting | `vision.json`, calibration, robot constraints | `canonical_trajectory.json`, `panda_trajectory.json` | 프레임 수 일치, invalid IK 0, joint limit 통과 |
| Physical Validation | canonical trajectory, explicit grasp/release, MuJoCo scene | `physics_validation.json`, `physics_rollout.mp4` | `passed=true`, `task_success=true`, collision-free, 목표 거리 기준 통과 |
| Data Scaling | 검증된 seed, augmentation/training request, provenance | `episodes/`, `dataset_manifest.json`, 선택적 RunPod 학습 artifact | 요청 episode 수·provenance·검증 결과 존재 |

앞 단계 Gate가 실패하면 다음 Agent에게 job을 넘기지 않는다. Validation
실패는 자동으로 Retargeting 수치를 바꾸는 근거가 아니다. 실패 artifact와
원인을 남기고, 입력 수정이 승인된 경우에만 새 시도를 만든다.

세부 역할 계약은 `reflex/agents/manifest.json`과 각 role prompt에 있다.

## Job ID·Artifact Handoff

목표 handoff 단위는 immutable artifact bundle이다.

```json
{
  "job_id": "video-001",
  "producer_role": "reconstruction",
  "gate_status": "passed",
  "artifacts": {"vision": {"uri": "...", "sha256": "..."}},
  "metrics": {},
  "attempt": 1
}
```

소비 Agent는 `job_id`, producer role, SHA-256를 검증한 뒤 전용 Devbox로
다운로드한다. 다른 job의 파일이나 `gate_status != passed`인 bundle을
사용하지 않는다. Secret은 bundle·prompt·log에 넣지 않는다.

> **현재 미구현:** Reflex role queue, Devbox 간 artifact store/upload/download,
> `handoff.json` SHA-256 검증, 4-session job router는 아직 연결되지 않았다.
> 현재 `src.robot_data_job`은 Reconstruction·Retargeting·Validation을 하나의
> process/Devbox에서 실행한다. 따라서 현재 데모를 “4-Agent live end-to-end”로
> 표현하지 않는다.

## Demo 2·DexYCB

Demo 2의 대상은 **DexYCB subject-07**의 right-hand
`006_mustard_bottle` sequence다. 3개를 요청했지만 현재 조건에 맞아
검증·선택된 sequence는 2개이므로, 2개만 학습 seed로 사용한다.

```text
subject-07 RGB (requested 3, verified 2)
→ Reconstruction
→ Franka Retargeting
→ MuJoCo Physical Validation
→ provenance를 보존한 Data Scaling
→ RunPod BC 학습/평가 artifact
```

DexYCB GT pose/depth를 trajectory 입력으로 사용했다고 표현하지 않는다.
현재 DexYCB pipeline의 trajectory는 RGB 2D hand mapping과 phase height로 만든
seed이며, carry/place/release는 generated segment로 표시한다. RunPod 학습은
Data Scaling Agent의 Devbox 안에서 하는 것이 아니라 해당 Agent가 별도 Pod
job을 요청하고 결과를 회수하는 구조다.

## 현재 시연 범위

- 구현됨: RGB 손 추출, 2D retargeting, Franka IK, MuJoCo 물리 검증,
  DexYCB subject-07의 검증된 2개 sequence 선별/hybrid seed, RunPod runner와
  개별 artifact. 3개 요청 대비 2개만 검증됐다는 selection provenance를 보존한다.
- 계약만 정의됨: 4개 role prompt, role manifest, job/gate/handoff schema.
- 미구현: 4개 Reflex session 생성 자동화, persistent worker queue, cross-Devbox
  artifact handoff, 전체 pipeline supervisor.

논문이 Agent 4개를 정의한 것은 아니다. 이 구조는 축소된 프로토타입을
제품 운영 역할로 분리한 데모 설계다.
