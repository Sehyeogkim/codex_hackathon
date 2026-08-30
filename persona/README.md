# Reflex Agent Personas

Reflex에는 역할이 고정된 Agent Session 4개를 만든다. 각 Session은 Reflex가
제공하는 전용 Runloop Devbox 하나에서 실행된다.

```text
Reconstruction Agent / Devbox R
Retargeting Agent / Devbox T
Physical Validation Agent / Devbox V
Data Scaling Agent / Devbox S
```

영상마다 Agent를 새로 만들지 않고 네 Session을 `job_id`별로 재사용한다.
Agent 안에서 nested Runloop Devbox를 만들지 않는다. 역할·Gate의 상세 계약은
[`../document/workflow_runloop.md`](../document/workflow_runloop.md)와
[`manifest.json`](manifest.json)에 있다.

## Persona 파일

- [`robot_data_agent_prompt.md`](robot_data_agent_prompt.md): 전체 제품 요청 실행용 Agent
- [`agents/reconstruction.prompt.md`](agents/reconstruction.prompt.md): RGB reconstruction
- [`agents/retargeting.prompt.md`](agents/retargeting.prompt.md): Franka retargeting·IK
- [`agents/physical_validation.prompt.md`](agents/physical_validation.prompt.md): MuJoCo 검증
- [`agents/data_scaling.prompt.md`](agents/data_scaling.prompt.md): 검증 데이터 확장·RunPod 감사
- [`manifest.json`](manifest.json): 역할, 입력·출력 artifact, Gate 계약
- [`reflex_agents.json`](reflex_agents.json): Reflex API Session 생성 설정
- [`reflex_api.py`](reflex_api.py): secret-safe Session REST client와 dry-run CLI
- [`reflex_stages.py`](reflex_stages.py): 독립 reconstruction·retargeting·validation·scaling CLI

Persona 정의, 실행 코드, launch 설정은 모두 이 디렉터리 안에서 경로가 완결된다.
과거 `src/reflex_*.py`, `config/reflex_agents.json`, `reflex/` 경로는 제거됐다.

## 준비

1. `https://reflex.runloop.ai`에서 조직을 만든다.
2. Runloop API key를 연결해 sandbox를 활성화한다.
3. 조직에 모델 provider를 연결한다.
4. API 실행용 개인 Reflex API key를 발급해 `REFLEX_API_KEY`로 설정한다.

키 값은 prompt, manifest, Git, 로그에 넣지 않는다. 조직 slug는
`sehyeog-workspace-1`이며 비밀이 아닌 설정으로
`reflex_agents.json`에 기록돼 있다.

## 실행

먼저 네 Agent 정의를 검증한다. 이 명령은 Devbox를 만들지 않는다.

```bash
python -m persona.reflex_api
```

GitHub 브랜치 반영과 조직의 OpenCode 모델 설정을 확인한 뒤 네 Session과
Devbox를 실제 생성한다.

```bash
export REFLEX_API_KEY='...'
python -m persona.reflex_api --launch
```

생성 결과의 Agent ID는 Reflex 웹 Sessions 화면에서 확인할 수 있다. 조회와
중지는 각각 `--list`, `--get <agent-id>`, `--stop <agent-id>`를 사용한다.

각 역할 CLI는 독립적으로 확인할 수 있다.

```bash
python -m persona.reflex_stages reconstruction --help
python -m persona.reflex_stages retargeting --help
python -m persona.reflex_stages validation --help
python -m persona.reflex_stages scaling --help
```

Reflex 첨부 파일은 크기와 형식 제한이 있어 MP4를 직접 첨부하지 않는다.
데모 영상은 저장소에 포함하거나, 운영 환경에서는 짧게 유효한 다운로드
URL을 에이전트에 전달한다.

## 현재 구현 경계

`persona.reflex_stages`에는 reconstruction, retargeting, validation, scaling의
독립 실행 명령이 구현돼 있다. Cross-Devbox artifact store, SHA handoff, role
queue/router는 아직 연결되지 않았으므로 연결 전에는 4-Agent live end-to-end
완료를 주장하지 않는다.

현재 네 Reflex Session은 실제로 생성되지 않았다. `python -m persona.reflex_api`는
manifest를 검증하는 dry-run일 뿐이며, `--launch`를 명시하고 유효한
`REFLEX_API_KEY`를 제공하기 전에는 Session이나 Devbox를 만들지 않는다.
