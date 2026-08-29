# Reflex 데모 연결

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
`workflow_runloop.md`와 `reflex/agents/manifest.json`에 있다.

## 준비

1. `https://reflex.runloop.ai`에서 조직을 만든다.
2. Runloop API key를 연결해 sandbox를 활성화한다.
3. 조직에 모델 provider를 연결한다.
4. API 실행용 개인 Reflex API key를 발급해 `REFLEX_API_KEY`로 설정한다.

키 값은 prompt, manifest, Git, 로그에 넣지 않는다. 조직 slug는
`sehyeog-workspace-1`이며 비밀이 아닌 설정으로
`config/reflex_agents.json`에 기록돼 있다.

## 실행

먼저 네 Agent 정의를 검증한다. 이 명령은 Devbox를 만들지 않는다.

```bash
python -m src.reflex_api config/reflex_agents.json
```

GitHub 브랜치 반영과 조직의 OpenCode 모델 설정을 확인한 뒤 네 Session과
Devbox를 실제 생성한다.

```bash
export REFLEX_API_KEY='...'
python -m src.reflex_api config/reflex_agents.json --launch
```

생성 결과의 Agent ID는 Reflex 웹 Sessions 화면에서 확인할 수 있다. 조회와
중지는 각각 `--list`, `--get <agent-id>`, `--stop <agent-id>`를 사용한다.

Reflex 첨부 파일은 크기와 형식 제한이 있어 MP4를 직접 첨부하지 않는다.
데모 영상은 저장소에 포함하거나, 운영 환경에서는 짧게 유효한 다운로드
URL을 에이전트에 전달한다.

## 현재 구현 경계

`src.reflex_stages`에는 reconstruction, retargeting, validation, scaling의
독립 실행 명령이 구현돼 있다. Cross-Devbox artifact store, SHA handoff, role
queue/router는 아직 연결되지 않았으므로 연결 전에는 4-Agent live end-to-end
완료를 주장하지 않는다.
