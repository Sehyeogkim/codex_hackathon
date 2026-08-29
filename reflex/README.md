# Reflex 데모 연결

Reflex는 trajectory 계산기가 아니라 Runloop Devbox에서 이 저장소의
결정론적 파이프라인을 실행하고, 채팅·이벤트 기록·재실행을 제공하는
에이전트 화면으로 사용한다. Reflex 실행 안에서 별도의 Runloop Devbox를
다시 만들지는 않는다.

두 진입점은 같은 `config/demo_request.json` 계약을 사용한다.

- 제품 API/백엔드: `src.product_runloop`가 Runloop Devbox를 생성한다.
- Reflex Agent: Reflex가 Devbox를 생성하므로 `src.product_request`를 직접 실행한다.

## 준비

1. `https://reflex.runloop.ai`에서 조직을 만든다.
2. Runloop API key를 연결해 sandbox를 활성화한다.
3. 조직에 모델 provider를 연결한다.
4. CLI를 쓸 때만 Reflex API key와 organization ID를 발급한다.
5. Node.js 22 이상에서 CLI를 설치한다.

```bash
npm install -g @runloop/reflex-cli
export REFLEX_BASE_URL=https://reflex.runloop.ai
export REFLEX_API_KEY='...'
export REFLEX_ORG='...'
reflex-cli doctor
```

## 실행

현재 작업트리의 변경사항이 GitHub에 반영된 뒤 실행한다.

```bash
reflex-cli run \
  --type codex \
  --name robot-data-agent \
  --repo Sehyeogkim/codex_hackathon#main \
  --json \
  -p "$(cat reflex/robot_data_agent_prompt.md)"
```

실행 후 반환된 agent ID로 진행 상황을 다시 볼 수 있다.

```bash
reflex-cli watch <agent-id> --json
reflex-cli chat <agent-id>
reflex-cli open <agent-id>
```

Reflex 첨부 파일은 크기와 형식 제한이 있어 MP4를 직접 첨부하지 않는다.
데모 영상은 저장소에 포함하거나, 운영 환경에서는 짧게 유효한 다운로드
URL을 에이전트에 전달한다.

## 제품 API에서 직접 실행

```bash
export RUNLOOP_API_KEY='...'
python -m src.product_runloop \
  config/demo_request.json \
  --output-dir artifacts/runloop_demo
```

이 경로에서는 요청 검증이 먼저 끝난 뒤 Devbox가 만들어지고, 완료 후 결과를
다운로드한 다음 Devbox가 종료된다.
