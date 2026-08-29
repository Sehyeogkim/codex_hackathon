# RunPod 상태

- 브라우저 로그인: 확인됨
- 사용 가능한 크레딧: 확인됨
- 실행 중인 Pod: 없음
- 로컬 CLI/API 연결: 없음

## 사용 계획

- 일반 학습: RunPod PyTorch 템플릿
- Isaac Sim 렌더링: RT Core가 있는 RTX A6000, RTX 6000 Ada 또는 L40S 사용
- A100/H100은 Isaac Sim 렌더링에 사용하지 않는다.
- Isaac Sim은 headless로 실행하고 결과를 MP4로 저장한다.

Pod 생성부터 비용이 발생하므로 실제 배포 직전에 GPU와 예상 실행 시간을 확인한다.
