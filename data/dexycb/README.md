# DexYCB 사용 계획

## 대상

- 물체: YCB mustard bottle
- 태스크: pick-up 후 지정 위치에서 hold/handover
- 선택: `meta.yml` 기준 `ycb_ids[ycb_grasp_ind] == 5`, `mano_sides[0] == right`
- 순서: sequence timestamp 오름차순 처음 3개
- 시점: 8개 고정 카메라의 오른손 검출률을 비교해 자동 선택

## 데이터 특성

- 8개 고정 외부 카메라
- RGB-D 30fps, 640×480
- 21개 손 관절과 MANO pose
- 물체 6D pose
- 라이선스: CC BY-NC 4.0

## 다운로드

공식 배포의 최소 단위는 subject archive 약 12GB다. RunPod volume에 다운로드한 뒤 mustard bottle 시퀀스만 추출한다.

- [공식 사이트](https://dex-ycb.github.io/)
- [subject-07 archive](https://drive.google.com/file/d/1oWEYD_o3PVh39pLzMlJcArkDtMj4nzI0/edit)

대용량 원본은 Git 저장소에 포함하지 않는다.

```bash
scripts/download_dexycb.sh /workspace/dexycb
tar -xzf /workspace/dexycb/subject-07.tar.gz -C /workspace/dexycb
python -m src.dexycb_pipeline /workspace/dexycb \
  --output-dir /workspace/dexycb-prepared --limit 3
```

`src.dexycb_pipeline` 은 각 카메라의 JPEG을 MP4로 변환하고 MediaPipe
오른손 검출 coverage가 가장 높은 시점을 선택한다. 라이브러리
API에 `trajectory_callback`을 주거나 기본 CLI를 실행하면 RGB로 추출한
human pickup과 생성한 carry/place/release를 합친 `hybrid_trajectory.json`도
저장한다. 각 프레임은
`human_segment` 또는 `generated_segment`로 표시된다.

기본 CLI의 pickup은 MediaPipe palm/pinch와 `config/demo_config.json` homography만
사용한다. 오른손 coverage 70% 미만이거나 pinch가 3프레임 연속으로
임계값을 통과하지 않으면 실패한다. 필요하면 `--grasp-frame`으로 RGB
프레임을 지정할 수 있다. `--camera-only`는 카메라 선택만 실행하는 진단용
옵션이다.

현재 한계: 기본 homography는 DexYCB 카메라의 metric calibration이 아니라
normalized 2D hand UV를 Franka 작업공간에 대응시키는 데모용 mapping이다. RGB에서
깊이를 알 수 없으므로 z는 `table_z`/`lift_z`와 phase로 생성한다. 따라서
현재 결과는 metric human-motion reconstruction이 아니라 RGB-derived robot seed
trajectory이다.

DexYCB GT pose는 sequence 선택과 평가에만 사용하고 trajectory 입력으로
사용하지 않는다. CC BY-NC 4.0이므로 상업용 학습 데이터로 사용하지
않는다.
