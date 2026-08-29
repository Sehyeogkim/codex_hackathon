# 데모 2 데이터

## 바로 사용할 파일

### `egoverse_object_in_container_human_001.mp4`

- 태스크: 테이블의 작은 물체를 오른손으로 집어 투명 용기에 넣기
- 시점: 머리 장착형 1인칭(egocentric), `human_right_arm`
- 영상: 15초, 30 fps, 640×480, H.264, 약 1.5 MB
- 원본: EgoVerse episode `2026-01-07-12-23-52-588000`
- 원본 범위: 230.834초·6,925프레임 중 60–75초 구간
- 출처: [공식 탐색기](https://partners.mecka.ai/egoverse?task=object_in_container), [원본 영상 API](https://partners.mecka.ai/api/egoverse/uploads/2026-01-07-12-23-52-588000/video?redirect=1)
- 라이선스: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) — 출처 표시와 동일 조건 공유 필요
- 무결성(SHA-256): `58873536a96d956bd8739810f8d42be45e544932d3d695fe4efdffea8b7ef6f2`
- 용도 판단: 영상→손/물체 reconstruction→Franka retargeting의 **파이프라인 데모용으로 적합**. 한 클립뿐이므로 정책 학습용으로는 부족함.

이 파일은 위 EgoVerse 원본에서 15초 구간을 잘라 만든 2차 저작물입니다. EgoVerse 저자·기여자에게 출처를 표시해야 합니다.

### `do_as_i_do_pick_place_preview.mp4`

- 태스크: 작은 물체를 집어 용기에 넣는 pick-and-place
- 시점: 머리 장착형 1인칭, 양손이 보이는 짧은 미리보기
- 출처: [Do as I Do 공식 프로젝트 영상](https://do-as-i-do.com/assets/highlights/23963/input_video.mp4)
- 무결성(SHA-256): `7ccb44d5e0115f61a401cc79466ed83703541ab9be65ace93d766fe9b58ff4d8`
- 라이선스: 프로젝트 논문은 CC BY 4.0이고 코드 저장소는 MIT이지만, 이 개별 웹 영상의 재배포 조건은 명시적으로 확인되지 않음
- 용도 판단: 로컬 파이프라인 smoke test와 발표 중 출처를 표시한 미리보기에만 사용하고, 학습 데이터 배포에는 포함하지 않는다.

## 데이터셋 검토 결과

| 데이터셋 | 적합성 | 접근/라이선스 | 결론 |
|---|---|---|---|
| [EgoVerse](https://egoverse.ai/) | `object_in_container`가 정확히 일치하며 1인칭 손 영상 제공 | 에피소드별 미리보기·다운로드 가능, 해당 에피소드 CC BY-SA 4.0 | **현재 데모에 채택** |
| [EgoDex](https://github.com/apple/ml-egodex) | `basic_pick_place`와 3D 손 pose가 있어 가장 좋은 확장 데이터 | 학습 Part 2가 300 GB 단위, CC BY-NC-ND | 오늘은 소량 추출 불가. 이후 Part 2를 RunPod에서 받아 사용 |
| [HOI4D](https://hoi4d.github.io/) | 1인칭 RGB-D·3D hand/object pose 제공 | CC BY-NC 4.0, 공식 배포가 전체 묶음 중심 | 단일 pick-place 샘플을 빠르게 고르기 어려워 보류 |
| [OakInk2](https://github.com/oakink/OakInk2) | 정밀 hand/object pose지만 복잡한 양손·다중 시점 태스크 중심 | 데이터 신청/대용량 다운로드 필요 | 이번 단일 팔 데모에는 과함 |

## Do As I Do 샘플 확인

[공식 저장소](https://github.com/malik-group/do-as-i-do)의 현재 공개 입력 파일은 `reconstruction/whisking/whisking.mp4`뿐입니다. 별도로 공식 프로젝트 웹사이트의 pick-and-place 미리보기를 로컬 테스트용으로 저장했습니다.

## 학습 사용 판단

- 현재 파일: 알고리즘 연결과 화면 데모 검증용
- 최소 배치 데모: 동일 태스크의 서로 다른 작업자 영상 10–20개
- 실제 정책 학습: 수백~수천 episode와 소량의 실제 Franka 데이터 필요
