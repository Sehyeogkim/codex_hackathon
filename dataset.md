# 공개 영상 데이터 전략

## 결론

공개 데이터는 기술 데모와 정확도 검증에는 쓸 수 있지만, 대부분 비상업적
라이선스이므로 고객 납품용 학습 데이터로 재판매하면 안 된다. 상업 제품에서는
직접 고용한 작업자에게 상업적 이용 동의를 받고 촬영한 데이터가 핵심 자산이다.

## 우선순위

1. **직접 촬영:** 고정 3인칭 카메라로 정확한 pick-and-place 전체 과정을 촬영한다.
2. **DexYCB:** 머스터드 병 pickup과 6D object pose를 이용해 변환 정확도를 검증한다.
3. **Assembly101:** 한 개의 고정 카메라 MP4부터 받아 대량 영상 처리 능력을 시험한다.
4. **Something-Something V2:** 22만 개 규모의 action 영상 확장성만 보여준다.

## 후보 비교

| 데이터셋 | 규모와 장점 | 한계 | 데모 용도 |
|---|---|---|---|
| [DexYCB](https://dex-ycb.github.io/) | 1,000 sequences, 8개 고정 RGB-D 카메라, 손 3D와 물체 6D pose, 머스터드 병 포함 | pickup 중심이며 place/release가 없음, 전체 약 119GB, CC BY-NC 4.0 | 정확성 검증 1순위 |
| [Assembly101](https://assembly-101.github.io/) | 4,321 videos, 513시간, 8개 고정+4개 1인칭 시점, 3D hand pose | 조립 작업 중심, object 6D pose 없음, 약 3.89TB, gated·비상업 | 대량 처리 smoke test |
| [Something-Something V2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset) | 220,847 clips, putting/moving/taking 동작 | 카메라 calibration·3D hand·object pose 없음, 연구용 | 규모 설명과 action pretraining |
| [H2O](https://h2odataset.ethz.ch/) | 5개 RGB-D 시점, grab/place/put-in, 손 3D와 물체 6D pose | 학술·비상업, 제3자 전송 제한 | 로컬 검증만 사용; Runloop 업로드 금지 |
| [ARCTIC](https://arctic.is.tue.mpg.de/) | 2.1M images, 8개 고정+1개 1인칭, 손·물체·접촉 GT | 양손 articulated-object 작업으로 현재 Franka pick-place와 불일치 | 현재 데모에서는 제외 |

## 다운로드 원칙

- 12GB 이상의 전체 데이터는 발표에 필요하지 않으므로 자동 다운로드하지 않는다.
- DexYCB는 subject 한 개, Assembly101은 recording 한 개와 fixed view 한 개부터 시작한다.
- gated 데이터의 약관 동의와 계정 인증은 데이터 소유자가 직접 수행한다.
- H2O 원본은 라이선스상 외부 Runloop Devbox에 업로드하지 않는다.
