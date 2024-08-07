# [순재생, 역재생 비디오를 활용한 연구 완전지도 시간적 행동 검출 연구]

코드는 A6000 `/mnt/HDD2/HW/OpenTAD`에 위치하고 있습니다.

# 구동 방법
## 가상환경
가상 환경 설정은 [OpenTAD](https://github.com/sming256/OpenTAD/blob/main/docs/en/install.md) 공식 Github 사이트에 자세한 설명이 제공되어 있습니다.

## Data 준비
### THUMOS14 
- THUMOS14 데이터 셋은 다음 [공식 사이트](https://www.crcv.ucf.edu/THUMOS14/download.html)에서 다운 받을 수 있습니다.
- 또한 다음 [사이트](https://github.com/sming256/OpenTAD/blob/main/tools/prepare_data/thumos/README.md)를 참고해 주시기 바랍니다.
### 역재생 비디오 생성
- './tools/Reverse_Playback_Video_Generator.py'를 통해 역재생 비디오를 생성할 수 있습니다.
- 51 번줄에 순재생 비디오의 경로를 입력하고, 52 번줄에 역재생 비디오를 저장할 경로를 입력해 주시면 됩니다.
- 
## Backbone
- Backbone은 다음 [사이트](https://github.com/sming256/OpenTAD/tree/main/configs/adatad)에서 다운 받을 수 있습니다.
- 모든 실험은 VideoMAE-H 모델을 통해 진행했습니다.

## 주요 디렉토리
```
# Backbone
|-- pretrained
    |-- vit-huge-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth
# Data
|-- data
    |-- for_thumos-14
        |-- annotations
            |-- category_idx.txt
            |-- thumos_14_anno.json
        |-- raw_data
            |-- video_test_0000004.mp4
            |-- ....
            |-- ....
            ...
    |-- back_thumos-14
        |-- annotations
            |-- category_idx.txt
            |-- thumos_14_anno.json
        |-- raw_data
            |-- video_test_0000004.mp4
            |-- ....
            |-- ....
            ...
```
# 실험 구성
## 실험 1
1. 두 개의 네트워크 생성한다.
2. 두 개의 네트워크에 순재생 비디오와 역재생 비디오를 각각 입력하고, 마지막 Loss를 더한 후 Backward를 진행한다.

## 실험 1-1
1. 순재생 및 역재생 비디오를 통해 비디오를 통해 사전학습을 진행한다.
2. 두 개의 네트워크 생성 후 각각의 네트워크는 순재생 비디오를 사전학습한 모델, 역재생 비디오를 사전학습한 모델을 불러온다. 
3. 두 개의 네트워크에 순재생 비디오와 역재생 비디오를 각각 입력하고, 마지막 Loss를 더한 후 Backward를 진행한다.

## 실험 2
1. 실험 1의 단계 1, 단계 2까지 동일하게 진행한다.
2. 두 네트워크를 Freeze한 후 학습 가능한 모듈 삽입하고, 이 모듈만 학습이 가능하도록 한다.
3. 실험 1의 단계 3을 그대로 진행한다.

## 실험 3-1

1. 두 개의 네트워크 생성 후 각각 순재생 비디오, 역재생 비디오를 입력
2. 두 네트워크는 매 시점에 대한 Class score(cs), 액션의 Start time, End time까지의 거리 ds, de를 출력
- 순재생 비디오 기준
3. 두 네트워크의 cs, ds, de를 순재생 비디오 기준으로 정렬
4. 순재생 ds와 역재생 de , 역재생 de와 순재생 ds를 각각 위아래로 concat
5. (순재생 ds-역재생 de), (순재생 de-역재생 ds), cs를 순서로  옆으로 concat 후 FC layer 입력
- 역재생 비디오 기준
6. 두 네트워크의 cs, ds, de를 역재생 비디오 기준으로 정렬
7. 역재생 ds와 순재생 de , 순재생 de와 역재생 ds를 각각 위아래로 concat
8. (역재생 ds-순재생 de), (역재생 de-순재생 ds), cs를 순서로 양옆으로 concat 후 FC layer 입력 
9. 최종적으로 순재생 비디오와 역재생 비디오 기준의 cs, ds, de를 각각 출력한다.
10. 최종 Loss를 계산하고 Backward 진행

## 실험 3-2
1. 두 개의 네트워크 생성 후 각각 순재생 비디오, 역재생 비디오를 입력
2. 두 네트워크는 매 시점에 대한 Class score(cs), 액션의 Start time, End time까지의 거리 ds, de를 출력
- 순재생 비디오 기준
3. 두 네트워크가 출력한 cs, ds, de를 순재생 비디오 기준으로 정렬
4. 두 네트워크가 출력한 ds, de를 순재생 비디오 기준으로 정렬
5. 순재생 ds와 역재생 de , 역재생 de와 순재생 ds를 각각 위아래로 concat
6. 이전 단계에서 생성한 두개의 값을 양옆으로 concat 후 FC layer 입력
7. 순재생 비디오 기준으로 cs를 정렬하고 stack 후 3d conv 에 입력 
- 역재생 비디오 기준
7. 두 네트워크가 출력한 ds, de를 역재생 비디오 기준으로 정렬
8. 역재생 ds와 순재생 de , 순재생 de와 역재생 ds를 각각 위아래로 concat
9. 이전 단계에서 생성한 두개의 값을 양옆으로 concat 후 FC layer 입력
10. 역재생 비디오 기준으로 cs를 정렬하고 stack 후 3d conv 에 입력 
- 최종 출력
11. 최종적으로 순재생 비디오와 역재생 비디오 기준의 cs, ds, de를 각각 출력한다.
12. 최종 Loss를 계산하고 Backward 진행

## 실험 3-3
1. 두 개의 네트워크 생성 후 각각 순재생 비디오, 역재생 비디오를 입력
2. 두 네트워크는 매 시점에 대한 Class score(cs), 액션의 Start time, End time까지의 거리 ds, de를 출력
- 순재생 비디오 기준
3. 두 네트워크가 출력한 cs, ds, de를 순재생 비디오 기준으로 정렬
4. 두 네트워크가 출력한 ds, de를 순재생 비디오 기준으로 정렬
5. 순재생 ds와 역재생 de , 역재생 de와 순재생 ds를 각각 위아래로 concat
6. 이전 단계에서 생성한 두개의 값을 양옆으로 concat 후 FC layer 입력
7. 순재생 비디오 기준으로 cs를 정렬하고 평균을 낸다. 
- 역재생 비디오 기준
7. 두 네트워크가 출력한 ds, de를 역재생 비디오 기준으로 정렬
8. 역재생 ds와 순재생 de , 순재생 de와 역재생 ds를 각각 위아래로 concat
9. 이전 단계에서 생성한 두개의 값을 양옆으로 concat 후 FC layer 입력
10. 역재생 비디오 기준으로 cs를 정렬하고 평균을 낸다. 
- 최종 출력
11. 최종적으로 순재생 비디오와 역재생 비디오 기준의 cs, ds, de를 각각 출력한다.
12. 최종 Loss를 계산하고 Backward 진행



