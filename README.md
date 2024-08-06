# [PBVS @ CVPR 2024 Thermal Images Super-resolution challenge- Track 1]

이 방법을 통해 [TISR 챌린지](https://codalab.lisn.upsaclay.fr/competitions/17013#results)에서 PSNR 4위, SSIM 3위를 기록하였습니다. 
# 구동 방법
##     가상환경
- Python == 3.8.0
- [PyTorch == 1.11.0](https://pytorch.kr/get-started/previous-versions/)
- BasicSR == 1.3.4.9
- eninops

```
python setup.py develop
```
## Data 준비
### 사전학습을 위한 DF2K 데이터 셋 
- DF2K 데이터 셋 다운로드는 다음 [페이지](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md)를 참조하시기 바랍니다.

- sub-images 추출
```
python df2k_extract_subimages.py
```
- meta-info file 생성(데이터 정보)
```
python df2k_generate_meta_info.py
```

### 파인 튜닝을 위한 적외선 이미지 데이터 셋
- 적외선 이미지는 다음 [페이지](https://codalab.lisn.upsaclay.fr/competitions/17013#learn_the_details)에서 다운받을 수 있습니다.

- sub-images 추출
```
python thermal_extract_subimages.py
```
- meta-info file 생성(데이터 정보)
```
python thermal_generate_meta_info.py
```
## Quick[test]
- Refer to `./options/test`
- 적외선 이미지를 통해 파인튜닝까지 완료한 모델입니다.
- 학습된 모델은 [구글 드라이브](https://drive.google.com/drive/folders/1UFVLyONwlqJpWE6hEw7Kqqxw2GdBo43m?usp=sharing)에 저장해 두었습니다.
- 위 사이트에서 다운받은 학습된 모델을 다음 경로에 저장하시면 됩니다. ./experiments/pretrained/. (가장 높은 성능은 14,000 iterations에서 달성하였습니다.)

- SR images 이미지 생성
```
python hat/test.py -opt options/test/HAT_SRx8_quick.yml
```
- 생성된 SR images는 `./results` 폴더에 저장됩니다.

## 사전학습
- 다음 경로를 참고해 주시길 바랍니다. `./options/train`

- 사전학습을 위한 명령어
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 hat/train.py -opt options/train/train_HAT_thermalSRx8_pre.yml --launcher pytorch
```
- log와 weight 파일은 다음 경로에 저장됩니다. `./experiments`.

## 파인튜닝
- 다음 경로를 참고해 주시길 바랍니다. `./options/train`
- 필요하시다면 DF2K 데이터 셋을 통해 사전학습된 모델은 [구글 드라이브](https://drive.google.com/drive/folders/1UFVLyONwlqJpWE6hEw7Kqqxw2GdBo43m?usp=sharing)에 저장되어 있습니다.
- 다음 경로에 저장해 주시면 됩니다. ./experiments/pretrained/. (191,000 iterations까지 RGB이미지에 대해 학습시킨 모델 입니다.)

- 파인튜닝을 위한 명령어
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 hat/train.py -opt options/train/train_HAT_thermalSRx8_48_cutblur_fineturn.yml --launcher pytorch
```
- log와 weight 파일은 다음 경로에 저장됩니다. `./experiments`.

# 실험 결과 
## Augmentation 방법에 따른 실험 결과
- Blend : 정규 분포를 가지는 벡터 v = (v_1, v_2, v_3)와 이미지를 혼합합니다.
- CutMix : 사각 형태의 영역을 무작위로 선택하고, 해당 영역에 다른 이미지를 잘라 넣습니다.
- CutBlur : 서로 같은 두 이미지에 대해 Cutmix[3]를 진행한 다. 이때 두 이미지의 해상도는 서로 다르며 저해상도의 이미지는 미리 스케일 업하여 고해상도 이미지와 동일한 사이즈를 가지도록 합니다.
- Cutout : 일정 확률로 이미지 픽셀의 일부를 제거하며 Cutout [2]된 픽셀은 손실함수에 영향을 미치지 못하도록 합니다.
- Mixup : 두 개의 이미지를 무작위로 선택하고, 선택된 두 이미지를 혼합합니다.
- CutMixup : Mixup과 CutMix의 혼합으로 두개의 이미지를 선택해 사각 형태의 영역을 무작위로 선택하고, 해당 영역에서 Mixup을 진행합니다.
- Mixture of Augmentations(MoA) : 위에 제시된 데이터 증강방식 중 한 개를 무작위로 선택합니다.

|Method|PSNR|SSIM|
|:----:|:----:|:----:|
|HAT|27.000|0.8233|
|Blend|27.97|0.8223|
|Cutout|27.01|0.8241|
|CutMix|26.97|0.8242|
|Mixup|27.07|0.8251|
|CutMixup|27.04|0.8246|
|CutBlur|**27.08**|**0.8261**|
|MoA|26.96|0.8241|

## Attention 방식에 따른 실험 결과

|Method|PSNR|SSIM|
|:----:|:----:|:----:|
|RCA|27.00|0.8233|
|RCSA|26.98|0.8229|
|RCA + CSA|26.96|0.8228|

