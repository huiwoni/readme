# [PBVS @ CVPR 2024 Thermal Images Super-resolution challenge- Track 1]

이 방법을 통해 [TISR 챌린지](https://codalab.lisn.upsaclay.fr/competitions/17013#results)에서 PSNR 4위, SSIM 3위를 기록하였습니다. 
# 구동 방법
## 가상환경
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

sub-images 추출
```
python df2k_extract_subimages.py
```
meta-info file 생성(데이터 정보)
```
python df2k_generate_meta_info.py
```

### 파인 튜닝을 위한 적외선 이미지 데이터 셋
- 적외선 이미지는 다음 [페이지](https://codalab.lisn.upsaclay.fr/competitions/17013#learn_the_details)에서 다운받을 수 있습니다.

sub-images 추출
```
python thermal_extract_subimages.py
```
meta-info file 생성(데이터 정보)
```
python thermal_generate_meta_info.py
```
## Quick[test]
- Refer to `./options/test`
- 적외선 이미지를 통해 파인튜닝까지 완료한 모델입니다.
- 학습된 모델은 [구글 드라이브](https://drive.google.com/drive/folders/1UFVLyONwlqJpWE6hEw7Kqqxw2GdBo43m?usp=sharing)에 저장해 두었습니다.
- 위 사이트에서 다운받은 학습된 모델을 다음 경로에 저장하시면 됩니다. ./experiments/pretrained/. (가장 높은 성능은 14,000 iterations에서 달성하였습니다.)

SR images 이미지 생성
```
python hat/test.py -opt options/test/HAT_SRx8_quick.yml
```
생성된 SR images는 `./results` 폴더에 저장됩니다..

## 사전학습
- 다음 경로를 참고해 주시길 바랍니다. `./options/train`
사전학습을 위한 명령어
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 hat/train.py -opt options/train/train_HAT_thermalSRx8_pre.yml --launcher pytorch
```
The training logs and weights will be saved in the `./experiments` folder.

## 파인튜닝
- 다음 경로를 참고해 주시길 바랍니다. `./options/train`
- 필요하시다면 DF2K 데이터 셋을 통해 사전학습된 모델은 [구글 드라이브](https://drive.google.com/drive/folders/1UFVLyONwlqJpWE6hEw7Kqqxw2GdBo43m?usp=sharing)에 저장되어 있습니다.
- 다음 경로에 저장해 주시면 됩니다. ./experiments/pretrained/. (191,000 iterations까지 RGB이미지에 대해 학습시킨 모델 입니다.)
파인튜닝을 위한 명령어
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 hat/train.py -opt options/train/train_HAT_thermalSRx8_48_cutblur_fineturn.yml --launcher pytorch
```
log와 weight 파일은 다음 경로에 저장됩니다. `./experiments` folder.
