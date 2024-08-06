# [PBVS @ CVPR 2024 Thermal Images Super-resolution challenge- Track 1]

이 방법을 통해 [TISR 챌린지](https://codalab.lisn.upsaclay.fr/competitions/17013#results)에서 PSNR 4위, SSIM 3위를 기록하였습니다. 

코드는 [Github]

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

# 주요 코드
## 사전학습 단계의 전처리
- 파인튜닝 때와 모델 구조를 동일하게 통일해 주기 위해 수정하였습니다.
- LR(Low Resolution) image를 x8 업스케일하고 사전학습을 진행합니다.
- 코드는 `./hat/archs/not_aug_build_SR_model.py` 의 102 ~ 104 번 줄에서 확인 가능합니다.
```
    if self.gt.size() != self.lq.size():
            scale = self.gt.size(2) // self.lq.size(2)                           # 몇 배 업스케일 할지 결정
            self.lq = F.interpolate(self.lq, scale_factor=scale, mode="nearest") # 업스케일 진행
```
## 파인튜닝 단계에서의 Augmentation 적용
- LR(Low Resolution) image x8 업스케일을 우선 진행합니다.
- 이후 선택된 Augmentation 방식을 적용하고 파인 튜닝을 진행합니다.
- 코드는 `./hat/archs/aug_build_SR_model.py` 의 104 ~ 110 번 줄에서 확인 가능합니다.
```
        if self.gt.size() != self.lq.size():
            scale = self.gt.size(2) // self.lq.size(2)                            # 몇 배 업스케일 할지 결정
            self.lq = F.interpolate(self.lq, scale_factor=scale, mode="nearest")  # 업스케일 진행

        self.gt, self.lq, mask, aug = augments.apply_augment(                     # 선택된 Augmentation 방식 적용
            self.gt, self.lq,
            self.opt['augs'], self.opt['prob'], self.opt['alpha'],
            self.opt['aux_prob'], self.opt['aux_alpha']
        )
```
## Test-time Augmentation 방식
- rotation, flip을 적용하여 8개의 이미지를 생성하고 이를 순차적으로 모델에 입력해 줍니다.
- 이후 출력된 SR이미지의 방향을 일치시키고 같은 위치의 pixel 값끼리 평균을 냅니다.
- 코드는 `./hat/models/TDA_hat_model.py`의 141 ~ 210 번 줄에서 확인 가능합니다.
  
```
#################################################################################### test time augmentation start
# f rotation, flip을 적용해 새로운 LR 이미지 생성

            lr_image_dic = {}

            sr_image_dic = {}

            lr_image_dic['img_0'] = val_data['lq']
            lr_image_dic['img_90'] = torch.rot90(val_data['lq'], 1, [2, 3])
            lr_image_dic['img_180'] = FF.rotate(val_data['lq'], 180)
            lr_image_dic['img_270'] = torch.rot90(val_data['lq'], 3, [2, 3])


            lr_image_dic['img_lr_flip'] = FF.hflip(val_data['lq'])
            lr_image_dic['img_lr_flip_90'] = torch.rot90(lr_image_dic['img_lr_flip'], 1, [2, 3])
            lr_image_dic['img_lr_flip_180'] = FF.rotate(lr_image_dic['img_lr_flip'], 180)
            lr_image_dic['img_lr_flip_270'] = torch.rot90(lr_image_dic['img_lr_flip'], 3, [2, 3])

            ##################################################################################### for start
            # 순차적인 LR 이미지 모델 입력 및 SR 이미지 생성

            for image_name, lr_image in lr_image_dic.items():

                self.feed_data(lr_image)

                self.pre_process()

                if 'tile' in self.opt:
                    self.tile_process()
                else:
                    self.process()
                self.post_process()

                #################################################### save in dict start
                # SR 이미지를 사전에 저장
                visuals = self.get_current_visuals()
                sr_image_dic[image_name] = visuals

                #################################################### save in dict end

            #################################################################################### for end
            
            ######################################################################### Orientation restoration start
            # 생성된 SR 이미지의 방향 통일
            sr_image_dic['img_0'] = sr_image_dic['img_0']['result']
            sr_image_dic['img_90'] = torch.rot90(sr_image_dic['img_90']['result'], 3, [2, 3])
            sr_image_dic['img_180'] = FF.rotate(sr_image_dic['img_180']['result'], 180)
            sr_image_dic['img_270'] = torch.rot90(sr_image_dic['img_270']['result'], 1, [2, 3])


            sr_image_dic['img_lr_flip'] = FF.hflip(sr_image_dic['img_lr_flip']['result'])
            sr_image_dic['img_lr_flip_90'] = FF.hflip(torch.rot90(sr_image_dic['img_lr_flip_90']['result'], 3, [2, 3]))
            sr_image_dic['img_lr_flip_180'] = FF.hflip(FF.rotate(sr_image_dic['img_lr_flip_180']['result'], 180))
            sr_image_dic['img_lr_flip_270'] = FF.hflip(torch.rot90(sr_image_dic['img_lr_flip_270']['result'], 1, [2, 3]))
            ######################################################################### Orientation restoration end

            # 같은 위치의 pixel 값끼리 평균을 냄
            one_channel_result = torch.cat(list(sr_image_dic.values()), dim=1).mean(dim = 1)

            result = torch.cat((one_channel_result,one_channel_result, one_channel_result), dim=0)

            sr_img = tensor2img(result)

#################################################################################### test time augmentation end
```

# 실험 결과
## Optimizer에 따른 PSNR, SSIM 비교
|     Optimizer    |        PSNR      |        SSIM       |
|:----------------:|:----------------:|:-----------------:|
|        Adam      | **    27.00   ** |       0.8233      |
|       AdamW      |       26.99      | **    0.8234   ** |

## Attention 방식에 따른 PSNR, SSIM 비교
- RCA : 정규 분포를 가지는 벡터 v = (v_1, v_2, v_3)와 이미지를 혼합합니다.
- RCSA : 사각 형태의 영역을 무작위로 선택하고, 해당 영역에 다른 이미지를 잘라 넣습니다.
- RCA + CSA: 서로 같은 두 이미지에 대해 Cutmix[3]를 진행한 다. 이때 두 이미지의 해상도는 서로 다르며 저해상도의 이미지는 미리 스케일 업하여 고해상도 이미지와 동일한 사이즈를 가지도록 합니다.

|Method|PSNR|SSIM|
|:----:|:----:|:----:|
|RCA|**27.00**|**0.8233**|
|RCSA|26.98|0.8229|
|RCA + CSA|26.96|0.8228|

## Augmentation 방법에 따른 PSNR, SSIM 비교
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

## Test-time Data Augmentation 방식 적용 결과
- 적용전은 Cutblur를 적용하여 모델을 학습시킨 결과 입니다.

| TDA | PSNR |  SSIM  |
|:-----:|:-----:|:------:|
|  x  | 27.08 | 0.8261 |
|  √  | **27.19** | **0.8284** |

## 사전학습에 따른 실험 결과
- DF2K 데이터 셋을 통한 사전학습을 진행합니다.
- 사전학습을 진행할 경우 TISR challenge의 validation set을 통해 평가를 진행합니다.
- TISR challenge의 데이터 셋을 통한 파인 튜닝을 진행합니다.
- 사전학습을 진행하면서 특정 iter의 사전학습을 진행한 모델을 가져와 파인튜닝을 진행하였고, 파인튜닝을 진행할 경우 PSNR이 하강할 때 학습을 종료하였습니다.

<table class="tg"><thead>
  <tr>
    <th class="tg-c3ow" colspan="3">  사전학습</th>
    <th class="tg-c3ow" colspan="2">파인튜닝 </th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-c3ow">  Iteration </td>
    <td class="tg-c3ow">   PSNR   </td>
    <td class="tg-c3ow">   SSIM   </td>
    <td class="tg-c3ow">   PSNR   </td>
    <td class="tg-c3ow">   SSIM   </td>
  </tr>
  <tr>
    <td class="tg-c3ow">   x   </td>
    <td class="tg-c3ow">   x   </td>
    <td class="tg-c3ow">   x   </td>
    <td class="tg-c3ow">   27.19   </td>
    <td class="tg-c3ow">   0.8284   </td>
  </tr>
  <tr>
    <td class="tg-c3ow">   50000iter   </td>
    <td class="tg-c3ow">   26.30   </td>
    <td class="tg-c3ow">   0.8229   </td>
    <td class="tg-c3ow">   27.28   </td>
    <td class="tg-c3ow">   0.8305   </td>
  </tr>
  <tr>
    <td class="tg-c3ow">   124000iter   </td>
    <td class="tg-c3ow">   26.53   </td>
    <td class="tg-c3ow">   0.8302   </td>
    <td class="tg-c3ow">   27.32   </td>
    <td class="tg-c3ow">   0.8317   </td>
  </tr>
  <tr>
    <td class="tg-c3ow">   146000iter   </td>
    <td class="tg-7btt">   26.57   </td>
    <td class="tg-7btt">   0.8314   </td>
    <td class="tg-c3ow">   27.32    </td>
    <td class="tg-c3ow">   0.8318   </td>
  </tr>
  <tr>
    <td class="tg-c3ow">   191000iter   </td>
    <td class="tg-c3ow">   26.49   </td>
    <td class="tg-c3ow">   0.8308   </td>
    <td class="tg-7btt">   27.34    </td>
    <td class="tg-7btt">   0.8322    </td>
  </tr>
</tbody></table>


