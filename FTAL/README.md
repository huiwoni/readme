# [PBVS @ CVPR 2024 Thermal Images Super-resolution challenge- Track 1]

이 방법을 통해 [TISR 챌린지](https://codalab.lisn.upsaclay.fr/competitions/17013#results)에서 PSNR 4위, SSIM 3위를 기록하였습니다 [1]. 

코드는 [Github](https://github.com/huiwoni/TISR-Challenge/tree/main)에서 이용 가능합니다.

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
- DF2K(DIV2K[10] +Flicker2K[11])) 데이터 셋 다운로드는 다음 [페이지](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md)를 참조하시기 바랍니다.

- sub-images 추출
```
python df2k_extract_subimages.py
```
- meta-info file 생성(데이터 정보)
```
python df2k_generate_meta_info.py
```

### 파인 튜닝을 위한 TISR challenge 데이터 셋[9]
- TISR challenge 데이터 셋[9]은 다음 [페이지](https://codalab.lisn.upsaclay.fr/competitions/17013#learn_the_details)에서 다운받을 수 있습니다.

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
# rotation, flip을 적용해 새로운 LR 이미지 생성

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
|        Adam      |     **27.00**    |       0.8233      |
|       AdamW      |       26.99      |     **0.8234**    |

## Attention 방식에 따른 PSNR, SSIM 비교
|Method|PSNR|SSIM|
|:----:|:----:|:----:|
|RCA[2]|**27.00**|**0.8233**|
|RCSA[3]|26.98|0.8229|
|RCA[2] + CSA[4]|26.96|0.8228|

## Augmentation 방법에 따른 PSNR, SSIM 비교
|Method|PSNR|SSIM|
|:----:|:----:|:----:|
|HAT[2]|27.000|0.8233|
|Cutout[5]|27.01|0.8241|
|CutMix[6]|26.97|0.8242|
|Mixup[7]|27.07|0.8251|
|Blend|27.97|0.8223|
|CutMixup[8]|27.04|0.8246|
|CutBlur[8]|**27.08**|**0.8261**|
|MoA[8]|26.96|0.8241|

## Test-time Data Augmentation 방식 적용 결과
- 적용전은 Cutblur를 적용하여 모델을 학습시킨 결과 입니다.

| TDA | PSNR |  SSIM  |
|:-----:|:-----:|:------:|
|  x  | 27.08 | 0.8261 |
|  √  | **27.19** | **0.8284** |

## 사전학습 진행에 따른 실험 결과
- DF2K 데이터 셋을 통한 사전학습을 진행합니다.
- 사전학습을 진행할 경우 TISR challenge[9]의 validation set을 통해 평가를 진행합니다.
- TISR challenge[9]의 데이터 셋을 통한 파인 튜닝을 진행합니다.
- 특정 iter의 사전학습을 진행한 모델을 가져와 파인튜닝을 진행하였고, 파인튜닝을 진행할 경우 PSNR이 하강할 때 학습을 종료하였습니다.

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

# Reference
[1] Rafael E. Rivadeneira, Angel D. Sappa, Chenyang Wang, Junjun Jiang, Zhiwei Zhong, Peilin Chen and Shiqi Wang, "Thermal Image Super-Resolution Challenge Results - PBVS 2024," In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pp.3113--3122, 2024.

[2] Xiangyu Chen, Xintao Wang, Jiantao Zhou, and Chao Dong. "Activating more pixels in image super-resolution transformer," In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp.22367-22377, 2023.

[3] Rafael E. Rivadeneira, Angel D. Sappa, Boris X. Vintimilla, Dai Bin, Li Ruodi, Li Shengye, Zhiwei Zhong, Xianming Liu, Junjun Jiang and Chenyang Wang, "Thermal Image Super-Resolution Challenge Results - PBVS 2023," In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pp.470--478, 2023.

[4] Ben Niu, Weilei Wen, Wenqi Ren, Xiangde Zhang, Lianping Yang, Shuzhen Wang, Kaihao Zhang, Xiaochun Cao, and Haifeng Shen, "Single image super-resolution via a holistic attention network," In European conference on computer vision, pp. 191-–207, 2020.

[5] Terrance DeVries and Graham W Taylor, "Improved regularization of convolutional neural networks with cutout," arXiv preprint arXiv:1708.04552, 2017.

[6] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo, "Cutmix: Regularization strategy to train strong classifiers with localizable features," arXiv preprint arXiv:1905.04899, 2019.

[7] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz, "Mixup: Beyond empirical risk minimization," arXiv preprint arXiv:1710.09412, 2017. 

[8] Jaejun Yoo, Namhyuk Ahn and Kyung-Ah Sohn, "Rethinking Data Augmentation for Image Super-resolution: A Comprehensive Analysis and a New Strategy," In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp.8375--8384, 2020.

[9] Rafael E Rivadeneira, Angel D Sappa, and Boris X Vintimilla, "Thermal image super-resolution: A novel architecture and dataset," In Proc. of the International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications, pages 111-–119, 2020.

[10] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah and Kyoung Mu Lee, "Enhanced deep residual networks for single image super-resolution," In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pages 136–-144, 2017. 

[11] Radu Timofte, Eirikur Agustsson, Luc Van Gool, MingHsuan Yang, and Lei Zhang. "Ntire 2017 challenge on single image super-resolution: Methods and results," In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pages 114–-125, 2017.
