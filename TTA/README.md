# [Domain Adapter and Inaccurate Pseudo-label Removal Filter for Test-Time Adaptation: A Detailed Analysis]

코드는 A6000의 `/mnt/HDD2/HW/Benchmark-TTA`에 위치합니다.

# 구동

## 가상환경

```bash
conda update conda
conda env create -f environment.yaml
conda activate vizwiz_TTA
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Data 준비
- 데이터 셋 구조는 아래와 같으며 DomainNet126 데이터 셋은 다음 [링크](https://ai.bu.edu/M3SDA/)에서 다운 받을 수 있습니다.

```
|-- datasets
    |-- DomainNet
        |-- clipart
        |-- painting
        |-- real
        |-- sketch
```
## Source 모델 학습
- `train_source_resnet.py`를 통해 DomainNet-126 데이터 셋의 각 도메인의 데이터를 학습한 사전학습 모델을 생성할 수 있습니다. 
- 265 번 줄의 clipart, painting, real, sketch 중 1개를 입력하여 해당 도메인에 속하는 데이터를 학습시킬 수 있습니다.
- 또는 다음 [링크](https://drive.google.com/drive/folders/1z9YosBHLTxKj8qeWkaDeMS6YFFAwnE_W)에서 사전학습 모델을 다운받을 수 있습니다.
- 사전학습 모델을 아래와 같이 위치 시켜야 합니다.
```
|-- ckpt
    |-- models
        |-- domainnet126
            |-- clipart
                |-- model.pt
            |-- painting
                |-- model.pt
            |-- real
                |-- model.pt
            |-- sketch
                |-- model.pt

```
## Adatation 진행
- `best_cfgs` 내부 yaml 파일 내부 내용에 따라 실험 방법을 달리할 수 있습니다.

```
 CUDA_VISIBLE_DEVICES=0,1,2,3 python challenge_test_time.py --cfg ./best_cfgs/[yaml 파일 이름] --output_dir ./output/test-time-evaluation/"[실험 이름]"
```
 
### 주요 yaml 파일명 및 실험 방법 정리
- adacontrast.yaml : adapter를 추가하지 않은 실험, 모델 내부 모든 파라미터를 업데이트 합니다.
- adacontrast_bn.yaml : adapter를 추가하지 않은 실험, 모델 내부 Batch Normalization Layer의 파라미터를 업데이트 합니다.
- adacontrast_conv.yaml : adapter를 추가한 실험, adapter의 파라미터 만을 업데이트 합니다.
- adacontrast_all_conv.yaml : adapter를 추가한 실험, adapter와 모델 내부 모든 파라미터를 업데이트 합니다.
- adacontrast_bn_conv.yaml :  adapter를 추가한 실험, adapter와 모델 내부 Batch Normalization Layer의 파라미터를 업데이트 합니다.
- adacontrast_all_conv_refine.yaml : adapter를 추가한 실험, adapter와 모델 내부 모든 파라미터를 업데이트 합니다.
- 
### a의 값 변화에 따른 실험 결과 확인 방법
- adapter를 추가한 실험의 경우 yaml 파일 내부 `A` 의 값을 수정하여 실험을 다르게 할 수 있습니다.
- 예시
```
MODEL:
  A: 0.1 // 0.01 # 원하는 값 입력
```

# 제안하는 방법 설명


#  주요 코드
## Adapter를 추가한 모델
- Adapter를 추가한 모델을 구현하기 위해 ResNet50의 코드를 torch 라이브러리에서 가져와 수정하였습니다.
- `./src/models/ResNet_para.py`의 142 ~ 174번 줄에 위치하고 있습니다.
```
        ################################################################# func start
        self.a = a                                                                                a를 입력 받음
        self.parallel = conv3x3(inplanes, planes * self.expansion, stride, groups, dilation)      adapter
        ################################################################# func end

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        ################################################################# func start
        para = self.parallel(x)                                                                   adapter 계산
        ################################################################# func end

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        ################################################################# func start
        out = out + self.a * para                                                                   relu 연산전 adapter 출력값에 를 곱하고 out 값에 더해줌
        ################################################################# func end
        out = self.relu(out)

        return out
```
## Pseudo-label Refinement
- 아래는 pseudo label을 정제하는 코드입니다..
- `./src/methods/adacontrast_all_conv_refine.py`의 296 ~ 317번 줄에 위치합니다.
```
    def save_refine_psedo_lable(self, pseudo, epoch, iter):
    ############################################################################## func start
        start = iter * self.batch_size                                                                             # pseudo label bank의 위치를 설정, 매 순간 batch size 만큼의 pseudo label이 입력됨
        end = start + len(pseudo)                                                                                  # 한 epoch에서 마지막으로 입력되는 이미지는 batch size와 다름. 
        mask = torch.ones(len(pseudo), dtype=bool, device="cuda")                                                  # 마스크 정의

        self.psedo_lable_bank[start:end, epoch] = pseudo                                                           # pseudo label bank에 pseudo label 입력


        if epoch != 0:
        ######################################################### if start
            select_past = range(0, epoch)                                                                          # 첫번째 epoch부터 현재 epoch까지 range 설정

            for i in range(len(pseudo)):
            ######################################## for start
                mask[i] = len(torch.where(self.psedo_lable_bank[(start + i), select_past] != pseudo[i])[0]) < 1    # 첫번째 epoch부터 현재 epoch까지 pseudo label 확인후 다른 것이 있다면 False, 모두 같다면 True 저장
            ######################################## for end

        ############################################################## if end

    ############################################################################## func end
        return torch.where(mask == True)[0]                                                                        # 마스크에서 True인 위치를 반환
```

# 실험 결과
## a에 따른 실험결과
<table><thead>
  <tr>
    <th>a</th>
    <th>0</th>
    <th>1</th>
    <th>0.1</th>
    <th>0.01</th>
  </tr></thead>
<tbody>
  <tr>
    <td>AVG.<br>ACC.</td>
    <td>60.46</td>
    <td>53.63</td>
    <td>60.82</td>
    <td>60.62</td>
  </tr>
</tbody>
</table>

## Ablation
<table><thead>
  <tr>
    <th rowspan="2">Domain adapter</th>
    <th rowspan="2"> BatchNorm</th>
    <th rowspan="2">ResNet Block </th>
    <th rowspan="2">   Accuracy   </th>
  </tr>
  <tr>
  </tr></thead>
<tbody>
  <tr>
    <td></td>
    <td></td>
    <td></td>
    <td>59.47</td>
  </tr>
  <tr>
    <td></td>
    <td>   <br>   </td>
    <td>   √   </td>
    <td>68.85</td>
  </tr>
  <tr>
    <td></td>
    <td>   √   </td>
    <td></td>
    <td>63.14</td>
  </tr>
  <tr>
    <td>   √   </td>
    <td></td>
    <td></td>
    <td>60.64</td>
  </tr>
  <tr>
    <td>   √   </td>
    <td>   √   </td>
    <td></td>
    <td>63.17</td>
  </tr>
  <tr>
    <td>   √   </td>
    <td></td>
    <td>   √   </td>
    <td>68.87</td>
  </tr>
</tbody>
</table>

## ab
<table><thead>
  <tr>
    <th>Adapter  </th>
    <th>IPRF</th>
    <th>AVG.<br>ACC.</th>
  </tr></thead>
<tbody>
  <tr>
    <td></td>
    <td></td>
    <td>68.85</td>
  </tr>
  <tr>
    <td>✓</td>
    <td></td>
    <td>68.87</td>
  </tr>
  <tr>
    <td>✓</td>
    <td>✓</td>
    <td>68.98</td>
  </tr>
</tbody>
</table>
