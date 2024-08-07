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
## Test-time Adaptation(TTA)
### 명령어

```
 CUDA_VISIBLE_DEVICES=0,1,2,3 python challenge_test_time.py --cfg ./best_cfgs/[yaml 파일 이름] --output_dir ./output/test-time-evaluation/"[실험 이름]"
```
 
### 주요 yaml 파일명 및 실험 방법 정리
- adacontrast.yaml : Domain Adapter를 추가하지 않은 실험, 모델 내부 모든 파라미터를 업데이트 합니다.
- adacontrast_bn.yaml : Domain Adapter를 추가하지 않은 실험, 모델 내부 Batch Normalization Layer의 파라미터를 업데이트 합니다.
- adacontrast_conv.yaml : Domain Adapter를 추가한 실험, Domain Adapter의 파라미터 만을 업데이트 합니다.
- adacontrast_all_conv.yaml : Domain Adapter를 추가한 실험, Domain Adapter와 모델 내부 모든 파라미터를 업데이트 합니다.
- adacontrast_bn_conv.yaml : Domain Adapter를 추가한 실험, Domain Adapter와 모델 내부 Batch Normalization Layer의 파라미터를 업데이트 합니다.
- adacontrast_all_conv_refine.yaml : Domain Adapter를 추가한 실험, Domain Adapter와 모델 내부 모든 파라미터를 업데이트 합니다.
- 
### a의 값 변화에 따른 실험 결과 확인 방법
- Domain Adapter를 추가한 실험의 경우 yaml 파일 내부 `A` 의 값을 수정하여 실험을 다르게 할 수 있습니다.
- 예시
- 
```
MODEL:
  A: 0.1 // 0.01 # 원하는 값 입력
```

# 제안하는 방법 설명
## Domain Adapter
![icce](https://github.com/user-attachments/assets/f4823e24-3d9e-4ec1-a315-752b0c0772d6)
- 위 방법은 ResNet50의 한 Block과 Domain Adapter를 병렬로 연결하는 방법입니다.
- Domain Adapter의 값이 Block의 값과 더해지기 전 a가 곱해집니다.
- Domain Adapter는 3x3 convolutiuon layer로 구성되어 있습니다.

## Inaccurate Pseudo-label Removal Filter(IPRF)
![icce2](https://github.com/user-attachments/assets/da691294-d6b3-479d-a8a9-b43de0348009)
- 현재부터 과거의 모든 Pseudo-label을 Pseudo-label Bank에 저장하고 관찰합니다.
- 모두 같은 Pseudo-label을 출력하였다면 이를 Pseudo-label로 사용합니다. 
- 하나라도 다른 Pseudo-label을 출력하였다면 제거됩니다. 

#  주요 코드
## Domain Adapter를 추가한 모델
- Domain Adapter를 추가한 모델을 구현하기 위해 ResNet50의 코드를 torch 라이브러리에서 가져와 수정하였습니다.
- `./src/models/ResNet_para.py`의 142 ~ 174번 줄에 위치하고 있습니다.
```
        ################################################################# func start
        self.a = a                                                                                a를 입력 받음
        self.parallel = conv3x3(inplanes, planes * self.expansion, stride, groups, dilation)      Domain Adapter
        ################################################################# func end

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        ################################################################# func start
        para = self.parallel(x)                                                                   Domain Adapter 출력값을 para에 저장
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
        out = out + self.a * para                                                                   relu 연산전 Domain Adapter의 출력값에 a를 곱하고 out 값에 더해줌
        ################################################################# func end
        out = self.relu(out)

        return out
```
## IPRF
- 아래는 부정확한 Pseudo-label을 제거하기 위한 코드입니다.
- 정확한 Pseudo-label의 위치를 Return 합니다.
- `./src/methods/adacontrast_all_conv_refine.py`의 296 ~ 317번 줄에 위치합니다.
```
    def save_refine_psedo_lable(self, pseudo, epoch, iter):
    ############################################################################## func start
        start = iter * self.batch_size                                                                             # Pseudo-label Bank의 위치를 설정, 매 순간 batch size 만큼의 pseudo label이 입력됨
        end = start + len(pseudo)                                                                                  # 한 epoch에서 마지막으로 입력되는 이미지는 batch size와 다름. 
        mask = torch.ones(len(pseudo), dtype=bool, device="cuda")                                                  # 마스크 정의

        self.psedo_lable_bank[start:end, epoch] = pseudo                                                           # Pseudo-label Bank에 Pseudo-label 입력


        if epoch != 0:
        ######################################################### if start
            select_past = range(0, epoch)                                                                          # 첫번째 epoch부터 현재 epoch까지 range 설정

            for i in range(len(pseudo)):
            ######################################## for start
                mask[i] = len(torch.where(self.psedo_lable_bank[(start + i), select_past] != pseudo[i])[0]) < 1    # 첫번째 epoch부터 현재 epoch까지 Pseudo-label 확인 후 다른 것이 있다면 False, 모두 같다면 True 저장
            ######################################## for end

        ############################################################## if end

    ############################################################################## func end
        return torch.where(mask == True)[0]                                                                        # 마스크에서 True인 위치를 반환
```

# 실험 결과
- real(R), clipart(C), painting(P), sketch(S) 의 도메인을 학습한 각각의 Source 모델을 사용하여 Adaptation을 진행합니다.
- R->C, R->P, P->C, C->S, S->P, R->S, P->R 7개의 도메인이 변화하는 경우에 대해 Adaptation을 진행하고 평균(AVG.) 정확도(ACC.)를 측정합니다.

## a의 값 변화에 따른 성능 변화
- Domain Adapter를 부착하고 Domain Adapter 내부의 파라미터만을 업데이트 하였을 때의 실험결과입니다.
- 평균 정확도를 측정할 때 사용하지 않는 C->P로 도메인이 변화하는 경우에 대해 성능을 평가하였습니다.
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

## 파라미터 업데이트 위치에 따른 성능 비교
- 파라미터를 업데이트 하는 위치에 따라 성능을 비교한 결과입니다.

<table><thead>
  <tr>
    <th rowspan="2">Domain Adapter</th>
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

## Ablation Study
<table><thead>
  <tr>
    <th> Domain Adapter </th>
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
