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

## 데이터
- 데이터 셋 구조는 아래와 같으며 DomainNet126 데이터 셋은 다음 [링크](https://ai.bu.edu/M3SDA/)에서 다운 받을 수 있습니다.
+ `./best_cfgs`: 실험 방법, 세부사항을 설정할 수 있습니다.
+ `./datasets`
       |-- datasets 
  	        |-- DomainNet
                |-- clipart
                |-- painting
                |-- real
                |-- sketch

## 실험에 따른 명령어

    CUDA_VISIBLE_DEVICES=0,1,2,3 python challenge_test_time.py --cfg ./best_cfgs/parallel_psedo_contrast.yaml --output_dir ./output/test-time-evaluation/"[YOUR EXPERIMENRT NAME]"


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
    <th rowspan="2">ResNet Bloc</th>
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
