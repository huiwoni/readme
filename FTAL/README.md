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
- `./tools/make_annotation.py` 를 통해 역재생 비디오의 annotation 파일을 생성할 수 있습니다.
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

## 학습
### 실험 1 
'''
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train_exp_2.py configs/adatad/thumos/e2e_thumos_videomae_h_768x1_160_adapter_bi_exp1.py
'''
### 실험 1-1
```
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train_exp_2.py configs/adatad/thumos/e2e_thumos_videomae_h_768x1_160_adapter_bi_exp1.py --for_resume [순재생 비디오 사전학습 모델 경로],--back_resume [역재생 비디오 사전학습 모델 경로]
```
순재생 비디오 사전학습 모델 경로 : `./exps/for_thumos/adatad/e2e_actionformer_videomae_h_768x1_160_adapter/gpu4_id0/checkpoint/epoch_73.pth`

역재생 비디오 사전학습 모델 경로 : `./exps/back_thumos/adatad/e2e_actionformer_videomae_h_768x1_160_adapter/gpu4_id0/checkpoint/epoch_61.pth`

### 실험 2
```
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train_exp_2.py configs/adatad/thumos/e2e_thumos_videomae_h_768x1_160_adapter_bi_exp2.py --for_resume [순재생 비디오 사전학습 모델 경로],--back_resume [역재생 비디오 사전학습 모델 경로]
```
순재생 비디오 사전학습 모델 경로 : `./exps/for_thumos/adatad/e2e_actionformer_videomae_h_768x1_160_adapter/gpu4_id0/checkpoint/epoch_73.pth`

역재생 비디오 사전학습 모델 경로 : `./exps/back_thumos/adatad/e2e_actionformer_videomae_h_768x1_160_adapter/gpu4_id0/checkpoint/epoch_61.pth`

### 실험 3-1
```
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train_exp_3.py configs/adatad/thumos/e2e_thumos_videomae_h_768x1_160_adapter_bi_exp3_1.py
```
### 실험 3-2
```
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train_exp_3.py configs/adatad/thumos/e2e_thumos_videomae_h_768x1_160_adapter_bi_exp3_2.py
```

### 실험 3-3
- 실험 3-2 완료 후 코드 작성 예정

# 실험 구성
## 실험 1
1. 두 개의 네트워크를 생성한다.
2. 두 개의 네트워크에 순재생 비디오와 역재생 비디오를 각각 입력하고, 마지막 Loss를 더한 후 Backward를 진행한다.

## 실험 1-1
1. 순재생 및 역재생 비디오를 통해 비디오를 통해 사전학습을 진행한다.
2. 두 개의 네트워크 생성 후 각각의 네트워크는 순재생 비디오를 사전학습한 모델, 역재생 비디오를 사전학습한 모델을 불러온다. 
3. 두 개의 네트워크에 순재생 비디오와 역재생 비디오를 각각 입력하고, 마지막 Loss를 더한 후 Backward를 진행한다.

## 실험 2
1. 실험 1-1의 단계 1, 단계 2까지 동일하게 진행한다.
2. 두 네트워크를 Freeze한 후 학습 가능한 모듈 삽입하고, 이 모듈만 학습이 가능하도록 한다.
3. 실험 1-1의 단계 3을 그대로 진행한다.

## 실험 3-1
1. 실험 1-1의 단계 1, 단계 2까지 동일하게 진행한다.
2. 두 네트워크는 매 시점에 대한 Class score(cs), 액션의 Start time, End time까지의 거리 ds, de를 출력한다.
- 순재생 비디오 기준
3. 두 네트워크의 cs, ds, de를 순재생 비디오 기준으로 정렬한다.
4. 순재생 ds와 역재생 de , 역재생 de와 순재생 ds를 각각 위아래로 concat한다.
5. (순재생 ds-역재생 de), (순재생 de-역재생 ds), cs를 순서로  옆으로 concat 후 FC layer 입력한다.
- 역재생 비디오 기준
6. 두 네트워크의 cs, ds, de를 역재생 비디오 기준으로 정렬한다.
7. 역재생 ds와 순재생 de , 순재생 de와 역재생 ds를 각각 위아래로 concat한다.
8. (역재생 ds-순재생 de), (역재생 de-순재생 ds), cs를 순서로 양옆으로 concat 후 FC layer 입력한다.
9. 최종적으로 순재생 비디오와 역재생 비디오 기준의 cs, ds, de를 각각 출력한다.
10. 최종 Loss를 계산하고 Backward를 진행한다.

## 실험 3-2
1. 실험 1-1의 단계 1, 단계 2까지 동일하게 진행한다.
2. 두 네트워크는 매 시점에 대한 Class score(cs), 액션의 Start time, End time까지의 거리 ds, de를 출력한다.
- 순재생 비디오 기준
3. 두 네트워크가 출력한 cs, ds, de를 순재생 비디오 기준으로 정렬한다.
4. 두 네트워크가 출력한 ds, de를 순재생 비디오 기준으로 정렬한다.
5. 순재생 ds와 역재생 de , 역재생 de와 순재생 ds를 각각 위아래로 concat한다.
6. 이전 단계에서 생성한 두개의 값을 양옆으로 concat 후 FC layer 입력한다.
7. 순재생 비디오 기준으로 cs를 정렬하고 stack 후 3d conv 에 입력한다. 
- 역재생 비디오 기준
7. 두 네트워크가 출력한 ds, de를 역재생 비디오 기준으로 정렬한다.
8. 역재생 ds와 순재생 de , 순재생 de와 역재생 ds를 각각 위아래로 concat한다.
9. 이전 단계에서 생성한 두개의 값을 양옆으로 concat 후 FC layer 입력한다.
10. 역재생 비디오 기준으로 cs를 정렬하고 stack 후 3d conv 에 입력한다.
- 최종 출력
11. 최종적으로 순재생 비디오와 역재생 비디오 기준의 cs, ds, de를 각각 출력한다.
12. 최종 Loss를 계산하고 Backward를 진행한다.

## 실험 3-3
1. 실험 1-1의 단계 1, 단계 2까지 동일하게 진행한다.
2. 두 네트워크는 매 시점에 대한 Class score(cs), 액션의 Start time, End time까지의 거리 ds, de를 출력한다.
- 순재생 비디오 기준
3. 두 네트워크가 출력한 cs, ds, de를 순재생 비디오 기준으로 정렬한다.
4. 두 네트워크가 출력한 ds, de를 순재생 비디오 기준으로 정렬한다.
5. 순재생 ds와 역재생 de , 역재생 de와 순재생 ds를 각각 위아래로 concat한다.
6. 이전 단계에서 생성한 두개의 값을 양옆으로 concat 후 FC layer 입력한다.
7. 순재생 비디오 기준으로 cs를 정렬하고 평균을 낸다.
- 역재생 비디오 기준
7. 두 네트워크가 출력한 ds, de를 역재생 비디오 기준으로 정렬한다.
8. 역재생 ds와 순재생 de , 순재생 de와 역재생 ds를 각각 위아래로 concat한다.
9. 이전 단계에서 생성한 두개의 값을 양옆으로 concat 후 FC layer 입력한다.
10. 역재생 비디오 기준으로 cs를 정렬하고 평균 낸다.
- 최종 출력
11. 최종적으로 순재생 비디오와 역재생 비디오 기준의 cs, ds, de를 각각 출력한다.
12. 최종 Loss를 계산하고 Backward를 진행한다.

# 주요 코드
## 실험 1, 실험 1-1
- Model, Optimizer, Scheduler, Model_ema을 모두 순재생 비디오 및 역재생 비디오를 위해 2개씩 정의해 구성한다.
- 예시
```
  for_model, back_model, for_optimizer, back_optimizer, for_scheduler, back_scheduler, for_model_ema, back_model_ema
```
- 두 네트워크가 출력한 Loss를 더하는 곳의 위치는 `/opentad/cores/train_bidirect_engine.py`의 64~71번 줄에서 확인 가능합니다.
## 실험 2
- 실험 2에서 새로 추가한 모듈만 학습 가능하도록 하지 않고, Detector도 함께 학습이 되도록 하였습니다.
- 이에 대한 수정이 필요합니다.
- 현재 코드는 Adatad에서 사용하는 adapter와 동일한 구조를 가지며, 기존 Adatad의 adapter 후방에 부착됩니다.
- 이에 대한 코드는 `./opentad/models/backbones/vit_adapter_bi_1.py`의 269 ~ 274번줄, 290~292번줄에서 확인가능합니다.
```
# 269 ~ 274 : 새로운 adpater 정의
self.bi_adapter = Adapter(
    embed_dims=embed_dims,
    kernel_size=3,
    dilation=1,
    temporal_size=temporal_size,
)
# 290~292 
if self.use_adapter:                  
    x = self.adapter(x, h, w)                  # 기존 Adatad의 adapter
    x = self.bi_adapter(x, h, w)               # 새로 추가한 adapter
```
- 또한 Backbone에서 새로 추가한 adapter만 학습이 가능하도록 하였습니다.
- 이에 대한 코드는 `./opentad/models/backbones/vit_adapter_bi_1.py`497~502번 줄에서 확인 가능합니다.
```
for block in self.blocks:
    for m, n in block.named_children():
        if "bi_adapter" not in m and m != "drop_path":       # bi_adapter의 파라미터가 아닌 것만 선택
            n.eval()
            for param in n.parameters():
                param.requires_grad = False                  # 선택된 파라미터는 학습이 불가능 하도록 함
```
## 실험 3-1, 3-2
### 역재생 비디오에 대한 GT 생성
- 순재생 비디오 기준 GT를 기반으로 역재생 비디오 기준 GT를 생성합니다.
- `./opentad/datasets/transforms/end_to_end.py`의 350~358번 줄에서 확인할 수 있습니다.
```
def mk_back_gt(self, forward_gt_segments, gt_labels, trunc_len):                                        # forward_gt_segments는 순재생 비디오 기준 행동 구간, GT label은 비디오에 나타나는 행동의 종류를 순서대로 나타낸 것, trunc_len은 전체 프레임의 개수
    backward_gt_segment = [[-1 * a[1], -1 * a[0]] for a in (forward_gt_segments - trunc_len + 1)]       # 역재생 행동 시작 시간= |행동 종료 frame - frame_num + 1|, 역재생 행동 종료 시간 =  |행동 시작 frame - frame_num + 1|
    backward_gt_segment.sort()
    backward_gt_segment = np.array(backward_gt_segment)
    backward_gt_label = gt_labels[list(range(np.size(gt_labels) - 1, -1, -1))]                          # 비디오 내부 행동의 순서를 거꾸로 뒤집음
    return backward_gt_segment, backward_gt_label

```
### 순재생 비디오 기준 또는 역재생 비디오 기준 정렬
- forward_sort, backward_sort 함수를 통해 확인할 수 있습니다.
- `./opentad/models/dense_heads/bi_anchor_free_head.py`에 위치해 있습니다.
- 순재생 비디오 기준 정렬 코드
```
    def forward_sort(self, for_cls_pred, back_cls_pred, reg_pred, back_reg_pred):
        # cls pred, bi cls pred
        ################################################################################################################
        for_cls_pred = [x.permute(0, 2, 1) for x in for_cls_pred]                                                    # list : (1, 20, 768), (1, 20, 384), ... , (1, 20, 24) -> list : (1, 768, 20), (1, 384, 20), ... , (1, 24, 20)
        for_cls_pred = torch.cat(for_cls_pred, dim=1)                                                                # list -> tensor : (1, 1512, 20)

        back_cls_pred = [pred[:, :, list(range(pred.size()[2] - 1, -1, -1))] for pred in back_cls_pred]              # 역재생 기준 cls_pred(cs)를 반대로 나열
        back_cls_pred = [x.permute(0, 2, 1) for x in back_cls_pred]                                                  # list : (1, 768, 20), (1, 384, 20), ... , (1, 24, 20)
        back_cls_pred = torch.cat(back_cls_pred, dim=1)                                                              # list -> tensor : (1, 1512, 20)
        ###############################################################################################################

        # reg_pred
        ################################################################################################################
        reg_pred = torch.cat(reg_pred, dim=-1).permute(0, 2, 1)                                                       # list : (1, 2, 766), (1, 2, 384), ... , (1, 2, 24) -> tensor : (1, 1512, 2)

        reg_pred_st = reg_pred[:, :, 0]                                                                               # ds만 분리, tensor : (1, 1512)
        reg_pred_ed = reg_pred[:, :, 1]                                                                               # de만 분리, tensor : (1, 1512)
        reg_pred_st = reg_pred_st.unsqueeze(dim=2)                                                                    # tensor : (1, 1512) -> tensor : (1, 1512, 1)
        reg_pred_ed = reg_pred_ed.unsqueeze(dim=2)                                                                    # tensor : (1, 1512) -> tensor : (1, 1512, 1)

        back_reg_pred = [pred[:,:,list(range(pred.size()[2]-1,-1,-1))] for pred in back_reg_pred]                     # 역재생 기준 reg_pred를 반대로 나열
        back_reg_pred = torch.cat(back_reg_pred, dim=-1).permute(0, 2, 1)

        back_reg_pred_st = back_reg_pred[:, :, 0]
        back_reg_pred_ed = back_reg_pred[:, :, 1] 
        back_reg_pred_st = back_reg_pred_st.unsqueeze(dim=2) 
        back_reg_pred_ed = back_reg_pred_ed.unsqueeze(dim=2) 
        ################################################################################################################
        return reg_pred_st, reg_pred_ed, back_reg_pred_st, back_reg_pred_ed, for_cls_pred, back_cls_pred            # 순재생 기준 행동 시작 및 종료 Frame까지 거리, 역재생 기준 행동 시작 및 종료 Frame까지 거리, 순재생 기준 Cs, 역재생 기준 ds, de
```

### 3-1 최종 예측
- forward_prediction, backward_prediction 함수를 통해 확인할 수 있으며 341~359, 361~380번줄에 위치합니다.
```
    def forward_prediction(self, for_reg_pred_st, for_reg_pred_ed, back_reg_pred_st, back_reg_pred_ed, for_cls_pred, back_cls_pred):
        for_reg_pred = torch.concat((for_reg_pred_st, for_reg_pred_ed), dim = 2)                                    # (순재생 ds - 순재생 de) concat, tensor : (1, 1512, 1) -> tensor : (1, 1512, 2)
        back_reg_pred = torch.concat((back_reg_pred_ed, back_reg_pred_st), dim=2)                                   # (역재생 de - 역재생 ds) concat, tensor : (1, 1512, 1) -> tensor : (1, 1512, 2)
        cls_pred = torch.concat((for_cls_pred, back_cls_pred), dim=1)                                               # (순재생 cs - 역재생 cs) concat, tensor : (1, 1512, 20) -> tensor : (1, 3024, 20)
        reg_pred = torch.concat((for_reg_pred, back_reg_pred), dim=1)                                               # (순재생 ds, de - 역재생 ds, de) concat, tensor : (1, 1512, 2) -> tensor : (1, 3024, 2)
        all_pred = torch.concat((reg_pred, cls_pred), dim=2)                                                        # ((de, ds) - cs) concat, tensor : (1, 3024, 2), (1, 3024, 20) -> tensor : (1, 3024, 22)

        all_pred = all_pred.squeeze(dim = 0)                                                                        # tensor : (1, 3024, 22) -> tensor : (3024, 22)
        all_pred = all_pred.permute(1, 0)                                                                           # tensor : (3024, 22) -> tensor : (22,3024)
        all_pred = self.forward_fc(all_pred)                                                                        # tensor : (22, 3024) * FC layer : (3024, 1512) = tensor : (22, 1512) 
        all_pred = all_pred.permute(1, 0)                                                                           # tensor : (22, 1512) -> tensor : (1512, 22) 
        all_pred = all_pred.unsqueeze(dim=0)                                                                        # tensor : (1512, 22) -> tensor : (1, 1512, 22) 

        final_reg_pred = all_pred[:,:,:2]                                                                           # ds, de 추출, tensor : (1, 1512, 2)
        final_cls_pre = all_pred[:,:,2:]                                                                            # cs 추출, tensor : (1, 1512, 20)
 
        final_reg_pred = F.relu(final_reg_pred)                                                                     # Relu

        return final_reg_pred, final_cls_pre                                                                        # (ds, de), cs 리턴
```

### 3-2 최종 예측
- forward_prediction, backward_prediction 함수를 통해 확인할 수 있으며 918~936, 938~957번줄에 위치합니다.
```
    def forward_prediction(self, for_reg_pred_st, for_reg_pred_ed, back_reg_pred_st, back_reg_pred_ed, for_cls_pred, back_cls_pred):
        for_reg_pred = torch.concat((for_reg_pred_st, for_reg_pred_ed), dim = 2)                                    # (순재생 ds - 순재생 de) concat, tensor : (1, 1512, 1) -> tensor : (1, 1512, 2)
        back_reg_pred = torch.concat((back_reg_pred_ed, back_reg_pred_st), dim=2)                                   # (역재생 de - 역재생 ds) concat, tensor : (1, 1512, 1) -> tensor : (1, 1512, 2)
        reg_pred = torch.concat((for_reg_pred, back_reg_pred), dim=1)                                               # (순재생 ds, de - 역재생 ds, de) concat, tensor : (1, 1512, 2) -> tensor : (1, 3024, 2)
        reg_pred = reg_pred.squeeze(dim = 0))                                                                       # tensor : (1, 1512, 2) -> tensor : (1512, 2) 
        reg_pred = reg_pred.permute(1, 0)                                                                           # tensor : (1512, 2)  -> tensor : (2, 1512) 

        reg_pred = self.forward_fc(reg_pred)                                                                        # tensor : (2, 1512) * FC layer : (1512, 1512) = tensor : (2, 1512)
 
        reg_pred = reg_pred.permute(1, 0)                                                                           # tensor : (2, 1512) -> tensor : (1512, 2)              
        reg_pred = reg_pred.unsqueeze(dim=0)                                                                        # tensor : (2, 1512) -> tensor : (1512, 2)    

        final_reg_pred = F.relu(reg_pred)                                                                           # Relu

        cls_pred = torch.concat((for_cls_pred, back_cls_pred), dim=0)                                               # (순재생 cs - 역재생 cs) concat, tensor : (1, 1512, 20) -> tensor : (1, 3024, 20)
        final_cls_pred = self.forward_conv(cls_pred.unsqueeze(dim=0))                                               # 3d conv
        final_cls_pred = final_cls_pred.squeeze(dim=0)                                                              # (1, 1, 1512, 20) -> (1, 1512, 20)

        return final_reg_pred, final_cls_pred
```
# 실험 결과
## 각 방법에 대한 평균 mAP 비교
- 실험 2는 다시 코드를 작성하고 진행해야 합니다.
- 실험 3-1, 실험 3-2는 Loss가 100 이상으로 출력되기 때문에 분석 진행 입니다.
- 실험 3-2에서 실험을 진행할 때 3d conv의 size = (2,3,3), padding = (0,1,1), stride = (1,1,1)로 설정하였습니다.

|     Method       |     순방향     |     역방향    |
|------------------|----------------|-------------- |
|     기존 방법    |     73.94      |     73.43     |
|     실험 1       |     73.91      |     73.58     |
|     실험 1-1     |     73.15      |     73.01     |
|     실험 2       |     73.91      |     73.12     |
|     실험 3-1     |     -          |     -         |
|     실험 3-2     |     -          |     -         |
## 실험 3-2 실험 분석
### cs 값 변화 분석

순재생 비디오 기준 실험
- 정답 Class 기준 Frame 방향
-> Action이 존재하는 Frame과 Action 사이의 Frame에 해당하는 score가 대부분 가장 작았습니다.
- Action이 존재하는 Frame 기준
-> 모든 Class에 대해 정답인 Class의 score가 대부분 가장 작았습니다.

역재생 비디오 기준 실험
- 정답 Class 기준 Frame 방향1
-> Action이 존재하는 Frame과 Action 사이의 Frame에 해당하는 score가 대부분 가장 컸습니다.
- Action이 존재하는 Frame 기준
-> 모든 Class에 대해 정답인 Class의 score가 대부분 가장 컸습니다.

정리
- 순재생 비디오 기준으로 정렬하여 3d conv를 통과한 값이 기대와 반대로 출력을 내고 있습니다.
- 순재생 비디오 기준 출력은 코드상 구현이 잘못되었을 확률이 있어 잘못된 부분이 있는지 확인해 보았습니다.
- 역재생 비디오 기준으로 정렬하여 3d conv를 통과한 값이 GT와 가까운 적절한 출력을 내고 있습니다.

#### video_validation_0000185.mp4
![image](https://github.com/user-attachments/assets/e2dc2f51-8f3a-43a6-a204-a4362a79d193)
#### video_validation_0000189.mp4
![image](https://github.com/user-attachments/assets/4aabc073-d305-4aa4-8f13-ef40b419c9a2)
#### video_validation_0000188.mp4
![image](https://github.com/user-attachments/assets/3a5dec1d-ba21-41fb-a71c-200c2bf2c88d)
### 순재생 비디오 기준 정렬 확인
- 역재생 비디오 기준 cs를 순재생 비디오 기준으로 나열 후 비교하였습니다.
- GT와 동일한 위치를 관찰했을 때 가장 작은 값이 위치하거나 주변 score보다 낮은 값들이 위치합니다.
![image](https://github.com/user-attachments/assets/774328be-cca3-432e-91f1-bd60fa070029)
### 3d conv 작동 확인
- 실제 3d conv의 weight, bias 값을 확인하고, GT주변에서 연산이 생각한 것과 동일하게 되고 있는지 확인해 보았습니다.
- 이때 역재생 비디오에 대한 cs는 순재생 비디오 기준으로 정렬되어 있습니다.
- 직접 계산한 값과 실제 출력값이 일치하는 것을 확인할 수 있습니다.
![image](https://github.com/user-attachments/assets/1141e934-e448-495d-837f-b99263c5584f)


