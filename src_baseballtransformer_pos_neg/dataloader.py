import os
import logging
import pickle
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
import torch
import numpy as np


logger = logging.getLogger(__name__)

ball2idx = {
    "CHUP": 0,
    "CURV": 1,
    "CUTT": 2,
    "FAST": 3,
    "FORK": 4,
    "KNUC": 5,
    "SINK": 6,
    "SLID": 7,
    "TWOS": 8,
}

# 긍/부정 상황을 모두 이용한 학습
'''
Case 1: 긍,부정 & 구질 인과관계 반영
        -> negative_label_change=True  (부정상황일 때 투구구질 Label 변화)
Case 2: Transfer Learning (긍정 상황에서 먼저 학습 → 부정 상황에서 학습)
        -> negative_label_change=True, is_transfer=True, transfer_positive_dataset(True, False)
          (부정상황일 때 투구구질 Label 변화 + Transfer learning)
Case 3: 긍, 부정 상황을 Input으로 입력 →  Inference 에서 긍, 부정 상황의 컨트롤을 통한 결과 기대
        -> negative_label_change=False, is_transfer=False, label_switch_inference(True, False)
        (부정상황일 때 투구구질 Label 변화하지 않음 + Inference 때 긍, 부정 상황 Control)
'''

def data_preprocess(data: List[Dict], is_binary=False, is_train=False, negative_label_change=False,\
    is_transfer=False, transfer_positive_dataset=False, label_switch_inference=False) -> List:
    # label_switch_inference 는 input에 sentiment 넣었을 때만 사용
    '''Data preprocessing.

    # Arguments
        data: binary_data(긍정&부정 -> 투구예측), multi_data(긍정&중립&부정 -> 투구예측)
        is_binary: True -> binary_data, False -> multi_data
        is_train: 데이터로 학습인지, inference인지 여부
        negative_label_change: 부정 상황일 때, 투구 label을 변화할지 여부
        is_transfer: 긍정상황 선수 학습 -> 부정상황 이후 학습 (Transfer learning: 부정상황 Label 더 반영하기 위함)
        transfer_positive_dataset: Transfer learning 일 때, dataset 선별 (긍정만 or 부정만)
        label_switch_inference: Case 3의 모델로 Inference 상황일 때, 긍/부정 상황을 반대로 하기 (결과 보기 위함)
    '''

    # 긍, 부정상황 binary (0, 1)
    if is_binary:
        # transfer learning 일 경우 (긍,부정 dataset 나눔)
        if is_transfer:
            if transfer_positive_dataset:
                # label=1 경우만 고려(투수에게 유리한 상황)
                game_id, data = list(zip(*data.items()))
                data_split = list(zip(*[list(d.values()) for d in data]))
                label = data_split.pop()
                game_id, data = [np.array(i)[np.array(label, dtype=bool)].tolist() for i in (game_id, data)]
            else:
                # label=0 경우만 고려(투수에게 불리한 상황)
                game_id, data = list(zip(*data.items()))
                data_split = list(zip(*[list(d.values()) for d in data]))
                label = data_split.pop()
                game_id, data = [np.array(i)[np.array(label) == 0].tolist() for i in (game_id, data)]
        
        # transfer learning 아닐 경우 (긍,부정 dataset 통합)
        else:
            game_id, data = list(zip(*data.items()))
            
    # 긍, 부정 상황 Tri (-1, 0, 1)
    else:
        # transfer learning 일 경우 (긍,부정 dataset 나눔)
        if is_transfer:
            if transfer_positive_dataset:
                # label=1 경우만 고려(투수에게 유리한 상황)
                game_id, data = list(zip(*data.items()))
                data_split = list(zip(*[list(d.values()) for d in data]))
                label = data_split.pop()
                game_id, data = [np.array(i)[np.array(label) == 1].tolist() for i in (game_id, data)]
            else:
                # label=-1 경우만 고려(투수에게 불리한 상황)
                game_id, data = list(zip(*data.items()))
                data_split = list(zip(*[list(d.values()) for d in data]))
                label = data_split.pop()
                game_id, data = [np.array(i)[np.array(label) == -1].tolist() for i in (game_id, data)]
        
        # transfer learning 아닐 경우 (긍,부정 dataset 통합): 중립은 제외
        else:
            game_id, data = list(zip(*data.items()))
            data_split = list(zip(*[list(d.values()) for d in data]))
            label = data_split.pop()
            # 0을 제외한 -1, 1 추출
            game_id, data = [np.array(i)[np.array(label, dtype=bool)].tolist() for i in (game_id, data)]


    # pitcher, batter, state, pitch, pitch_set, event_id, label 구분
    data_split = list(zip(*[list(d.values()) for d in data]))

    # multi label일 경우 -1 -> 0으로 변환 (중립은 제외했기 때문)
    if is_binary:
        label, event_id = [data_split.pop() for _ in range(2)]
    else:
        label, event_id = [data_split.pop() for _ in range(2)]
        label = np.array(label)
        label[label == -1] = 0
        label = label.tolist()
    
    # Case 3 모델일 때, Inference 상황에서 긍,부정 변화(label 0, 1 -> 1, 0)
    if label_switch_inference:
        label = np.array(label)
        label += 1
        label[label == 2] = 0
        label = label.tolist()

    pitch_ = data_split.pop(3)

    # 투구구질, 긍/부정 상황의 비율이 다르기 때문에 Sampling 빈도 설정을 위한 count 추출
    '''
    투구구질: 총 9개 -> 이후 "KNUC", "SINK" 구종은 제외함(빈도수 낮음)  -> 총 7개
    긍/부정: 총 2개 -> 6:1 비율
    '''
    # Get counts of the pitches
    pitch_counts = list(np.unique(pitch_, return_counts=True))
    pitch_counts[0] = [ball2idx[t] for t in pitch_counts[0]]
    pitch_counts = dict(sorted(list(zip(*pitch_counts)), key=lambda x: x[0]))

    # Get counts of the batter labels
    label_counts = dict(zip(*np.unique(label, return_counts=True)))

    # Get the intersection of pitch and label counts
    pitch_and_label_count = list(np.unique(list(zip(*(pitch_, label))), return_counts=True, axis=0))
    counts = np.unique(list(zip(*(pitch_, label))), return_counts=True, axis=0)
    value = counts[1].tolist()
    key = [(ball2idx[p], int(l)) for p, l in counts[0].tolist()]
    pitch_and_label_count = dict(zip(*(key, value)))

    # pitcher, batter, state, pitch_set
    data_split2 = [[list(d.items()) for d in data] for data in data_split]
    data_split3 = [[list(zip(*d)) for d in data] for data in data_split2]
    (
        (pitcher_name, pitcher),
        (batter_name, batter),
        (state_name, state),
        (pitcher_set_name, pitch_set),
    ) = [list(zip(*data)) for data in data_split3]

    pitcher_name, batter_name, state_name, pitcher_set_name = [
        list(set(names))[0] for names in (pitcher_name, batter_name, state_name, pitcher_set_name)
    ]

    # 해당 투수가 던질 수 없는 구종 mask
    pitch_avail = list(zip(*pitch_set))[1]
    ball_list = list(ball2idx.keys())

    def pitch_mask(x):
        # 던질 수 있는 구종 : 1, 던질수 없는 구종 : 0
        x = x.split("_")
        # rarely happend label is discared (KNUC, SINK)
        pitch_list = [ball_list[i] in x if i not in [5, 6] else False for i in range(9)]
        return pitch_list

    # 정답 label에 불가능한 구종은 -100으로 masking -> 추후 crossentropy 계산 시 무시하기
    '''
    Train 과정에서
    -> pitch = np.where(pitch_unavail_mask, pitch, -100).tolist()
    '''
    pitch_unavail_mask = list(map(pitch_mask, pitch_avail))


    ##### 긍/부정에 따른 pitch label 부여
    # change pitch type to onehot
    def to_onehot_train(idx, length, label):
        vec = [0] * length
        if label == 1:
            vec[idx] = 1
        else:
            vec = [0 if i == idx else 1.0 / 8.0 for i, j in enumerate(vec)]
        return vec

    # Test에서는 Label 변화시키지 않음
    def to_onehot_test(idx, length):
        vec = [0] * length
        vec[idx] = 1
        return vec

    ##### 긍/부정에 따른 pitch label 부여(학습, 테스트시)
    if negative_label_change:
        if is_train:
            pitch = [to_onehot_train(ball2idx[p], 9, l) for p, l in zip(pitch_, label)]
            origin_pitch = [to_onehot_test(ball2idx[p], 9) for p in pitch_]
        else:
            pitch = [to_onehot_test(ball2idx[p], 9) for p in pitch_]
            origin_pitch = [to_onehot_test(ball2idx[p], 9) for p in pitch_]
    else:
        pitch = [to_onehot_test(ball2idx[p], 9) for p in pitch_]
        origin_pitch = [to_onehot_test(ball2idx[p], 9) for p in pitch_]
    
    # 원래 못던지는 것들 다 0으로 
    pitch = np.where(pitch_unavail_mask, pitch, 0).tolist()

    return (
        game_id,
        label,
        event_id,
        pitch,
        origin_pitch,
        pitch_unavail_mask,
        pitcher,
        batter,
        state,
        pitch_counts,
        label_counts,
        pitch_and_label_count,
        (pitcher_name, batter_name, state_name, pitcher_set_name),
    )


class DataSets(Dataset):
    def __init__(self, file_path, reverse=False, is_binary=False, is_train=False, 
    negative_label_change=False, is_transfer=False, transfer_positive_dataset=False, label_switch_inference=False):
        '''Make dataset.

        # Arguments
            file_path: binary_data 경로(긍정&부정 -> 투구예측), multi_data 경로(긍정&중립&부정 -> 투구예측)
            reverse: 관계없는 변수(Multi-task: 구질&타격 여부 동시예측에서 사용했음)
            is_binary: True -> binary_data, False -> multi_data
            is_train: 데이터로 학습인지, inference인지 여부
            negative_label_change: 부정 상황일 때, 투구 label을 변화할지 여부
            is_transfer: 긍정상황 선수 학습 -> 부정상황 이후 학습 (Transfer learning: 부정상황 Label 더 반영하기 위함)
            transfer_positive_dataset: Transfer learning 일 때, dataset 선별 (긍정만 or 부정만)
            label_switch_inference: Case 3의 모델로 Inference 상황일 때, 긍/부정 상황을 반대로 하기 (결과 보기 위함)
        '''
        self.ball2idx = ball2idx
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)

        # Case 2 모델의 Transfer learning 경우: pos, neg 상황 분리
        if is_transfer:
            # Postive dataset
            if transfer_positive_dataset:
                cached_features_file = os.path.join(directory, "cached_positive_" + filename)
            # Negative dataset
            else:
                cached_features_file = os.path.join(directory, "cached_negative_" + filename)
        # Transfer learning 아닐 경우
        else:
            # Case 3 모델일 경우: 긍,부정 label 바꿀지
            if label_switch_inference:
                cached_features_file = os.path.join(directory, "cached_label_switch_" + filename)
            # 긍, 부정 label 그대로 할지
            else:
                cached_features_file = os.path.join(directory, "cached_" + filename)

        # embedding layer를 위한 dictionary
        # inn 1~13 => 1~10으로 수정(1~9이닝 + alpha)
        self.inn2idx = dict([(j, i) if j < 10 else (j, 9) for i, j in enumerate(range(1, 13))])
        self.batorder2idx = dict([(j, i) for i, j in enumerate(range(1, 10))])

        if os.path.exists(cached_features_file):
            # preprocess가 진행된 cached 파일을 불러옵니다.
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                (
                    self.pitcher,
                    self.batter,
                    self.state,
                    self.hit,
                    self.pitch,
                    self.origin_pitch,
                    self.pitch_mask,
                    self.label,
                    self.pitcher_name,
                    self.batter_name,
                    self.state_name,
                    self.pitcher_set_name,
                    self.pitch_counts,
                    self.label_counts,
                    self.pitch_and_label_count,
                ) = pickle.load(handle)
        else:
            # 최초 데이터 로드 시 시간이 걸리는 작업을 수행합니다.
            # 두 번째 실행부터는 cached 파일을 이용합니다.
            logger.info("Creating features from dataset file at %s", directory)

            with open(file_path, "rb") as f:
                data = pickle.load(f)

            (
                game_id,
                self.label,
                event_id,
                self.pitch,
                self.origin_pitch,
                self.pitch_mask,
                self.pitcher,
                self.batter,
                state,
                self.pitch_counts,
                self.label_counts,
                self.pitch_and_label_count,
                (self.pitcher_name, self.batter_name, self.state_name, self.pitcher_set_name),
            ) = data_preprocess(data, is_binary, is_train, negative_label_change,
            is_transfer, transfer_positive_dataset, label_switch_inference)

            if reverse:
                self.hit = [eid2hit_reverse[int(e)] for e in event_id]
            else:
                self.hit = [eid2hit[int(e)] for e in event_id]
            # self.pitch = [ball2idx[b] for b in pitch]
            # state는 모두 integer / str 인데 str 변수도 모두 int형태라 int로 통일
            self.state = [[int(i) for i in s] for s in state]

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(
                    (
                        self.pitcher,
                        self.batter,
                        self.state,
                        self.hit,
                        self.pitch,
                        self.origin_pitch,
                        self.pitch_mask,
                        self.label,
                        self.pitcher_name,
                        self.batter_name,
                        self.state_name,
                        self.pitcher_set_name,
                        self.pitch_counts,
                        self.label_counts,
                        self.pitch_and_label_count,
                    ),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

    def __len__(self):
        return len(self.pitcher)

    def __getitem__(self, item):
        pitcher_discrete = torch.tensor(
            [bool(i) for i in self.pitch_mask[item]] + list(self.pitcher[item][2:5]),
            dtype=torch.long,
        )
        pitcher_continuous = torch.tensor(self.pitcher[item][5:], dtype=torch.float)
        batter_discrete = torch.tensor(self.batter[item][2:4], dtype=torch.long)
        batter_continuous = torch.tensor(self.batter[item][4:], dtype=torch.float)

        state_discrete = torch.tensor(
            [self.inn2idx[self.state[item][0]]]
            + [self.state[item][4]]
            + [self.batorder2idx[self.state[item][5]]],
            dtype=torch.long,
        )
        state_continuous = torch.tensor(
            self.state[item][1:4] + self.state[item][6:], dtype=torch.float
        )

        pitch = torch.tensor(self.pitch[item], dtype=torch.float)
        origin_pitch = torch.tensor(self.pitch[item], dtype=torch.float)

        hit = torch.tensor(self.hit[item], dtype=torch.long)
        label = torch.tensor(self.label[item], dtype=torch.long)
        masked_pitch = torch.tensor([bool(i) for i in self.pitch_mask[item]], dtype=torch.long)
        return (
            pitcher_discrete,  # num of unique values : (*([2] * 9), 2, 2, 2)
            pitcher_continuous,
            batter_discrete,  # num of unique values : (2, 2)
            batter_continuous,
            state_discrete,  # num of unique values : (10, 8, 9)
            state_continuous,
            pitch,           # 7 가지의 투구구질
            hit,
            label,           # 긍,부정 상황
            masked_pitch,
            origin_pitch,
        )
