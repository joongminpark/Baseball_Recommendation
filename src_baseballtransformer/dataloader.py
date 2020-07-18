import os
import logging
import pickle
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
import torch
import numpy as np


logger = logging.getLogger(__name__)

"""
구질 분류

직구류: FAST, TWOS, SINK, CUTT -> 0
횡(horizontal): SLID -> 1
종(vertical): CURV, CHUP, FORK -> 2
[cls] -> 3
[pad] -> 4
"""
ball2idx = {
    "FAST": 0,
    "TWOS": 0,
    "SINK": 0,
    "CUTT": 0,
    "KNUC": 0,
    "SLID": 1,
    "CURV": 2,
    "CHUP": 2,
    "FORK": 2,
}


def data_preprocess(data: List[Dict], is_train=False, bin_size=5) -> List:

    if is_train:
        # label=1인 경우만 고려(투수에게 유리한 상황)
        game_id, data = list(zip(*data.items()))
        data_split = list(zip(*[list(d.values()) for d in data]))
        label = data_split[-3]
        game_id, data = [np.array(i)[np.array(label, dtype=bool)].tolist() for i in (game_id, data)]
    else:
        game_id, data = list(zip(*data.items()))

    # data가 홀수인 경우 한 개 제거(for multi-gpu processing)
    if len(data) % 2 > 0:
        game_id = game_id[:-1]
        data = data[:-1]

    # pitcher, batter, state, pitch, pitch_set, event_id, label 구분
    data_split = list(zip(*[list(d.values()) for d in data]))
    event_id, label, pitch_memory, label_memory = data_split[-4:]
    data_split = data_split[:-4]
    pitch_ = data_split.pop(-2)

    # Convert str type in the pitch_memory to int type by ball2idx
    pitch_memory = [
        [ball2idx[m] if type(m) == str else 4 for m in memory] for memory in pitch_memory
    ]
    label_memory = [
        [3 if m == -1 else 4 if m == -2 else m for m in memory] for memory in label_memory
    ]

    pitch = [ball2idx[p] for p in pitch_]

    # pitcher, batter, state, pitch_set
    data_split2 = [[list(d.items()) for d in data] for data in data_split]
    data_split3 = [[list(zip(*d)) for d in data] for data in data_split2]
    (
        (pitcher_name, pitcher),
        (batter_name, batter),
        (state_name, state),
        (pitcher_set_name, pitch_set),
    ) = [list(zip(*data)) for data in data_split3]

    # Use only short features
    pitcher = [p[:5] + p[-28:-9] for p in pitcher]
    pitcher_name = [p[:5] + p[-28:-9] for p in pitcher_name]
    batter = [p[:4] + p[-27:-9] for p in batter]
    batter_name = [p[:4] + p[-27:-9] for p in batter_name]

    pitcher_name, batter_name, state_name, pitcher_set_name = [
        list(set(names))[0] for names in (pitcher_name, batter_name, state_name, pitcher_set_name)
    ]

    # quantile 이용해서 continuous feature를 discretization
    tmp = []
    for feat in (pitcher, batter):
        num_disc = 5 if feat == pitcher else 4
        feat_disc = np.array(feat).T[:num_disc]
        feat_cont = np.array(feat).T[num_disc:].astype(np.float)
        for i, f in enumerate(feat_cont):
            f = discretize(f, bin_size)
            feat_cont[i] = f
        tmp.append(np.concatenate((feat_disc, feat_cont.astype(np.int))).T.tolist())
    pitcher, batter = tmp

    # state는 'pit_total_count'만 10개로 discretization 수행(115개 -> 10개)
    state = np.array(state).T
    state[6] = discretize(state[6].astype(float), 10)
    state = np.array(state).T.tolist()

    # 해당 투수가 던질 수 없는 구종 mask
    pitch_avail = list(zip(*pitch_set))[1]

    def pitch_mask(x):
        # 던질 수 있는 구종 : 1, 불가능한 구종 구종 : 0
        ball_type = list(ball2idx.keys())
        x = x.split("_")
        pitch_list = [1 if i in x else 0 for i in ball_type]
        return pitch_list

    pitch_unavail_mask = list(map(pitch_mask, pitch_avail))

    return (
        game_id,
        label,
        pitch,
        pitch_unavail_mask,
        pitcher,
        batter,
        state,
        pitch_memory,
        label_memory,
        (pitcher_name, batter_name, state_name, pitcher_set_name),
    )


def discretize(feat_in, n_bin=5):
    """ Discretize the feature according to its quantile """
    feat = np.unique(feat_in)
    discrete_feat = feat
    prev_quantile = 0
    for i in range(n_bin):
        q = 1 / (n_bin) * (i + 1)
        quantile = np.quantile(feat, q)
        discrete_feat = np.where(
            (feat > prev_quantile) & (feat <= np.quantile(feat, q)), i, discrete_feat
        )
        prev_quantile = np.quantile(feat, q)
    feat_dict = {f: int(d) for f, d in zip(feat, discrete_feat)}
    feat_out = [feat_dict[f] for f in feat_in]
    return feat_out


class DataSets(Dataset):
    def __init__(self, file_path, memory_length=50, is_train=False):
        self.ball2idx = ball2idx
        self.memory_length = memory_length
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, "cached_" + filename)

        # embedding layer에 맞도록 0부터 시작하도록 조정
        # inn 1~13 => 1~10으로 수정(1~9이닝 + alpha)
        self.inn2idx = dict([(str(j), i) if j < 10 else (j, 9) for i, j in enumerate(range(1, 13))])
        self.batorder2idx = dict([(str(j), i) for i, j in enumerate(range(1, 10))])

        if os.path.exists(cached_features_file):
            # preprocess가 진행된 cached 파일을 불러옵니다.
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                (
                    self.pitcher,
                    self.batter,
                    self.state,
                    self.pitch,
                    self.pitch_mask,
                    self.label,
                    self.pitch_memory,
                    self.label_memory,
                    self.pitcher_name,
                    self.batter_name,
                    self.state_name,
                    self.pitcher_set_name,
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
                self.pitch,
                self.pitch_mask,
                self.pitcher,
                self.batter,
                self.state,
                self.pitch_memory,
                self.label_memory,
                (self.pitcher_name, self.batter_name, self.state_name, self.pitcher_set_name),
            ) = data_preprocess(data, is_train)

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(
                    (
                        self.pitcher,
                        self.batter,
                        self.state,
                        self.pitch,
                        self.pitch_mask,
                        self.label,
                        self.pitch_memory,
                        self.label_memory,
                        self.pitcher_name,
                        self.batter_name,
                        self.state_name,
                        self.pitcher_set_name,
                    ),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

    def __len__(self):
        return len(self.pitcher)

    def __getitem__(self, item):
        # 투수 feature 앞에 가능 구질 정보 추가
        pitcher = torch.tensor(
            np.array([int(i) for i in self.pitch_mask[item]] + list(self.pitcher[item][2:])).astype(
                np.int
            ),
            dtype=torch.long,
        )
        batter = torch.tensor(np.array(self.batter[item][2:]).astype(np.int), dtype=torch.long)
        state = torch.tensor(
            np.array(
                self.state[item][1:5]
                + [self.batorder2idx[self.state[item][5]]]
                + self.state[item][6:]
            ).astype(np.int),
            dtype=torch.long,
        )
        pitch = torch.tensor(self.pitch[item], dtype=torch.long)
        label = torch.tensor(self.label[item], dtype=torch.long)
        pitch_memory = self.pitch_memory[item][-self.memory_length :]
        label_memory = self.label_memory[item][-self.memory_length :]

        pitch_memory = torch.tensor(
            self.pitch_memory[item][-self.memory_length :] + [3], dtype=torch.long
        )
        label_memory = torch.tensor(
            self.label_memory[item][-self.memory_length :] + [1], dtype=torch.long
        )
        memory_mask = torch.where(
            pitch_memory == 4,
            torch.ones(pitch_memory.shape, dtype=torch.long),
            torch.zeros(pitch_memory.shape, dtype=torch.long),
        ).to(torch.bool)

        return (
            pitcher,  # num of unique values : (2 * 3, 5 * 19)
            batter,  # num of unique values : (2 * 2, 5 * 18)
            state,  # num of unique values : (4, 4, 5, 8, 9, 10, 16, 22, 23)
            pitch,
            label,  # 2 labels contained
            pitch_memory,
            label_memory,
            memory_mask,
        )
