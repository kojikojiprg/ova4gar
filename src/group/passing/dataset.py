import os
from logging import Logger
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch
from individual.individual import Individual
from tqdm import tqdm
from utility import json_handler
from utility.functions import cos_similarity, gauss


class PassingDataset(torch.utils.data.Dataset):
    def __init__(self, x_dict, y_dict, seq_len):
        self.x, self.y = [], []
        for key in tqdm(x_dict.keys()):
            x_lst = x_dict[key]
            y_lst = y_dict[key]
            x_seq, y_seq = self.create_sequence(x_lst, y_lst, seq_len)
            self.x += x_seq
            self.y += y_seq

    def __getitem__(self, index):
        return (
            torch.tensor(self.x[index]).float(),
            torch.tensor(np.identity(2)[self.y[index]]).float(),  # one-hot
        )

    def __len__(self):
        return len(self.x)

    @staticmethod
    def create_sequence(x_lst, y_lst, seq_len):
        x_seq = []
        y_seq = []
        for i in range(len(x_lst) - seq_len + 1):
            x = x_lst[i : i + seq_len]
            x_seq.append(x)
            y_seq.append(y_lst[i + seq_len - 1])

        return x_seq, y_seq


def make_data_loader(
    x_dict: dict,
    y_dict: dict,
    seq_len: int,
    batch_size: int,
    shuffle: bool,
):
    dataset = PassingDataset(x_dict, y_dict, seq_len)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True
    )

    return loader


def make_data_loaders(
    individuals: Dict[str, Individual],
    cfg: dict,
    passing_defs: dict,
    logger: Logger,
):
    # transform dataset setting into time series data
    x_dict, y_dict = make_all_data(individuals, cfg["setting"], passing_defs, logger)

    seq_len = passing_defs["seq_len"]
    batch_size = cfg["batch_size"]

    train_ratio = cfg["train_ratio"]
    val_ratio = cfg["val_ratio"]
    train_len = int(len(x_dict) * train_ratio)
    val_len = int(len(x_dict) * val_ratio)

    np.random.seed(cfg["random_seed"])
    random_keys = np.random.choice(list(x_dict.keys()), size=len(x_dict), replace=False)

    train_keys = random_keys[:train_len]
    val_keys = random_keys[train_len : train_len + val_len]
    test_keys = random_keys[train_len + val_len :]

    if len(train_keys) > 0:
        logger.info("=> create train loader")
        x_train_dict = {key: x_dict[key] for key in train_keys}
        y_train_dict = {key: y_dict[key] for key in train_keys}
        train_loader = make_data_loader(
            x_train_dict, y_train_dict, seq_len, batch_size, True
        )
    else:
        logger.info("=> skip creating train loader")
        train_loader = None

    if len(val_keys) > 0:
        logger.info("=> create val loader")
        x_val_dict = {key: x_dict[key] for key in val_keys}
        y_val_dict = {key: y_dict[key] for key in val_keys}
        val_loader = make_data_loader(x_val_dict, y_val_dict, seq_len, 1, False)
    else:
        logger.info("=> skip creating val loader")
        val_loader = None

    if len(test_keys) > 0:
        logger.info("=> create test loader")
        x_test_dict = {key: x_dict[key] for key in test_keys}
        y_test_dict = {key: y_dict[key] for key in test_keys}
        test_loader = make_data_loader(x_test_dict, y_test_dict, seq_len, 1, False)
    else:
        logger.info("=> skip creating test loader")
        test_loader = None

    return train_loader, val_loader, test_loader


def make_all_data(
    individuals: Dict[str, Individual],
    dataset_cfg: dict,
    passing_defs: dict,
    logger: Logger,
):
    x_dict: Dict[str, Any] = {}
    y_dict: Dict[str, Any] = {}
    data_all = _make_time_series_from_cfg(dataset_cfg, logger)
    for room_num, room_data in data_all.items():
        for surgery_num, surgery_data in room_data.items():
            logger.info(f"=> extracting feature {room_num}_{surgery_num}")
            for data_num, time_series in tqdm(surgery_data.items()):
                queue_dict: Dict[str, list] = {}
                for pair_key, row in time_series.items():
                    id1, id2 = pair_key.split("_")
                    for (frame_num, is_pass) in row:
                        # frame_num, is_pass = row[0], row[1]

                        ind1 = individuals[f"{room_num}_{surgery_num}_{data_num}_{id1}"]
                        ind2 = individuals[f"{room_num}_{surgery_num}_{data_num}_{id2}"]

                        # queue
                        pair_key = f"{ind1.id}_{ind2.id}"
                        if pair_key not in queue_dict:
                            queue_dict[pair_key] = []
                        feature_que = queue_dict[pair_key]

                        # extract feature
                        feature_que = extract_feature(
                            ind1,
                            ind2,
                            feature_que,
                            frame_num,
                            passing_defs,
                            with_padding=False,
                        )

                        # save data
                        key = f"{room_num}_{surgery_num}_{data_num}_{ind1.id}_{ind2.id}"
                        if key not in x_dict:
                            x_dict[key] = []
                            y_dict[key] = []

                        if len(feature_que) > 0:
                            x_dict[key].append(feature_que[-1])
                            y_dict[key].append(is_pass)

    return x_dict, y_dict


def _make_time_series_from_cfg(dataset_cfg: dict, logger: Logger):
    ret_data = {}
    for room_num, room_cfg in dataset_cfg.items():
        room_data = {}
        for surgery_num, date_cfg in room_cfg.items():
            logger.info(f"=> createing time series {room_num}_{surgery_num}")
            surgery_data = {}
            for data_num, row in tqdm(date_cfg.items()):
                settings = []
                for item in row:
                    settings.append(
                        {
                            "id1": item[0],
                            "id2": item[1],
                            "begin": int(item[2]),
                            "end": int(item[3]),
                        }
                    )

                kps_json_path = os.path.join(
                    "data",
                    room_num,
                    surgery_num,
                    "passing",
                    data_num,
                    ".json",
                    "keypoints.json",
                )
                kps_data = json_handler.load(kps_json_path)

                max_frame = kps_data[-1]["frame"]
                time_series: Dict[str, list] = {}
                for frame_num in range(max_frame):
                    frame_data = [
                        data for data in kps_data if data["frame"] == frame_num
                    ]

                    for i in range(len(frame_data) - 1):
                        for j in range(i + 1, len(frame_data)):
                            id1 = frame_data[i]["id"]
                            id2 = frame_data[j]["id"]

                            pair_key = f"{id1}_{id2}"
                            if pair_key not in time_series:
                                time_series[pair_key] = []

                            is_pass = 0
                            for item in settings:
                                if (id1 == item["id1"] and id2 == item["id2"]) or (
                                    id1 == item["id2"] and id2 == item["id1"]
                                ):
                                    if (
                                        item["begin"] <= frame_num
                                        and frame_num <= item["end"]
                                    ):
                                        is_pass = 1

                            time_series[pair_key].append((frame_num, is_pass))

                surgery_data[data_num] = time_series
            room_data[surgery_num] = surgery_data
        ret_data[room_num] = room_data
    return ret_data


def _get_indicators(ind: Individual, frame_num: int):
    pos = ind.get_indicator("position", frame_num)
    body = ind.get_indicator("body", frame_num)
    arm = ind.get_indicator("arm", frame_num)
    wrist = (
        ind.get_keypoints("LWrist", frame_num),
        ind.get_keypoints("RWrist", frame_num),
    )
    ret = {"pos": pos, "body": body, "arm": arm, "wrist": wrist}
    return ret


def extract_feature(
    ind1: Individual,
    ind2: Individual,
    que: list,
    frame_num: int,
    defs: dict,
    with_padding: bool = True,
):
    # get indicator
    ind1_data = _get_indicators(ind1, frame_num)
    ind2_data = _get_indicators(ind2, frame_num)

    if not (None in ind1_data.values() or None in ind2_data.values()):
        ind1_data = SimpleNamespace(**ind1_data)
        ind2_data = SimpleNamespace(**ind2_data)

        # calc distance of position
        p1_pos = np.array(ind1_data.pos)
        p2_pos = np.array(ind2_data.pos)

        norm = np.linalg.norm(p1_pos - p2_pos, ord=2)
        distance = gauss(norm, mu=defs["dist_mu"], sigma=defs["dist_sig"])

        p1p2 = p2_pos - p1_pos
        p2p1 = p1_pos - p2_pos

        p1p2_sim = cos_similarity(ind1_data.body, p1p2)
        p2p1_sim = cos_similarity(ind2_data.body, p2p1)
        body_distance = (np.average([p1p2_sim, p2p1_sim]) + 1) / 2

        # calc arm average
        arm_ave = np.average([ind1_data.arm, ind2_data.arm])

        # calc wrist distance
        min_norm = np.inf
        for i in range(2):
            for j in range(2):
                norm = np.linalg.norm(
                    np.array(ind1_data.wrist[i]) - np.array(ind2_data.wrist[j]), ord=2
                )
                min_norm = float(norm) if norm < min_norm else min_norm

        wrist_distance = gauss(min_norm, mu=defs["wrist_mu"], sigma=defs["wrist_sig"])

        # concatnate to feature
        feature = [distance, body_distance, arm_ave, wrist_distance]
        que.append(feature)

    if len(que) < defs["seq_len"]:
        # 0 padding
        if with_padding:
            return que + [[0, 0, 0, 0] for _ in range(defs["seq_len"] - len(que))]
        else:
            return que
    else:
        return que[-defs["seq_len"] :]
