import os
from typing import Any, Dict

import numpy as np
import torch
from group.passing.passing_detector import PassingDetector
from individual.individual import Individual
from tqdm import tqdm
from utility import json_handler


class PassingDataset(torch.utils.data.Dataset):
    def __init__(self, x_dict, y_dict, seq_len):
        self.x, self.y = [], []
        for key in tqdm(x_dict.keys()):
            x_lst = x_dict[key]
            y_lst = y_dict[key]
            x_seq, y_seq = self.create_sequence(x_lst, y_lst, seq_len)
            self.x += x_seq
            self.y += y_seq

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __getitem__(self, index):
        return (
            torch.tensor(self.x[index]).float().to(self.device),
            torch.tensor(np.identity(2)[self.y[index]])
            .float()
            .to(self.device),  # one-hot
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
    x_dict: dict, y_dict: dict, seq_len: int, batch_size: int, shuffle: bool
):
    dataset = PassingDataset(x_dict, y_dict, seq_len)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return loader


def make_data_loaders(
    passing_detector: PassingDetector,
    individuals: Dict[str, Individual],
    cfg: dict,
):
    x_dict, y_dict = _make_all_data(passing_detector, individuals, cfg["frame_number"])

    seq_len = cfg["seq_len"]  # seq_lenをどこにするか決める？術室08と02のデータを入力する
    batch_size = cfg["batch_size"]

    train_ratio = cfg["train_ratio"]
    val_ratio = cfg["val_ratio"]
    train_len = int(len(x_dict) * train_ratio)
    val_len = int(len(x_dict) * val_ratio)

    random_keys = np.random.choice(list(x_dict.keys()), size=len(x_dict), replace=False)

    train_keys = random_keys[:train_len]
    val_keys = random_keys[train_len : train_len + val_len]
    test_keys = random_keys[train_len + val_len :]

    x_train_dict = {key: x_dict[key] for key in train_keys}
    y_train_dict = {key: y_dict[key] for key in train_keys}
    train_loader = make_data_loader(
        x_train_dict, y_train_dict, seq_len, batch_size, True
    )

    x_val_dict = {key: x_dict[key] for key in val_keys}
    y_val_dict = {key: y_dict[key] for key in val_keys}
    val_loader = make_data_loader(x_val_dict, y_val_dict, seq_len, 1, False)

    x_test_dict = {key: x_dict[key] for key in test_keys}
    y_test_dict = {key: y_dict[key] for key in test_keys}
    test_loader = make_data_loader(x_test_dict, y_test_dict, seq_len, 1, False)

    return train_loader, val_loader, test_loader


def _make_time_series_from_cfg(dataset_cfg: dict):
    ret_data = {}
    for room_num, room_cfg in dataset_cfg.items():
        room_data = {}
        for date, date_cfg in room_cfg.items():
            date_data = {}
            for num, row in date_cfg.items():
                frame_numbers = []
                for item in row:
                    target_persons = sorted([int(item[0]), int(item[1])])
                    begin = int(item[2])
                    end = int(item[3])
                    frame_numbers.append({
                        "targets": target_persons,
                        "begin": begin,
                        "end": end
                    })

                kps_json_path = os.path.join(
                    "data", room_num, date, "passing", num, ".json", "keypoints.json"
                )
                kps_data = json_handler.load(kps_json_path)

                max_frame = kps_data[-1]["frame"]
                time_series = []
                for frame_num in range(max_frame):
                    frame_data = [
                        data for data in kps_data if data["frame"] == frame_num
                    ]

                    for i in range(len(frame_data) - 1):
                        for j in range(i + 1, len(frame_data)):
                            id1 = frame_data[i]["id"]
                            id2 = frame_data[j]["id"]
                            for item in frame_numbers:
                                is_pass = 0
                                if id1 in item["targets"] and id2 in item["targets"]:
                                    if item["begin"] <= frame_num and frame_num <= item["end"]:
                                        is_pass = 1

                                time_series.append((frame_num, id1, id2, is_pass))

                time_series = sorted(
                    time_series, key=lambda x: x[2]
                )  # sorted by person2
                time_series = sorted(
                    time_series, key=lambda x: x[1]
                )  # sorted by person1
                date_data[num] = time_series
            room_data[date] = date_data
        ret_data[room_num] = room_data
        return ret_data


def _make_all_data(
    passing_detector: PassingDetector,
    individuals: Dict[str, Individual],
    dataset_cfg: dict,
):
    x_dict: Dict[str, Any] = {}
    y_dict: Dict[str, Any] = {}
    time_series = _make_time_series_from_cfg(dataset_cfg)
    for room_num, room_data in time_series.items():
        for date, date_data in room_data.items():
            for num, row in tqdm(date_data.items()):
                frame_num = row[0]
                id1 = f"{room_num}_{date}_{num}_{row[1]}"
                id2 = f"{room_num}_{date}_{num}_{row[2]}"

                features = passing_detector.extract_features(
                    frame_num, id1, id2, individuals
                )
                key = f"{room_num}_{date}_{num}_{row[1]}_{row[2]}"

                if key not in x_dict:
                    x_dict[key] = []
                    y_dict[key] = []

                x_dict[key].append(features)
                y_dict[key].append(row[3])

    return x_dict, y_dict
