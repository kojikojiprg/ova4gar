from typing import Dict, List

import numpy as np
import torch
from individual.individual import Individual
from numpy.typing import NDArray
from utility.functions import gauss

from group.passing.dataset import extract_feature
from group.passing.lstm_model import LSTMModel


def passing(
    frame_num: int,
    individuals: List[Individual],
    queue_dict: Dict[str, list],
    passing_defs: dict,
    model: LSTMModel,
    device: str,
):
    all_features = []
    idx_pairs = []
    for i in range(len(individuals) - 1):
        for j in range(i + 1, len(individuals)):
            ind1 = individuals[i]
            ind2 = individuals[j]

            # get queue
            pair_key = f"{ind1.id}_{ind2.id}"
            if pair_key not in queue_dict:
                queue_dict[pair_key] = []
            queue = queue_dict[pair_key]

            # push and pop queue
            queue = extract_feature(ind1, ind2, queue, frame_num, passing_defs)

            idx_pairs.append([i, j])
            all_features.append(queue)

    if len(all_features) == 0:
        return [], queue_dict

    # predict
    with torch.no_grad():
        features = torch.Tensor(np.array(all_features)).float().to(device)
        preds = model(features)
        preds = preds.max(1)[1]
        preds = preds.cpu().numpy()

    # if a distance is long, overwrite predict result to 0
    idx = np.where(np.array(all_features).T[0] > passing_defs["dist_max"])
    preds[idx] = 0

    data = [
        {
            "persons": [individuals[idx_pair[0]].id, individuals[idx_pair[1]].id],
            "points": [
                individuals[idx_pair[0]].get_indicator("position", frame_num),
                individuals[idx_pair[1]].get_indicator("position", frame_num),
            ],
        }
        for idx_pair, pred in zip(idx_pairs, preds)
        if pred == 1
    ]

    return data, queue_dict


def attention(
    frame_num,
    individuals: List[Individual],
    queue: list,
    field: NDArray,
    defs: dict,
):
    angle_range = defs["angle"]
    division = defs["division"]
    length = defs["length"]
    sigma = defs["sigma"]
    seq_len = defs["seq_len"]

    angle_range = np.deg2rad(angle_range)

    # sum queue data
    if len(queue) > 0:
        sum_data = np.sum(queue, axis=0)
    else:
        sum_data = None

    # extract position and face vector
    poss_tmp, faces_tmp = [], []
    for ind in individuals:
        pos = ind.get_indicator("position", frame_num)
        face = ind.get_indicator("face", frame_num)
        if pos is not None and face is not None:
            poss_tmp.append(pos)
            faces_tmp.append(face)
    poss = np.array(poss_tmp)
    faces = np.array(faces_tmp)

    # prepair coordinations
    coors = np.array(
        [
            [x, y]
            for y in range(0, field.shape[0], division)
            for x in range(0, field.shape[1], division)
        ]
    )

    # calc difference of all individuals
    diffs_all_ind = np.array([coors - pos for pos in poss])

    # calc each pixel value
    pixcel_data = np.zeros((field.shape[1], field.shape[0]), dtype=np.float32)
    for face, diffs in zip(faces, diffs_all_ind):
        shitas = [
            np.arccos(
                np.dot(diff, face)
                / (np.linalg.norm(diff) * np.linalg.norm(face) + 1e-10)
            )
            for diff in diffs
        ]
        for i in range(len(diffs)):
            if -angle_range <= shitas[i] and shitas[i] <= angle_range:
                norm = np.linalg.norm(diffs[i])
                if norm <= length:
                    pixcel_data[tuple(coors[i])] += 1.0
                else:
                    pixcel_data[tuple(coors[i])] += gauss(norm, mu=length, sigma=sigma)

    if sum_data is not None:
        # moving average
        vals = (sum_data + pixcel_data) / seq_len
    else:
        vals = pixcel_data

    # concat result
    data = [
        {
            "point": coor,
            "value": vals[tuple(coor)],
        }
        for coor in coors
        if pixcel_data[tuple(coor)] > 1 / seq_len
    ]

    # push and pop queue
    queue.append(pixcel_data)
    queue = queue[-seq_len:]

    return data, queue
