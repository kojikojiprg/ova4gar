from typing import Dict, List

import numpy as np
from individual.individual import Individual
from utility.functions import gauss

from group.passing_detector import PassingDetector


def attention(
    frame_num,
    individuals: List[Individual],
    queue,
    field,
    **defs,
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

    datas = []
    pixcel_datas = np.zeros((field.shape[1], field.shape[0]), dtype=np.float32)
    for y in range(0, field.shape[1], division):
        for x in range(0, field.shape[0], division):
            point = np.array([y, x])  # the coordination of each pixel
            value = 0.0  # value of each pixel
            for ind in individuals:
                pos = ind.get_indicator("position", frame_num)
                face_vector = ind.get_indicator("face", frame_num)
                if pos is None or face_vector is None:
                    continue

                # calc angle between position and point
                diff = point - np.array(pos)
                shita = np.arctan2(diff[1], diff[0])

                # calc face angle
                face_shita = np.arctan2(face_vector[1], face_vector[0])

                if (
                    face_shita - angle_range <= shita
                    and shita <= face_shita + angle_range
                ):
                    # calc norm between position and point
                    norm = np.linalg.norm(diff)
                    if norm <= length:
                        value += 1.0
                    else:
                        value += gauss(norm, mu=length, sigma=sigma)

                pixcel_datas[y, x] = value  # save every frame value

                if value >= 1 / seq_len:
                    total = value
                    if sum_data is not None:
                        # sum all pixel data in queue
                        total += sum_data[y, x]
                    total /= seq_len

                    datas.append(
                        {
                            "frame": frame_num,
                            "point": [y, x],
                            "value": total,
                        }
                    )

    # push and pop queue
    queue.append(pixcel_datas)
    queue = queue[-seq_len:]

    return datas, queue


def passing(
    frame_num: int,
    individuals: List[Individual],
    queue_dict: Dict[str, list],
    model: PassingDetector,
):
    datas = []
    for i in range(len(individuals) - 1):
        for j in range(i + 1, len(individuals)):
            p1 = individuals[i]
            p2 = individuals[j]
            p1_id = p1["label"]
            p2_id = p2["label"]

            # get queue
            pair_key = f"{p1_id}_{p2_id}"
            if pair_key not in queue_dict:
                queue_dict[pair_key] = []
            queue = queue_dict[pair_key]

            # push and pop queue
            queue = model.extract_feature(p1, p2, queue)

            # predict
            pred = model.predict(queue)

            # update queue
            queue_dict[pair_key] = queue

            if pred == 1:
                datas.append(
                    {
                        "frame": frame_num,
                        "persons": [p1.id, p2.id],
                        "points": [
                            p1.get_indicator("position", frame_num),
                            p2.get_indicator("position", frame_num),
                        ],
                        "pred": pred,
                    }
                )

    return datas, queue_dict
