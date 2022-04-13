import inspect

import numpy as np
from common.default import ATTENTION_DEFAULT

# from common.functions import cos_similarity, normalize_vector
from common.functions import gauss
from common.json import GA_FORMAT, IA_FORMAT, START_IDX

# from common.object_point import EX0304


def calc_attention(
    frame_num,
    individual_activity_datas,
    queue,
    field,
    angle_range=ATTENTION_DEFAULT["angle"],
    division=ATTENTION_DEFAULT["division"],
    length=ATTENTION_DEFAULT["length"],
    sigma=ATTENTION_DEFAULT["sigma"],
    seq_len=ATTENTION_DEFAULT["seq_len"],
):
    key = inspect.currentframe().f_code.co_name.replace("calc_", "")
    json_format = GA_FORMAT[key]

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
            for data in individual_activity_datas:
                pos = data[IA_FORMAT[START_IDX + 0]]
                face_vector = data[IA_FORMAT[START_IDX + 1]]
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
                            json_format[0]: frame_num,
                            json_format[2]: [y, x],
                            json_format[4]: total,
                        }
                    )

    # push and pop queue
    queue.append(pixcel_datas)
    queue = queue[-seq_len:]

    return datas, queue


# def calc_attention(
#     frame_num,
#     individual_activity_datas,
#     object_points=EX0304,
#     angle_th=ATTENTION_DEFAULT["angle_th"],
#     th=ATTENTION_DEFAULT["count_th"],
# ):
#     key = inspect.currentframe().f_code.co_name.replace("calc_", "")
#     json_format = GA_FORMAT[key]

#     object_count = {label: 0 for label in object_points.keys()}
#     object_persons = {label: [] for label in object_points.keys()}
#     # person_num = len(individual_activity_datas)
#     th_cos = np.cos(np.deg2rad(angle_th))

#     for individual in individual_activity_datas:
#         position = individual[IA_FORMAT[START_IDX + 0]]
#         face = individual[IA_FORMAT[START_IDX + 1]]

#         if position is None or face is None:
#             continue

#         for label, obj in object_points.items():
#             # ポジションと対象物のベクトルを求める
#             pos2obj = np.array(obj) - position
#             pos2obj = normalize_vector(pos2obj.astype(float))

#             # コサイン類似度
#             cos = cos_similarity(face, pos2obj)
#             if cos >= th_cos:
#                 object_count[label] += 1
#                 object_persons[label].append(individual[IA_FORMAT[START_IDX + 0]])

#     datas = []
#     for label in object_points.keys():
#         datas.append(
#             {
#                 json_format[0]: frame_num,
#                 json_format[1]: label,
#                 json_format[2]: object_points[label],
#                 json_format[3]: object_persons[label],
#                 json_format[4]: object_count[label],
#             }
#         )

#     return datas


def calc_passing(
    frame_num, individual_activity_datas, queue_dict, model, pass_length=5
):
    key = inspect.currentframe().f_code.co_name.replace("calc_", "")
    json_format = GA_FORMAT[key]

    datas = []
    for i in range(len(individual_activity_datas) - 1):
        for j in range(i + 1, len(individual_activity_datas)):
            p1 = individual_activity_datas[i]
            p2 = individual_activity_datas[j]
            p1_id = p1["label"]
            p2_id = p2["label"]

            # get queue
            feature_key = f"{p1_id}_{p2_id}"
            if feature_key not in queue_dict:
                queue_dict[feature_key] = {"features": [], "duration": 0}
            queue = queue_dict[feature_key]

            # push and pop queue
            queue["features"] = model.extract_feature(p1, p2, queue["features"])

            # predict
            pred, queue["duration"] = model.predict(
                queue["features"], queue["duration"], pass_length
            )

            # update queue
            queue_dict[feature_key] = queue

            if pred == 1:
                datas.append(
                    {
                        json_format[0]: frame_num,
                        json_format[1]: [p1[IA_FORMAT[0]], p2[IA_FORMAT[0]]],
                        json_format[2]: [
                            p1[IA_FORMAT[START_IDX + 0]],
                            p2[IA_FORMAT[START_IDX + 0]],
                        ],
                        json_format[3]: pred,
                    }
                )

    return datas, queue_dict


# def calc_passing(frame_num, individual_activity_datas, clf):
#     key = inspect.currentframe().f_code.co_name.replace("calc_", "")
#     json_format = GA_FORMAT[key]

#     datas = []
#     for i in range(len(individual_activity_datas) - 1):
#         for j in range(i + 1, len(individual_activity_datas)):
#             p1 = individual_activity_datas[i]
#             p2 = individual_activity_datas[j]
#             pred = clf.predict(p1, p2)

#             if pred is not None:
#                 datas.append(
#                     {
#                         json_format[0]: frame_num,
#                         json_format[1]: [p1[IA_FORMAT[0]], p2[IA_FORMAT[0]]],
#                         json_format[2]: [
#                             p1[IA_FORMAT[START_IDX + 0]],
#                             p2[IA_FORMAT[START_IDX + 0]],
#                         ],
#                         json_format[3]: pred,
#                     }
#                 )

#     return datas


keys = list(GA_FORMAT.keys())
INDICATOR_DICT = {
    keys[0]: calc_attention,
    keys[1]: calc_passing,
}
