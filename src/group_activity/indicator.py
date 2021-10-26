from common.default import ATTENTION_DEFAULT
from common.json import IA_FORMAT, GA_FORMAT
from common.functions import cos_similarity, normalize_vector
from common.object_point import EX0304
import inspect
import numpy as np


# def calc_attention(
#         frame_num,
#         individual_activity_datas,
#         field,
#         angle_range=ATTENTION_DEFAULT['angle'],
#         division=ATTENTION_DEFAULT['division']):
#     key = inspect.currentframe().f_code.co_name.replace('calc_', '')
#     json_format = GA_FORMAT[key]

#     angle_range = np.deg2rad(angle_range)

#     pixcel_datas = np.zeros((field.shape[1], field.shape[0]))
#     for x in range(0, field.shape[1], division):
#         for y in range(0, field.shape[0], division):
#             point = np.array([x, y])
#             for data in individual_activity_datas:
#                 pos = data[IA_FORMAT[3]]
#                 face_vector = data[IA_FORMAT[4]]
#                 if pos is None or face_vector is None:
#                     continue

#                 diff = point - np.array(pos)
#                 shita = np.arctan2(diff[1], diff[0])

#                 face_shita = np.arctan2(face_vector[1], face_vector[0])

#                 if (
#                     face_shita - angle_range <= shita and
#                     shita <= face_shita + angle_range
#                 ):
#                     pixcel_datas[x, y] += 1

#     datas = []
#     for x, row in enumerate(pixcel_datas):
#         for y, data in enumerate(row):
#             if data > 0:
#                 datas.append({
#                     json_format[0]: frame_num,
#                     json_format[1]: [x, y],
#                     json_format[2]: data
#                 })

#     return datas


def calc_attention(
        frame_num,
        individual_activity_datas,
        object_points=EX0304,
        angle_th=ATTENTION_DEFAULT['angle_th'],
        th=ATTENTION_DEFAULT['count_th']):
    key = inspect.currentframe().f_code.co_name.replace('calc_', '')
    json_format = GA_FORMAT[key]

    object_count = {label: 0 for label in object_points.keys()}
    object_persons = {label: [] for label in object_points.keys()}
    # person_num = len(individual_activity_datas)
    th_cos = np.cos(np.deg2rad(angle_th))

    for individual in individual_activity_datas:
        position = individual[IA_FORMAT[3]]
        face = individual[IA_FORMAT[4]]

        if position is None or face is None:
            continue

        for label, obj in object_points.items():
            # ポジションと対象物のベクトルを求める
            pos2obj = np.array(obj) - position
            pos2obj = normalize_vector(pos2obj.astype(float))

            # コサイン類似度
            cos = cos_similarity(face, pos2obj)
            if cos >= th_cos:
                object_count[label] += 1
                object_persons[label].append(individual[IA_FORMAT[3]])

    datas = []
    for label in object_points.keys():
        datas.append({
            json_format[0]: frame_num,
            json_format[1]: label,
            json_format[2]: object_points[label],
            json_format[3]: object_persons[label],
            json_format[4]: object_count[label]
        })

    return datas


def calc_passing(
    frame_num, individual_activity_datas, clf
):
    key = inspect.currentframe().f_code.co_name.replace('calc_', '')
    json_format = GA_FORMAT[key]

    datas = []
    for i in range(len(individual_activity_datas) - 1):
        for j in range(i + 1, len(individual_activity_datas)):
            p1 = individual_activity_datas[i]
            p2 = individual_activity_datas[j]
            pred = clf.predict(p1, p2)

            if pred is not None:
                datas.append({
                    json_format[0]: frame_num,
                    json_format[1]: [p1[IA_FORMAT[0]], p2[IA_FORMAT[0]]],
                    json_format[2]: [p1[IA_FORMAT[3]], p2[IA_FORMAT[3]]],
                    json_format[3]: pred})

    return datas


keys = list(GA_FORMAT.keys())
INDICATOR_DICT = {
    keys[0]: calc_attention,
    keys[1]: calc_passing,
}
