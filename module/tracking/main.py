from common import json
from common.keypoint import Keypoints, KeypointsList
from tracking.tracking import tracking
import numpy as np


def main(keypoints_path, result_path):
    # keypoints.json を開く
    keypoints_all_frame = load_pose_json(keypoints_path)

    results = tracking.track(keypoints_all_frame)

    # jsonに書き込み
    json.dump(results, result_path)


def load_pose_json(json_path):
    json_data = json.load(json_path)
    datas = []

    keypoints_lst = KeypointsList()
    pre_no = 0
    for item in json_data:
        frame_no = item['image_id']

        if frame_no != pre_no:
            datas.append(keypoints_lst)
            keypoints_lst = KeypointsList()

        keypoints = Keypoints(np.array(item['keypoints']).reshape(17, 3))
        keypoints_lst.append(keypoints)
        pre_no = frame_no
    else:
        datas.append(keypoints_lst)

    return datas
