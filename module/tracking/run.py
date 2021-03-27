from common.keypoint import Keypoints, KeypointsList
from tracking.tracking import tracking
import json
import numpy as np


TRACKING_FORMAT = [
    'person_id',
    'image_id',
    'keypoints',
    'vector',
    'average',
]


def main(keypoints_path, result_path):
    # keypoints.json を開く
    keypoints_all_frame = load_pose_json(keypoints_path)

    results = tracking.track(keypoints_all_frame, TRACKING_FORMAT)

    # jsonに書き込み
    write_tracking_json(results, result_path)


def load_pose_json(json_path):
    return_lst = []
    with open(json_path, 'r') as f:
        dat = json.load(f)

        keypoints_lst = KeypointsList()
        pre_no = 0
        for item in dat:
            frame_no = item['image_id']

            if frame_no != pre_no:
                return_lst.append(keypoints_lst)
                keypoints_lst = KeypointsList()

            keypoints = Keypoints(np.array(item['keypoints']).reshape(17, 3))
            keypoints_lst.append(keypoints)
            pre_no = frame_no
        else:
            return_lst.append(keypoints_lst)

    return return_lst


def write_tracking_json(datas, json_path):
    with open(json_path, 'w') as f:
        json.dump(datas, f)
