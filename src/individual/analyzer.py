import os
import numpy as np
from utility import json_handler, video
from individual.individual import Individual
from tqdm import tqdm


def analyze(data_dir: str, homo: np.typing.NDArray):
    kps_json_path = os.path.join(data_dir, "json", "keipoints.json")
    keypoints_data = json_handler.load(kps_json_path)

    individuals = {}
    json_data = []
    for data in tqdm(keypoints_data):
        # trackingのデータを取得
        frame_num = data["frame"]
        pid = data["person"]
        keypoints = data["keypoints"]

        if pid not in individuals:
            individuals[pid] = Individual(pid, homo)
        ind = individuals[pid]

        ind.calc_indicator(frame_num, keypoints)

        # jsonフォーマットを作成して追加
        data = ind.to_json(frame_num)
        if data is not None:
            json_data.append(data)

    # jsonに書き込み
    json_path = os.path.join(data_dir, "json", "individual.json")
    json_handler.dump(json_data, json_path)
