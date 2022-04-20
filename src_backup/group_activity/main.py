from common import json
from tqdm import tqdm

from group_activity.group_activity import GroupActivity


def main(
    individual_activity_json_path, group_activity_json_path, field, method=None, **karg
):
    print("Running group activity...")
    individual_activity_datas = json.load(individual_activity_json_path)

    group_activity = GroupActivity(field, method)

    last_frame_num = individual_activity_datas[-1]["frame"] + 1
    for frame_num in tqdm(range(last_frame_num)):
        # フレームごとにデータを取得する
        frame_individual_activity_datas = [
            data for data in individual_activity_datas if data["frame"] == frame_num
        ]

        # 指標の計算
        group_activity.calc_indicator(
            frame_num, frame_individual_activity_datas, **karg
        )

    # jsonフォーマットを生成して書き込み
    group_activity_datas = group_activity.to_json()
    json.dump(group_activity_datas, group_activity_json_path)
