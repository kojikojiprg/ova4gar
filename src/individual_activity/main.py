from common import json
from common.json import TRACKING_FORMAT
from individual_activity.individual_activity import IndividualActivity
from tqdm import tqdm


def main(tracking_json_path, individual_activity_json_path, homo):
    print("Running individual activity...")
    tracking_datas = json.load(tracking_json_path)

    individual_activitys = []
    individual_activity_datas = []
    for item in tqdm(tracking_datas):
        # trackingのデータを取得
        individual_activity_id = item[TRACKING_FORMAT[0]]
        frame_num = item[TRACKING_FORMAT[1]]
        keypoints = item[TRACKING_FORMAT[2]]

        # individual_activityクラスを新規作成
        if len(individual_activitys) == individual_activity_id:
            individual_activitys.append(
                IndividualActivity(individual_activity_id, homo)
            )

        # 指標を計算
        individual_activitys[individual_activity_id].calc_indicator(
            frame_num, keypoints
        )

        # jsonフォーマットを作成して追加
        data = individual_activitys[individual_activity_id].to_json(frame_num)
        if data is not None:
            individual_activity_datas.append(data)

    # jsonに書き込み
    json.dump(individual_activity_datas, individual_activity_json_path)
