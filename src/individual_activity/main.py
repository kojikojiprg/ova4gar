from common import json
from common.json import TRACKING_FORMAT
from individual_activity.individual_activity import IndividualActivity


def main(tracking_json_path, individual_activity_json_path, homo):
    tracking_datas = json.load(tracking_json_path)

    individual_activitys = []
    individual_activity_datas = []
    for item in tracking_datas:
        # trackingのデータを取得
        individual_activity_id = item[TRACKING_FORMAT[0]]
        frame_num = item[TRACKING_FORMAT[1]]
        keypoints = item[TRACKING_FORMAT[2]]
        vector = item[TRACKING_FORMAT[3]]
        average = item[TRACKING_FORMAT[4]]

        # individual_activityクラスを新規作成
        if len(individual_activitys) == individual_activity_id:
            individual_activitys.append(
                IndividualActivity(individual_activity_id, frame_num, homo))

        # 指標を計算
        individual_activitys[individual_activity_id].calc_indicator(
            keypoints, vector, average)

        # jsonフォーマットを作成して追加
        data = individual_activitys[individual_activity_id].to_json(frame_num)
        individual_activity_datas.append(data)

    # jsonに書き込み
    json.dump(individual_activity_datas, individual_activity_json_path)