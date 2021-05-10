from common import json
from common.json import TRACKING_FORMAT
from indivisual_activity.indivisual_activity import IndiVisualActivity


def main(tracking_json_path, indivisual_activity_json_path, homo):
    tracking_datas = json.load(tracking_json_path)

    indivisual_activitys = []
    indivisual_activity_datas = []
    for item in tracking_datas:
        # trackingのデータを取得
        indivisual_activity_id = item[TRACKING_FORMAT[0]]
        frame_num = item[TRACKING_FORMAT[1]]
        keypoints = item[TRACKING_FORMAT[2]]
        vector = item[TRACKING_FORMAT[3]]
        average = item[TRACKING_FORMAT[4]]

        # indivisual_activityクラスを新規作成
        if len(indivisual_activitys) == indivisual_activity_id:
            indivisual_activitys.append(
                IndiVisualActivity(indivisual_activity_id, frame_num, homo))

        # 指標を計算
        indivisual_activitys[indivisual_activity_id].calc_indicator(
            keypoints, vector, average)

        # jsonフォーマットを作成して追加
        data = indivisual_activitys[indivisual_activity_id].to_json(frame_num)
        indivisual_activity_datas.append(data)

    # jsonに書き込み
    json.dump(indivisual_activity_datas, indivisual_activity_json_path)
