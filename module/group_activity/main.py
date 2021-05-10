from common import json
from group_activity.group_activity import GroupActivity


def main(
        indivisual_activity_json_path,
        group_activity_json_path,
        homo,
        field,
        method=None,
        **karg):
    indivisual_activity_datas = json.load(indivisual_activity_json_path)

    group_activity = GroupActivity(homo, field, method)

    last_frame_num = indivisual_activity_datas[-1]['image_id'] + 1
    for frame_num in range(last_frame_num):
        # フレームごとにデータを取得する
        frame_indivisual_activity_datas = [
            data for data in indivisual_activity_datas if data['image_id'] == frame_num]

        # 指標の計算
        group_activity.calc_indicator(
            frame_num,
            frame_indivisual_activity_datas,
            **karg)

    # jsonフォーマットを生成して書き込み
    group_activity_datas = group_activity.to_json()
    json.dump(group_activity_datas, group_activity_json_path)
