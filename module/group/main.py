from common import json
from group.group import Group


def main(person_json_path, group_json_path, homo):
    person_datas = json.load(person_json_path)

    group = Group(homo)

    last_frame_num = person_datas[-1]['image_id']
    for frame_num in last_frame_num + 1:
        # フレームごとにデータを取得する
        frame_person_datas = [
            data for data in person_datas if data['image_id'] == frame_num]

        # 指標の計算
        group.calc_indicator(frame_num, frame_person_datas)

    # jsonフォーマットを生成して書き込み
    group_datas = group.to_json()
    json.dump(group_datas, group_json_path)
