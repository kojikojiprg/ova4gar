from common import json
from common.json import TRACKING_FORMAT
from person.person import Person


def main(tracking_json_path, person_json_path, homo):
    tracking_datas = json.load(tracking_json_path)

    persons = []
    person_datas = []
    for item in tracking_datas.items():
        # trackingのデータを取得
        person_id = item[TRACKING_FORMAT[0]]
        frame_num = item[TRACKING_FORMAT[1]]
        keypoints = item[TRACKING_FORMAT[2]]
        vector = item[TRACKING_FORMAT[3]]
        average = item[TRACKING_FORMAT[4]]

        # personクラスを新規作成
        if len(persons) == person_id:
            persons.append(Person(person_id, frame_num, homo))

        # 指標を計算
        persons[person_id].calc_indicator(keypoints, vector, average)

        # jsonフォーマットを作成して追加
        data = persons[person_id].to_json(frame_num)
        if data is not None:
            person_datas.append(data)

    # jsonに書き込み
    json.dump(person_datas, person_json_path)
