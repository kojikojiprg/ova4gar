from common import common, keypoint, database
from person import Person
import numpy as np


def track(keypoints_path, result_db_path):
    # keypoints.json を開く
    keypoints_all_frame = keypoint.read_json(keypoints_path)

    # データベースとテーブルを作成
    db = database.DataBase(result_db_path)
    db.drop_table(common.TRACKING_TABLE_NAME)
    db.create_table(common.TRACKING_TABLE_NAME, common.TRACKING_TABLE_COLS)

    # person クラスを初期化
    persons = []
    for i, keypoints in enumerate(keypoints_all_frame[0]):
        persons.append(Person(i, keypoints))

    # トラッキング
    datas = []
    for i, keypoints_lst in enumerate(keypoints_all_frame):
        # 状態をリセット
        for person in persons:
            person.reset()

        for keypoints in keypoints_lst:
            target = keypoints.get_middle('Hip')
            max_person = None
            max_prob = 0.0
            for person in persons:
                if not person.is_reset():
                    # アップデート済は飛ばす
                    continue

                # パーティクルが移動する確率を求める
                prob = person.probability(target)

                # 一番確率が高い人を取り出す
                if max_prob < prob:
                    max_person = person
                    max_prob = prob

            if max_person is not None:
                # パーティクルフィルタを更新
                max_person.update(keypoints)
            else:
                # 近くに人が見つからなかったときは新しい人を追加
                new = Person(len(persons), keypoints)
                new.update(keypoints)
                persons.append(new)

        for person in persons:
            if person.is_reset():
                # アップデートされていない人にNoneを入力してアップデート
                person.update(None)
            elif person.is_deleted():
                person.delete()

        for person in persons:
            datas.append((
                person.id,
                i,
                np.array(person.keypoints_lst[-1]),
                person.vector))

    # データベースに書き込み
    db.insert_datas(
        common.TRACKING_TABLE_NAME,
        list(common.TRACKING_TABLE_COLS.keys()),
        datas)
