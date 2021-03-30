from tracker.person import Person
from common.json import TRACKING_FORMAT


def track(keypoints_all_frame):
    # person クラスを初期化
    persons = []
    for i, keypoints in enumerate(keypoints_all_frame[0]):
        persons.append(Person(i, keypoints))

    # トラッキング
    tracking_results = []
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
                person.update_deleted()

        for person in persons:
            if person.keypoints_lst[-1] is not None:
                tracking_results.append({
                    TRACKING_FORMAT[0]: person.id,
                    TRACKING_FORMAT[1]: i,
                    TRACKING_FORMAT[2]: person.keypoints_lst[-1].to_json(),
                    TRACKING_FORMAT[3]: person.average_lst[-1].tolist(),
                    TRACKING_FORMAT[4]: person.vector.tolist(),
                })

    return tracking_results
