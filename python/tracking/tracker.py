from person import Person
from heatmap import Population


class Tracker:
    def __init__(self, initial_keypoints_lst, homography):
        self.persons = []
        for i, keypoints in enumerate(initial_keypoints_lst):
            self.persons.append(Person(i, keypoints))

        self.populations = Population(homography)

    def track(self, frame_num, keypoints_lst):
        # 人口密度を計算
        self.populations.calc(keypoints_lst)

        # 状態をリセット
        for person in self.persons:
            person.reset()

        for keypoints in keypoints_lst:
            target = keypoints.get_middle('Hip')
            max_person = None
            max_prob = 0.0
            for person in self.persons:
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
                new = Person(len(self.persons), keypoints)
                new.update(keypoints)
                self.persons.append(new)

        for person in self.persons:
            if person.is_reset():
                # アップデートされていない人にNoneを入力してアップデート
                person.update(None)
            elif person.is_deleted():
                person.delete()

        return self.persons, self.populations
