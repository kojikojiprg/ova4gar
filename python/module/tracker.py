from person import Person


class Tracker:
    def __init__(self, initial_keypoints_lst):
        self.persons = []
        for i, keypoints in enumerate(initial_keypoints_lst):
            self.persons.append(Person(i, keypoints))

    def track(self, frame_num, keypoints_lst):
        targets = keypoints_lst.get_middle_points('Hip')

        # 状態をリセット
        for person in self.persons:
            person.reset()

        for target, keypoints in zip(targets, keypoints_lst):
            max_person = None
            max_prob = 0.0
            for person in self.persons:
                if not person.is_reset():
                    # アップデート済は飛ばす
                    continue

                # パーティクルが移動する確率を求める
                prob = person.probability(target, 0.0000000001)

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

        return self.persons
