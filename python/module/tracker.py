from person import Person


class Tracker:
    def __init__(self, keypoints_frame_lst):
        self.keypoints_frame_lst = keypoints_frame_lst
        self.persons = []

        keypoints_lst = self.keypoints_frame_lst[0]
        for i, keypoints in enumerate(keypoints_lst):
            self.persons.append(Person(i, keypoints))

    def track(self):
        for frame_num, keypoints_lst in enumerate(self.keypoints_frame_lst):
            targets = keypoints_lst.get_middle_points('Hip')

            # 状態をリセット
            for person in self.persons:
                person.reset()

            for target, keypoints in zip(targets, keypoints_lst):
                max_person = None
                max_prob = 0.0
                for person in self.persons:
                    if person.is_updated():
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
                    new = Person(len(self.persons), keypoints, frame_num=frame_num)
                    new.update(keypoints)
                    self.persons.append(new)

            for person in self.persons:
                # アップデートされていない人にNoneを入力してアップデート
                if not person.is_updated():
                    person.update(None)

        return self.persons
