from common import database


def calc_save(persons, db):
    datas = []

    for person in persons:
        frame_num = person.start_frame_num

        for keypoints, average, vector in zip(person.keypoints_lst, person.average_lst, person.vector_lst):
            # 腰と足首の差分を計算
            if keypoints is not None:
                hip = keypoints.get_middle('Hip')
                ankle = keypoints.get_middle('Ankle')
                diff_y = (ankle - hip)[1]
                average[1] += diff_y

            datas.append((
                person.id,
                frame_num,
                average,
                vector))
            frame_num += 1

    # データベースに書き込み
    table = database.VECTOR_TABLE
    db.drop_table(table.name)
    db.create_table(table.name, table.cols)
    db.insert_datas(
        table.name,
        list(table.cols.keys()),
        datas)
