from common import common


def calc_save(persons, db):
    datas = []

    for person in persons:
        frame_num = person.start_frame_num

        for average, vector in zip(person.average_lst, person.vector_lst):
            datas.append((
                person.id,
                frame_num,
                average,
                vector))
            frame_num += 1

    # データベースに書き込み
    db.drop_table(common.VECTOR_TABLE_NAME)
    db.create_table(common.VECTOR_TABLE_NAME, common.VECTOR_TABLE_COLS)
    db.insert_datas(
        common.VECTOR_TABLE_NAME,
        list(common.VECTOR_TABLE_COLS.keys()),
        datas)
