from common import database


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
    table = database.VECTOR_TABLE
    db.drop_table(table.name)
    db.create_table(table.name, table.cols)
    db.insert_datas(
        table.name,
        list(table.cols.keys()),
        datas)
