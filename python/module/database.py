import common as cm
import sqlite3 as sql
import os


class DataBase:
    def __init__(self, path):
        self.path = cm.db_dir + path + '.db'
        os.makedirs(self.path, exist_ok=True)

        self.conn = sql.connect(self.path)

    def __del__(self):
        self.conn.commit()
        self.conn.close()

    def _create_table(self, name, cols):
        c = self.conn.cursor()

        col_syntax = ''
        for col, typ in cols.items():
            col_syntax += '{} {}, '.format(col, typ)
        col_syntax = col_syntax[:-2]

        syntax = 'create table {}({})'.format(name, col_syntax)
        c.execute(syntax)

        self.conn.commit()

    def _insert_datas(self, name, datas):
        c = self.conn.cursor()

        datas_syntax = '('
        for _ in range(len(datas)):
            datas_syntax += '?,'
        datas_syntax = datas_syntax[:-1]

        syntax = 'insert into {} values {}'.format(name, datas_syntax)
        c.executemany(syntax, datas)

        self.conn.commit()
