import numpy as np
import sqlite3 as sql
import os
import io


class DataBase:
    def __init__(self, path):
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write('')

        # Converts np.array to TEXT when inserting
        sql.register_adapter(np.ndarray, self._adapt_array)

        # Converts TEXT to np.array when selecting
        sql.register_converter('array', self._convert_array)

        self.conn = sql.connect(path, detect_types=sql.PARSE_DECLTYPES)

    def __del__(self):
        self.conn.commit()
        self.conn.close()

    def _adapt_array(self, arr):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sql.Binary(out.read())

    def _convert_array(self, text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out, allow_pickle=True).astype(float)

    def create_table(self, name, cols):
        c = self.conn.cursor()

        col_syntax = ''
        for col, typ in cols.items():
            col_syntax += '{} {}, '.format(col, typ)
        col_syntax = col_syntax[:-2]

        syntax = 'create table if not exists "{}"({})'.format(name, col_syntax)
        print(syntax)
        c.execute(syntax)

        self.conn.commit()

    def drop_table(self, name):
        c = self.conn.cursor()

        syntax = 'drop table if exists "{}"'.format(name)
        print(syntax)
        c.execute(syntax)

        self.conn.commit()

    def insert_data(self, name, cols, data):
        c = self.conn.cursor()

        col_syntax = ''
        data_syntax = ''
        for col in cols:
            col_syntax += '{}, '.format(col)
            data_syntax += '?,'
        col_syntax = col_syntax[:-2]
        name_syntax = '"{}" ({})'.format(name, col_syntax)

        data_syntax = data_syntax[:-1]
        data_syntax = '({})'.format(data_syntax)

        syntax = 'insert into {} values {}'.format(name_syntax, data_syntax)
        print(syntax)
        c.execute(syntax, data)

        self.conn.commit()

    def insert_datas(self, name, cols, datas):
        c = self.conn.cursor()

        col_syntax = ''
        data_syntax = ''
        for col in cols:
            col_syntax += '{}, '.format(col)
            data_syntax += '?,'
        col_syntax = col_syntax[:-2]
        name_syntax = '"{}" ({})'.format(name, col_syntax)

        data_syntax = data_syntax[:-1]
        data_syntax = '({})'.format(data_syntax)

        syntax = 'insert into {} values {}'.format(name_syntax, data_syntax)
        print(syntax)
        c.executemany(syntax, datas)

        self.conn.commit()

    def select(self, name, cols=None, where=None, array_size=1000):
        c = self.conn.cursor()

        col_syntax = '*'
        if cols is not None:
            col_syntax = ''
            for col in cols:
                col_syntax += '{}, '.format(col)
            col_syntax = col_syntax[:-2]

        where_syntax = ''
        if where is not None:
            where_syntax = 'where {}'.format(where)

        syntax = 'select {} from "{}" {}'.format(col_syntax, name, where_syntax)
        print(syntax)
        c.execute(syntax)

        data = c.fetchall()

        return data


class table:
    def __init__(self, name, cols):
        self.name = name
        self.cols = cols

    def index(self, key):
        for i, col in enumerate(self.cols.keys()):
            if key == col:
                return i
        return -1


TRACKING_TABLE = table(
    'Tracking',
    {
        'Person_ID': 'integer',
        'Frame_No': 'integer',
        'Keypoints': 'array',
        'Vector': 'array',
        'Average': 'array',
    }
)

PERSON_TABLE = table(
    'Person',
    {
        'Person_ID': 'integer',
        'Frame_No': 'integer',
        'Keypoints': 'array',
        'Face_Vector': 'array',
        'Body_Vector': 'array',
    }
)

DENSITY_TABLE = table(
    'Density',
    {
        'Frame_No': 'integer',
        'Cluster': 'array',
        'Count': 'integer'
    }
)

ATTENTION_TABLE = table(
    'Attention',
    {
        'Frame_No': 'integer',
        'Point': 'array',
        'Count': 'integer'
    }
)

GROUP_TABLE_LIST = [
    DENSITY_TABLE,
    ATTENTION_TABLE,
]