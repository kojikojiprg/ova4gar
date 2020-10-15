from common import database
from display import heatmap
import numpy as np


class Indicator:
    def __init__(self, table_name, display_method, is_reset):
        self.table_name = table_name
        self.indicator_lst = []
        self.display = display_method
        self.is_reset_display = is_reset

    def append(self, frame_num, indicator_data):
        if frame_num >= len(self.indicator_lst):
            for _ in range(frame_num - len(self.indicator_lst) + 1):
                self.indicator_lst.append([])

        self.indicator_lst[frame_num].append(indicator_data)

    def make_heatmap(self):
        # カラーマップの最大値最小値を求めるために分布を取り出す
        distribution = []
        for indicator_datas in self.indicator_lst:
            for indicator_data in indicator_datas:
                data = indicator_data[-1]

                if data is None:
                    continue

                data = np.linalg.norm(data)
                distribution.append(data)

        # ヒートマップ作成
        hm = heatmap.Heatmap(distribution)

        # ヒートマップを計算
        copy = self.indicator_lst.copy()
        self.indicator_lst.clear()

        for indicator_datas in copy:
            for indicator_data in indicator_datas:
                # 該当テーブル取り出し
                for table in database.INDICATOR_TABLES:
                    if self.table_name == table.name:
                        break

                # フレーム番号のインデックスを取り出す
                for idx, key in enumerate(table.cols.keys()):
                    if key == 'Frame_No':
                        break

                frame_num = indicator_data[idx]
                data = indicator_data[-1]

                if data is None:
                    continue

                data = np.linalg.norm(data)
                cmap = hm.colormap(data)
                indicator_data = list(indicator_data)
                indicator_data.append(cmap)

                self.append(frame_num, indicator_data)
