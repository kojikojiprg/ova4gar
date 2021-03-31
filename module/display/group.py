from common.json import GROUP_FORMAT
import numpy as np
import cv2

    def make_heatmap(self):
        for k in self.indicator_dict.keys():
            if HEATMAP_SETTING_DICT[k][0]:
                # ヒートマップを作成する場合
                distribution = []
                for data in self.indicator_dict[k]:
                    # ヒートマップの対象となる列を取得
                    data_idx = HEATMAP_SETTING_DICT[k][1]
                    distribution.append(data[data_idx])
                # ヒートマップ作成
                self.heatmap_dict[k] = Heatmap(distribution)
            else:
                self.heatmap_dict[k] = None

    def display(self, k, frame_num, field):
        if HEATMAP_SETTING_DICT[k][0]:
            field = DISPLAY_DICT[k](
                self.get_data(k, frame_num), field, self.heatmap_dict[k])
        else:
            field = DISPLAY_DICT[k](
                self.get_data(k, frame_num), field)

        return field

def display_density(datas, field, heatmap, min_r=8):
    for data in datas:
        point = np.average(data[1], axis=0).astype(int)
        r = min_r + len(data[1])
        color = heatmap.colormap(len(data[1]))
        cv2.circle(field, tuple(point), r, color, thickness=-1)
    return field


def display_attention(datas, field, heatmap, min_r=8):
    for data in datas:
        if data[1] is not None:
            for point in data[1]:
                # 視線が重なったところを黒色で表示
                cv2.circle(field, tuple(point.astype(int)), 5, (0, 0, 0), thickness=-1)

            # クラスターを表示
            point = np.average(data[1], axis=0).astype(int)
            r = min_r + len(data[1])
            color = heatmap.colormap(len(data[1]))
            cv2.circle(field, tuple(point), r, color, thickness=-1)
    return field


keys = list(GROUP_FORMAT.keys())
DISPLAY_DICT = {
    keys[0]: display_density,
    keys[1]: display_attention,
}

HEATMAP_SETTING_DICT = {
    # key: [is_heatmap, heatmap_data]
    keys[0]: [True, -1],
    keys[1]: [True, -1],
}
