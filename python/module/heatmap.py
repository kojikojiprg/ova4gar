import numpy as np


class Heatmap:
    def __init__(self, keypoints_lst):
        self.keypoints_lst = keypoints_lst
        self.verocity_map = self.verocity()
        self.move_hand_map = self.move_hand()

    def _calc_args(self, distribution):
        xmax = np.nanmax(distribution)
        xmin = np.nanmin(distribution)
        half = (xmax - xmin) / 2
        inclination = 255 / half
        xmid = half + xmin
        return xmin, xmax, xmid, inclination

    def _colormap(self, x, xmin, xmax, xmid, inclination):
        if x <= xmid:
            r = 0
            g = inclination * (x - xmin)
            b = inclination * (xmid - x)
        else:
            r = inclination * (x - xmid)
            g = inclination * (xmax - x)
            b = 0
        return (int(r), int(g), int(b))

    def verocity(self):
        lst = [np.nan]
        for i in range(len(self.keypoints_lst) - 1):
            now = self.keypoints_lst[i]
            nxt = self.keypoints_lst[i + 1]
            if now is None or nxt is None:
                lst.append(np.nan)
                continue
            now = now.get_middle('Ankle')
            nxt = nxt.get_middle('Ankle')

            d = nxt - now
            vero = np.linalg.norm(d, ord=2)
            lst.append(vero)

        xmin, xmax, xmid, inclination = self._calc_args(lst)
        ret_lst = [None]
        for i in range(len(self.keypoints_lst) - 1):
            now = self.keypoints_lst[i]
            nxt = self.keypoints_lst[i + 1]
            if lst[i] is np.nan:
                ret_lst.append(None)
            else:
                now = now.get_middle('Ankle')
                nxt = nxt.get_middle('Ankle')
                ret_lst.append((
                    tuple(now),
                    tuple(nxt),
                    self._colormap(lst[i], xmin, xmax, xmid, inclination)))

        return ret_lst

    def move_hand(self):
        ret_lst = []
        xmin, xmax, xmid, inclination = self._calc_args([0., np.pi])
        for kp in self.keypoints_lst:
            if kp is None:
                ret_lst.append(None)
                continue
            mid_shoulder = kp.get_middle('Shoulder')
            mid_hip = kp.get_middle('Hip')
            mid_ankle = kp.get_middle('Ankle')

            # 体軸ベクトルとノルム
            axis = mid_shoulder - mid_hip
            norm_axis = np.linalg.norm(axis, ord=2)

            ankle = 0.
            for side in ('R', 'L'):
                elbow = kp.get(side + 'Elbow', ignore_confidence=True)
                wrist = kp.get(side + 'Wrist', ignore_confidence=True)

                # 前肢ベクトルとノルム
                vec = wrist - elbow
                norm = np.linalg.norm(vec, ord=2)

                # 体軸と前肢の角度(左右の大きい方を選択する)
                angle = max(ankle, np.arccos(np.dot(axis, vec) / (norm_axis * norm)))

            ret_lst.append((
                mid_ankle,
                self._colormap(angle, xmin, xmax, xmid, inclination)))

        return ret_lst
