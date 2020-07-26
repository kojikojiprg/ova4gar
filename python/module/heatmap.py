import numpy as np


class Heatmap:
    def __init__(self, keypoints_lst):
        self.keypoints_lst = keypoints_lst
        self.verocities = self.verocity()

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

            dx = nxt[0] - now[0]
            dy = nxt[1] - now[1]
            vero = np.sqrt(dx**2 + dy**2)
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
        pass
