import numpy as np


class Heatmap:
    def __init__(self, min_v, max_v):
        self.max = max_v
        self.min = min_v
        half = (max_v - min_v) / 2
        self.inclination = 255 / half
        self.mid = half + min_v

    def calc(self, distribution):
        distribution = np.array(distribution)
        shape = list(distribution.shape)
        shape.append(3)

        distribution = np.ravel(distribution)
        rslt = []
        for x in distribution:
            rslt.append(self.colormap(x))

        rslt = np.reshape(rslt, shape).tolist()
        return rslt

    def colormap(self, x):
        if x <= self.mid:
            r = 0
            g = self.inclination * (x - self.min)
            b = self.inclination * (self.mid - x)
        else:
            r = self.inclination * (x - self.mid)
            g = self.inclination * (self.max - x)
            b = 0
        return (int(r), int(g), int(b))
