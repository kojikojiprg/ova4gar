class HalfLine:
    def __init__(self, point, vector):
        self.x0 = point[0]
        self.y0 = point[1]
        self.a = vector[1] / vector[0]              # 傾き
        self.b = -1 * self.a * self.x0 + self.y0    # 切片

        # 象限
        if vector[0] > 0 and vector[1] >= 0:
            self.quadrant = 1
        elif vector[0] <= 0 and vector[1] > 0:
            self.quadrant = 2
        elif vector[0] < 0 and vector[1] <= 0:
            self.quadrant = 3
        elif vector[0] >= 0 and vector[1] < 0:
            self.quadrant = 4
        else:
            self.quadrant = 0

    def func(self, x):
        y = self.a * x + self.b
        return y

    def limit(self, x, y):
        # 半直線の先端を判定
        diffx = x - self.x0
        diffy = y - self.y0
        if self.quadrant == 1:
            return diffx > 0 and diffy >= 0
        elif self.quadrant == 2:
            return diffx <= 0 and diffy > 0
        elif self.quadrant == 3:
            return diffx < 0 and diffy <= 0
        elif self.quadrant == 4:
            return diffx >= 0 and diffy < 0
        else:
            return False

    def is_online(self, point):
        x = point[0]
        y = point[1]

        if int(self.func(x) - y) == 0:
            return self.limit(x, y)
        else:
            return False


def calc_cross(l1, l2):
    a_c = l1.a - l2.a
    d_b = l2.b - l1.b
    ad_bc = l1.a * l2.b - l1.b * l2.a

    if abs(a_c) < 1e-5 or abs(d_b) > 1e+5 or abs(ad_bc) > 1e+5:
        return None
    else:
        x = d_b / a_c
        y = ad_bc / a_c
        return x, y
