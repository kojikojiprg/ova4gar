from particle_filter import ParticleFilter


class Person:
    def __init__(self, _id, keypoints):
        self.id = _id
        self.pf = ParticleFilter()
        self.keypoints_lst = []
        self.now = None
        self.history = []

    def get_coordinate(self):
        return self.now[0], self.now[1]

    def append_point(self, point):
        self.history.append(point)
        self.now = point

    def calc_vector(self):
        pass
