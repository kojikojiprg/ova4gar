import numpy as np
from keypoint import KeypointsList
from particle_filter import ParticleFilter


class Person:
    def __init__(self, _id, keypoints):
        self.id = _id
        self.keypoints_lst = KeypointsList()
        self.keypoints_lst.append(keypoints)

        point = keypoints.get_middle('Ankle')
        self.state = State(point)
        self.pf = ParticleFilter(point[0], point[1])
        self.particles_lst = [self.pf.particles]

        self.verocity_lst = [0.]

    def update(self, point, keypoints):
        self.state.update(point)
        if point is not None:
            self.pf.predict(point[0], point[1])

        self.particles_lst.append(self.pf.particles)
        self.keypoints_lst.append(keypoints)

        self.verocity()

    def track_results(self):
        points = []
        for keypoints in self.keypoints_lst:
            if keypoints is not None:
                points.append(keypoints.get_middle('Ankle'))
            else:
                points.append(None)

        return points, self.particles_lst, self.verocity_lst

    def verocity(self):
        point = self.state.now
        prepoint = self.state.pre
        interval = self.state.interval
        dx = int((point[0] - prepoint[0]) / interval)
        dy = int((point[1] - prepoint[1]) / interval)

        speed = np.sqrt(dx**2 + dy**2)
        for _ in range(interval):
            self.speed_lst.append(speed)


class State:
    def __init__(self, point):
        self.now = point
        self.pre = None
        self.interval = 1

    def update(self, point):
        if self.now is None:
            self.interval += 1
        else:
            self.pre = self.now
            self.now = point
            self.interval = 1
