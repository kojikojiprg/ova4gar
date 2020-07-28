from keypoint import KeypointsList
from particle_filter import ParticleFilter


class Person:
    def __init__(self, _id, keypoints):
        self.id = _id
        self.keypoints_lst = KeypointsList()
        self.keypoints_lst.append(keypoints)

        point = keypoints.get_middle('Ankle')
        self.pf = ParticleFilter(point[0], point[1])
        self.particles_lst = [self.pf.particles]

    def update(self, point, keypoints):
        if point is not None:
            self.pf.predict(point[0], point[1])

        self.particles_lst.append(self.pf.particles)
        self.keypoints_lst.append(keypoints)

    def vector(self, point):
