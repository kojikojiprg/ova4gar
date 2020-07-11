from particle_filter import ParticleFilter


class Person:
    def __init__(self, _id, keypoints):
        self.id = _id
        self.keypoints_lst = [keypoints]

        point = keypoints.get_middle_ankle()
        self.pf = ParticleFilter(point[0], point[1])
        self.particles_lst = [self.pf.particles]

    def update(self, point, keypoints):
        if point is not None:
            self.pf.predict(point[0], point[1])

        self.particles_lst.append(self.pf.particles)
        self.keypoints_lst.append(keypoints)

    def track_rslts(self):
        points = []
        for keypoints in self.keypoints_lst:
            if keypoints is not None:
                points.append(keypoints.get_middle_ankle())
            else:
                points.append(None)

        return points, self.particles_lst
