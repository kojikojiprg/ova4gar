import numpy as np
import cv2


class ParticleFilter:
    def __init__(self):
        pass

    def likelihood(self, x, y, func, image, w=30, h=30):
        x1 = max(0, x - w / 2)
        y1 = max(0, y - h / 2)
        x2 = min(image.shape[1], x + w / 2)
        y2 = min(image.shape[0], y + h / 2)

        region = image[y1:y2, x1:x2]
        count = region[func(region)].size

        return (float(count) / image.size) if count > 0 else 0.0001

    def init_particles(self, func, image):
        mask = image.copy()
        mask[not func(mask)] = 0

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) <= 0:
            return None

        max_contour = max(contours, key=cv2.contourArea)
        max_rect = np.array(cv2.boundingRect(max_contour))
        max_rect = max_rect[:2] + max_rect[2:] / 2

        weight = self.likelihood(max_rect[0], max_rect[1], func, image)

        particles = np.ndarray((500, 3), dtype=np.float32)
        particles[:] = [max_rect[0], max_rect[1], weight]

        return particles

    def resample(self, particles):
        tmp_particles = particles.copy()

        weights = particles[:, 2].cumsum()
        last_weight = weights[weights.shape[0] - 1]

        for i in range(particles.shape[0]):
            weight = np.random.rand() * last_weight
            particles[i] = tmp_particles[(weights > weight).argmax()]
            particles[i][2] = 1.0

    def predict(self, particles, variance=13.0):
        particles[:, 0] += np.random.randn((particles.shape[0])) * variance
        particles[:, 1] += np.random.randn((particles.shape[0])) * variance

    def weight(self, particles, func, image):
        for i in range(particles.shape[0]):
            particles[i][2] = self.likelihood(particles[i][0], particles[i][1], func, image)
        sum_weight = particles[:, 2].sum()
        particles[:, 2] *= (particles.shape[0] / sum_weight)

    def measure(self, particles):
        x = (particles[:, 0] * particles[:, 2]).sum()
        y = (particles[:, 1] * particles[:, 2]).sum()
        weight = particles[:, 2].sum()
        return x / weight, y / weight

    particle_filter_cur_frame = 0

    def particle_filter(self, particles, func, image, max_frame=10):
        global particle_filter_cur_frame
        if image[func(image)].size <= 0:
            if particle_filter_cur_frame >= max_frame:
                return None, -1, -1
            particle_filter_cur_frame = min(particle_filter_cur_frame + 1, max_frame)
        else:
            particle_filter_cur_frame = 0
            if particles is None:
                particles = self.init_particles(func, image)

        if particles is None:
            return None, -1, -1

        self.resample(particles)
        self.predict(particles)
        self.weight(particles, func, image)
        x, y = self.measure(particles)

        return particles, x, y
