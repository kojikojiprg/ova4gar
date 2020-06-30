import cv2


class Reader:
    def __init__(self, path):
        self._video = cv2.VideoCapture(path)

        # image size
        self.width = int(self._video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.size = (self.width, self.height)

        # video info
        self.frame_count = int(self._video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = int(self._video.get(cv2.CAP_PROP_FPS))

    def read(self):
        _, frame = self._video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
        return frame
