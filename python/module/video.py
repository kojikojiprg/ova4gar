import cv2


class Video:
    def __init__(self, in_path, out_path):
        self._video = cv2.VideoCapture(in_path)

        # image size
        self.width = int(self._video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.size = (self.width, self.height)

        # video info
        self.frame_count = int(self._video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = int(self._video.get(cv2.CAP_PROP_FPS))
        self.frame_num = self._video.get(cv2.CAP_PROP_FRAME_COUNT)

        # writer object
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self._writer = cv2.VideoWriter(out_path, fmt, self.frame_rate, self.size)

    def __del__(self):
        self._video.release()
        self._writer.release()
        cv2.destroyAllWindows()

    def read(self):
        _, frame = self._video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
        return frame

    def writer(self, frames):
        for frame in frames:
            self._writer.write(frame)
