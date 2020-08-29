import cv2


class Video:
    def __init__(self, in_path):
        self._video = cv2.VideoCapture(in_path)

        # video info
        self.frame_count = int(self._video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = int(self._video.get(cv2.CAP_PROP_FPS))
        self.frame_num = int(self._video.get(cv2.CAP_PROP_FRAME_COUNT))

    def __del__(self):
        self._video.release()
        cv2.destroyAllWindows()

    def read(self):
        _, frame = self._video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
        return frame

    def write(self, frames, out_path, size):
        # writer object
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self._writer = cv2.VideoWriter(
            out_path, fmt, self.frame_rate, size)

        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB to BGR
            self._writer.write(frame)

        self._writer.release()
