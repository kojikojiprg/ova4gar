import os
import cv2


class Video:
    def __init__(self, in_path):
        if not os.path.isfile(in_path):
            raise ValueError(f"not exist file {in_path}")

        self._video = cv2.VideoCapture(in_path)

        # video info
        self.fps = int(self._video.get(cv2.CAP_PROP_FPS))
        self.frame_num = int(self._video.get(cv2.CAP_PROP_FRAME_COUNT))

    def __del__(self):
        self._video.release()
        cv2.destroyAllWindows()

    def set_pos_frame(self, begin_sec):
        self._video.set(cv2.CAP_PROP_POS_FRAMES, begin_sec * self.fps)

    def read(self):
        ret, frame = self._video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
            return frame
        else:
            return None

    def write(self, frames, out_path, size):
        out_dir = os.path.dirname(out_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # writer object
        fmt = cv2.VideoWriter_fourcc("h", "2", "6", "4")
        self._writer = cv2.VideoWriter(out_path, fmt, self.fps, size)

        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB to BGR
            self._writer.write(frame)

        self._writer.release()
