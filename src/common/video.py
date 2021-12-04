import os

import cv2


class Capture:
    def __init__(self, in_path):
        if not os.path.isfile(in_path):
            raise ValueError(f"not exist file {in_path}")

        self._cap = cv2.VideoCapture(in_path)

        # video info
        self.fps = int(self._cap.get(cv2.CAP_PROP_FPS))
        self.frame_num = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.hw = (self.h, self.w)

    def __del__(self):
        self._cap.release()
        cv2.destroyAllWindows()

    def set_pos_frame(self, begin_sec):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, begin_sec * self.fps)

    def read(self):
        ret, frame = self._cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
            return frame
        else:
            return None


class Writer:
    def __init__(self, out_path, fps, size, fmt="h264"):
        out_dir = os.path.dirname(out_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # writer object
        fmt = cv2.VideoWriter_fourcc(fmt[0], fmt[1], fmt[2], fmt[3])
        self._writer = cv2.VideoWriter(out_path, fmt, fps, size)

    def __del__(self):
        self._writer.release()

    def write(self, frames):
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB to BGR
            self._writer.write(frame)
