import cv2
import numpy as np
import torch
from data.video import letterbox  # from unitrack
from utility.video import Capture  # from this project


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, video_cap: Capture, img_size=[640, 480]):
        self.cap = video_cap
        self.cap.set_pos_frame_count(0)
        vw, vh = self.cap.size

        self.width = img_size[0]
        self.height = img_size[1]
        self.w, self.h = self._get_size(vw, vh, self.width, self.height)

    def __len__(self):
        return self.cap.frame_count

    def __getitem__(self, idx):
        ret, img0 = self.cap.read()

        img = cv2.resize(img0, (self.w, self.h))

        # Padded resize
        img, _, _, _ = letterbox(img, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1]
        img = np.ascontiguousarray(img, dtype=np.float32)

        return img, img0

    @staticmethod
    def _get_size(vw, vh, dw, dh):
        wa, ha = float(dw) / vw, float(dh) / vh
        a = min(wa, ha)
        return int(vw * a), int(vh * a)


def make_test_dataloader(video_cap: Capture):
    dataset = TestDataset(video_cap)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    )

    return data_loader, dataset
