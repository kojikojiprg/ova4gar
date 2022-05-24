import os

import torch
from utility.video import Capture


class KeypointDetaset(torch.utils.data.Dataset):
    def __init__(self, cap: Capture):
        self._cap = cap
        self._cap.set_pos_frame_count(0)

    def __len__(self):
        return self._cap.frame_count

    def __getitem__(self, idx):
        _, frame = self._cap.read()
        return idx + 1, frame


def make_data_loader(cap: Capture, batch_size):
    dataset = KeypointDetaset(cap)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )

    return loader
