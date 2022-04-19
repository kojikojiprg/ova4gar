import sys

import torch

sys.path.append("src")
from utils.video import Capture


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, video_path: str):
        self.cap = Capture(video_path)
        assert self.cap.is_opened, f"{video_path} does not exist or is wrong file type."

    def __len__(self):
        return len(self.cap)

    def __getitem__(self, idx):
        self.cap.set_pos_frame_count(idx)
        return self.cap.read()


def make_test_dataloader(video_path: str):
    dataset = TestDataset(video_path)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    )

    return data_loader, dataset
