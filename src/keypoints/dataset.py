import sys

import torch

sys.path.append("src")
from utils.video import Capture


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, video_cap: Capture):
        self.cap = video_cap

    def __len__(self):
        return self.cap.frame_count

    def __getitem__(self, idx):
        return self.cap.read(idx)


def make_test_dataloader(video_cap: Capture):
    dataset = TestDataset(video_cap)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    )

    return data_loader, dataset
