import torch
from utility.video import Capture


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, video_cap: Capture):
        self.cap = video_cap
        self.cap.set_pos_frame_count(0)

    def __len__(self):
        return self.cap.frame_count

    def __getitem__(self, idx):
        ret, img = self.cap.read()
        return img


def make_test_dataloader(video_cap: Capture):
    dataset = TestDataset(video_cap)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    )

    return data_loader, dataset
