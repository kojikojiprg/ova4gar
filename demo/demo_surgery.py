import os
from glob import glob

from api.inference import InferenceModel
from api.parser import surgery_parser


def main():
    args = surgery_parser()

    video_dir = os.path.join("video", args.room_num, args.date)
    video_paths = sorted(glob(os.path.join(video_dir, "*.mp4")))

    data_dirs = []
    for video_path in video_paths:
        name = os.path.basename(video_path).replace(".mp4", "")
        data_dir = os.path.join("data", args.room_num, args.date, name)
        data_dirs.append(data_dir)
        os.makedirs(data_dir, exist_ok=True)

    model = InferenceModel(args)

    for video_path, data_dir in zip(video_paths, data_dirs):
        model.inference(video_path, data_dir)


if __name__ == "__main__":
    main()
