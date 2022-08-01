import argparse
import os
import sys

sys.path.append("src")
from analysis.attention import AttentionAnalyzer
from utility.logger import logger


def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rn",
        "--room_num",
        required=True,
        type=str,
        help="operating room number",
    )
    parser.add_argument(
        "-s",
        "--surgery_num",
        required=True,
        type=str,
        help="surgery number of each room",
    )
    parser.add_argument("-c", "--cfg_path", type=str, default="config/group.yaml")
    parser.add_argument("-ms", "--ma_size", type=int, default=1800)
    parser.add_argument("-pp", "--peak_prominence", type=float, default=0.3)
    parser.add_argument("-ph", "--peak_height", type=float, default=1.5)
    parser.add_argument("-th", "--trough_height", type=float, default=0.5)
    parser.add_argument("-mfn", "--margin_frame_num", type=int, default=900)
    parser.add_argument("-ft", "--frame_total", type=int, default=54000)

    return parser.parse_args()


def main():
    args = _setup_parser()
    analyzer = AttentionAnalyzer(args.room_num, args.surgery_num, args.cfg_path, logger)
    fig_path = os.path.join(
        "data", "attention", "image", f"{args.room_num}_{args.surgery_num}.pdf"
    )
    analyzer.extract_results(
        args.ma_size,
        args.peak_prominence,
        args.peak_height,
        args.trough_height,
        fig_path,
    )

    analyzer.crop_videos_random(args.margin_frame_num, args.frame_total)


if __name__ == "__main__":
    main()
