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
    parser.add_argument("-pp", "--peak_prominence", type=float, default=0.2)
    parser.add_argument("-ph", "--peak_height", type=float, default=1.5)
    parser.add_argument("-phi", "--peak_height_inv", type=float, default=1.0)
    parser.add_argument("-mfn", "--margin_frame_num", type=int, default=900)

    return parser.parse_args()


def main():
    args = _setup_parser()
    analyzer = AttentionAnalyzer(args.cfg_path, logger)
    results = analyzer.extract_results(
        args.room_num,
        args.surgery_num,
        args.ma_size,
        args.peak_prominence,
        args.peak_height,
        args.peak_height_inv,
    )
    excel_path = os.path.join(
        "data",
        "attention",
        f"ga_pr{args.peak_prominence}_h{args.peak_height}_hi{args.peak_height_inv}.xlsx",
    )
    analyzer.crop_videos(
        args.room_num, args.surgery_num, results, args.margin_frame_num, excel_path
    )


if __name__ == "__main__":
    main()
