import argparse
import sys

sys.path.append("src")
from analysis.passing import PassingAnalyzer
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
    parser.add_argument(
        "-c", "--cfg_path", type=str, default="config/passing/pass_train.yaml"
    )
    parser.add_argument("-td", "--th_duration", type=int, default=10)
    parser.add_argument("-ti", "--th_interval", type=int, default=30)
    parser.add_argument("-mfn", "--margin_frame_num", type=int, default=60)

    return parser.parse_args()


def main():
    args = _setup_parser()
    analyzer = PassingAnalyzer(args.cfg_path, logger)
    results = analyzer.extract_results(
        args.room_num, args.surgery_num, args.th_duration, args.th_interval
    )
    analyzer.crop_videos(
        args.room_num, args.surgery_num, results, args.margin_frame_num
    )


if __name__ == "__main__":
    main()
