import argparse
import os
import sys

sys.path.append("src")
from analysis.attention_score import AttentionScore
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
    parser.add_argument("-sig", "--sigma", type=int, default=50)

    return parser.parse_args()


def main():
    args = _setup_parser()
    attention_score = AttentionScore(
        args.room_num, args.surgery_num, args.cfg_path, logger
    )
    fig_dir = os.path.join("data", "attention", "image", "plot")
    excel_path = os.path.join("data", "attention", f"mwu_result_sig{args.sigma}.xlsx")
    attention_score.calc_score(args.sigma)
    attention_score.mannwhitneyu(excel_path)
    attention_score.save_plot(fig_dir)


if __name__ == "__main__":
    main()
