import argparse


def _common_parser():
    parser = argparse.ArgumentParser()

    # requires
    parser.add_argument(
        "-rn",
        "--room_num",
        required=True,
        type=str,
        help="The room number of operating room",
    )

    # options
    parser.add_argument(
        "-c",
        "--cfg_path",
        type=str,
        default="config/config.yaml",
        help="Config file path.",
    )
    parser.add_argument(
        "-wk",
        "--without_keypoint",
        default=False,
        action="store_true",
        help="Without keypoint extraction.",
    )
    parser.add_argument(
        "-wi",
        "--without_individual",
        default=False,
        action="store_true",
        help="Without idividual analyzation.",
    )
    parser.add_argument(
        "-wg",
        "--without_group",
        default=False,
        action="store_true",
        help="Without group analyzation.",
    )
    return parser


def surgery_parser():
    parser = _common_parser()

    # requires
    parser.add_argument("-d", "--date", required=True, type=str, help="Date")

    return parser.parse_args()


def file_parser():
    parser = _common_parser()

    # requires
    parser.add_argument("video_path", type=str, help="Path of a raw video.")
    parser.add_argument(
        "data_dir", type=str, help="Path of data directory where results are saved."
    )

    return parser.parse_args()
