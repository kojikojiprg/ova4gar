import argparse


def _common_parser():
    parser = argparse.ArgumentParser()

    # requires
    parser.add_argument(
        "-rn",
        "--room_num",
        required=True,
        type=str,
        help="operating room number",
    )

    # options
    parser.add_argument(
        "-c",
        "--cfg_path",
        type=str,
        default="config/demo_config.yaml",
        help="config file path.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="gpu number",
    )
    parser.add_argument(
        "-k",
        "--keypoint",
        default=False,
        action="store_true",
        help="with keypoint extraction",
    )
    parser.add_argument(
        "-i",
        "--individual",
        default=False,
        action="store_true",
        help="with idividual analyzation",
    )
    parser.add_argument(
        "-g",
        "--group",
        default=False,
        action="store_true",
        help="without group analyzation",
    )
    parser.add_argument(
        "-v",
        "--video",
        default=False,
        action="store_true",
        help="with writing video",
    )
    return parser


def surgery_parser():
    parser = _common_parser()

    # requires
    parser.add_argument(
        "-s",
        "--surgery_num",
        required=True,
        type=str,
        help="surgery number of each room",
    )

    # options
    parser.add_argument(
        "-ex", "--expand_name", type=str, default="", help="'passing' or 'attention'"
    )

    return parser.parse_args()


def file_parser():
    parser = _common_parser()

    # requires
    parser.add_argument("video_path", type=str, help="video file path")
    parser.add_argument(
        "data_dir", type=str, help="data directory path where results are saved"
    )

    return parser.parse_args()
