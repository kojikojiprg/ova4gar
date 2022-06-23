import argparse
import os
import sys
from glob import glob

from tqdm import tqdm

sys.path.append("src")
from utility.json_handler import dump, load
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
    parser.add_argument("-tl", "--timebar_height", type=int, default=20)
    parser.add_argument("-ex", "--expand_name", type=str, default="")

    return parser.parse_args()


def main():
    args = _setup_parser()
    data_dirs = sorted(
        glob(
            os.path.join("data", args.room_num, args.surgery_num, args.expand_name, "*")
        )
    )
    logger.info(f"=> {data_dirs}")

    for data_dir in data_dirs:
        logger.info(data_dir)

        # keypoints
        json_path = os.path.join(data_dir, ".json", "keypoints.json")
        logger.info(f"=> loading {json_path}")
        kps_data = load(json_path)
        for kps_item in tqdm(kps_data, desc="keypoints"):
            kps = kps_item["keypoints"]
            for kp in kps:
                kp[1] -= args.timebar_height
        logger.info(f"=> writing {json_path}")
        dump(kps_data, json_path)

        # individual
        json_path = os.path.join(data_dir, ".json", "individual.json")
        logger.info(f"=> loading {json_path}")
        ind_data = load(json_path)
        for ind_item in tqdm(ind_data, desc="individual"):
            for kp in ind_item["keypoints"]:
                kp[1] -= args.timebar_height
            ind_item["position"][1] -= args.timebar_height
        logger.info(f"=> writing {json_path}")
        dump(ind_data, json_path)

        # group
        json_path = os.path.join(data_dir, ".json", "group.json")
        logger.info(f"=> loading {json_path}")
        grp_data = load(json_path)
        for pass_item in tqdm(grp_data["passing"], desc="passing"):
            pass_item["points"][0][1] -= args.timebar_height
            pass_item["points"][1][1] -= args.timebar_height
        for atte_item in tqdm(grp_data["attention"], desc="attention"):
            atte_item["point"][1] -= args.timebar_height
        logger.info(f"=> writing {json_path}")
        dump(grp_data, json_path)


if __name__ == "__main__":
    main()
