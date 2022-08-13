import gc
import os
from logging import Logger
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy import stats
from tqdm import tqdm
from utility.activity_loader import get_data_dirs, load_group
from utility.excel_handler import save
from utility.json_handler import load
from utility.video import Capture, Writer, concat_field_with_frame, get_size
from visualize.group import GroupVisualizer, Heatmap
from visualize.individual import write_field as ind_write_field
from visualize.keypoint import write_frame as kps_write_frame

sns.set()
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 40
plt.rcParams["xtick.direction"] = "in"  # x axis in
plt.rcParams["ytick.direction"] = "in"  # y axis in


class AttentionScore:
    def __init__(self, room_num: str, surgery_num: str, cfg_path: str, logger: Logger):
        self._room_num = room_num
        self._surgery_num = surgery_num
        with open(cfg_path, "r") as f:
            self._grp_cfg = yaml.safe_load(f)
        with open(self._grp_cfg["attention"]["object_path"], "r") as f:
            self._object_points = yaml.safe_load(f)[room_num][surgery_num]

        self._field = cv2.imread("image/field.png")

        self._logger = logger
        self._grp_vis = GroupVisualizer(["attention"])

        self._scores: Dict[str, Dict[int, float]] = {}

    @staticmethod
    def _calc_score(
        object_point: Tuple[int, int],
        attention_data: Dict[int, List[Dict[str, Any]]],
        sigma: int = 50,
    ) -> Dict[int, float]:
        px = object_point[0]
        py = object_point[1]

        value_dict: dict = {}
        for frame_num, data in tqdm(attention_data.items()):
            if frame_num not in value_dict:
                value_dict[frame_num] = {"values": [], "weights": []}

            for item in data:
                x, y = item["point"]
                val = item["value"]
                gauss = np.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * sigma**2))

                value_dict[frame_num]["values"].append(val)
                value_dict[frame_num]["weights"].append(gauss + 1e-10)

        results: Dict[int, float] = {}
        for frame_num, value in value_dict.items():
            if len(value["weights"]) > 0:
                results[frame_num] = np.average(
                    value["values"], weights=value["weights"]
                )
            else:
                results[frame_num] = np.nan

        return results

    def calc_score(self, sigma: int = 50):
        data_dirs = []
        for data_dir in get_data_dirs(self._room_num, self._surgery_num, "attention"):
            if "random" not in data_dir:
                data_dirs.append(data_dir)
        self._logger.info(f"=> data directories: {data_dirs}")

        attention_dict = {}
        for data_dir in data_dirs:
            self._logger.info(f"=> loading attention result from {data_dir}")
            num = os.path.basename(data_dir)
            json_path = os.path.join(data_dir, ".json", "group.json")
            if os.path.exists(json_path):
                group = load_group(
                    json_path,
                    self._grp_cfg,
                    self._field,
                    self._logger,
                    only_data_loading=True,
                )
                attention_dict[num] = group.get("attention")

            del group
            gc.collect()

        self._logger.info("=> calcurating scores")
        for num, attention_data in attention_dict.items():
            for i, point in enumerate(self._object_points[num]):
                key = f"{num}_{i + 1}"
                self._scores[key] = self._calc_score(point, attention_data, sigma)

    def mannwhitneyu(self, excel_path: str):
        pvals = {}
        self._logger.info("=> testing Mann-Whiteney U")
        for i in range(3):
            key1 = f"{i + 1:02d}_{1}"
            key2 = f"04_{i + 1}"
            data1 = np.array(list(self._scores[key1].values()))
            data2 = np.array(list(self._scores[key2].values()))

            # drop nan
            data1 = data1[~np.isnan(data1)]
            data2 = data2[~np.isnan(data2)]

            _, p = stats.mannwhitneyu(
                data1, data2, alternative="greater", use_continuity=True
            )
            pvals[f"O_{i + 1}"] = [p]

        df = pd.DataFrame.from_dict(pvals, orient="index")
        sheet_name = f"{self._room_num}_{self._surgery_num}"
        self._logger.info(f"=> writing excel file to {excel_path}, sheet: {sheet_name}")
        save(excel_path, sheet_name, df, index=True, header=False)

    def _boxplot(self, df_lst: List[pd.DataFrame], fig_dir: str):
        fig = plt.figure(figsize=(5, 2))
        axs = [
            fig.add_axes((0.1, 0.1, 0.29, 0.85)),
            fig.add_axes((0.4, 0.1, 0.29, 0.85)),
            fig.add_axes((0.7, 0.1, 0.29, 0.85)),
        ]

        for j in range(3):
            sns.boxplot(
                x="variable", y="value", data=df_lst[j], showfliers=False, ax=axs[j]
            )
            axs[j].set_ylabel("")
            axs[j].set_xlabel("")
            axs[j].set_ylim((0, 1.5))

        axs[0].set_ylabel("Attention Score")
        axs[1].set_yticklabels([])
        axs[2].set_yticklabels([])

        box_path = os.path.join(
            fig_dir, f"box_{self._room_num}_{self._surgery_num}.pdf"
        )
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig(box_path)

    def _hist(
        self, df_lst: List[pd.DataFrame], labels_lst: List[List[str]], fig_dir: str
    ):
        fig = plt.figure(figsize=(5, 2))
        axs = [
            fig.add_axes((0.06, 0.1, 0.26, 0.85)),
            fig.add_axes((0.39, 0.1, 0.26, 0.85)),
            fig.add_axes((0.72, 0.1, 0.26, 0.85)),
        ]

        for j in range(3):
            sns.histplot(
                data=df_lst[j], x="value", hue="variable", bins=20, kde=True, ax=axs[j]
            )

            axs[j].set_ylabel("")
            axs[j].set_xlabel("")
            axs[j].legend(
                labels=labels_lst[j][::-1],
                fontsize=7,
                handlelength=0.5,
                handletextpad=0.2,
            )

        box_path = os.path.join(
            fig_dir, f"hist_{self._room_num}_{self._surgery_num}.pdf"
        )
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig(box_path)

    def save_plot(self, fig_dir: str):
        os.makedirs(fig_dir, exist_ok=True)
        self._logger.info(f"=> saving plot figure to {fig_dir}")

        df_lst = []
        labels_lst = []
        for i in range(3):
            data_dict = {}
            key1 = f"{i + 1:02d}_{1}"
            key2 = f"04_{i + 1}"
            data1 = list(self._scores[key1].values())
            data2 = list(self._scores[key2].values())

            label1 = f"O{i % 3 + 1}-A"
            label2 = f"O{i % 3 + 1}-N/A"
            labels = [label1, label2]
            labels_lst.append(labels)

            data_dict[label1] = data1
            data_dict[label2] = data2

            df = pd.DataFrame.from_dict(data_dict)
            df = pd.melt(df)
            df = df.dropna()
            df_lst.append(df)

        self._boxplot(df_lst, fig_dir)
        self._hist(df_lst, labels_lst, fig_dir)

    def _load_jsons(self, data_dir):
        self._logger.info(f"=> load json files from {data_dir}")
        json_path = os.path.join(data_dir, ".json", "keypoints.json")
        kps_data = load(json_path)
        json_path = os.path.join(data_dir, ".json", "individual.json")
        ind_data = load(json_path)
        json_path = os.path.join(data_dir, ".json", "group.json")
        grp_data = load(json_path)
        return kps_data, ind_data, grp_data

    def _write_score(
        self,
        value: float,
        object_point: NDArray,
        field: NDArray,
        heatmap: Heatmap,
        max_radius: int = 20,
        alpha: float = 0.9,
    ):
        if np.isnan(value):
            value = 0.0
        color = heatmap.colormap(value)

        copy = field.copy()

        # calc radius of circle
        max_value = heatmap.xmax
        radius = int(value / max_value * max_radius)
        if radius == 0:
            radius = 1

        # plot the heatmap circle
        cv2.circle(copy, tuple(object_point), radius, color, thickness=-1)

        # plot the center of object
        cv2.circle(copy, tuple(object_point), 3, (0, 0, 0), -1)

        # plot the value text
        point = np.array(object_point) - [50, 10]
        cv2.putText(
            copy, f"{value:.1f}", tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2
        )

        field = cv2.addWeighted(copy, alpha, field, 1 - alpha, 0)

        return field

    def _write_frame(
        self,
        frame_num: int,
        frame: NDArray,
        kps_data,
        ind_data,
        grp_data,
        score: float,
        object_point: NDArray,
        heatmap: Heatmap,
    ):
        frame = kps_write_frame(frame, kps_data, frame_num)
        field_tmp = ind_write_field(ind_data, self._field.copy(), frame_num)
        field_tmp = self._grp_vis.write_field(
            "attention", frame_num, grp_data, field_tmp
        )
        field_tmp = self._write_score(score, object_point, field_tmp, heatmap)
        frame = concat_field_with_frame(frame.copy(), field_tmp)
        return frame

    def save_videos(self, frame_total: int = 1800):
        for i in range(3):
            key1 = f"{i + 1:02d}_{1}"
            key2 = f"04_{i + 1}"
            data1 = self._scores[key1]
            data2 = self._scores[key2]
            obj1 = self._object_points[f"{i + 1:02d}"][0]
            obj2 = self._object_points["04"][i]
            label1 = f"O{i % 3 + 1}-A"
            label2 = f"O{i % 3 + 1}-NA"
            heatmap = Heatmap([0, np.nanmax(list(data1.values()))])

            # load jsons
            data_dir = os.path.join(
                "data", self._room_num, self._surgery_num, "attention"
            )
            kps1, ind1, grp1 = self._load_jsons(os.path.join(data_dir, f"{i + 1:02d}"))
            kps2, ind2, grp2 = self._load_jsons(os.path.join(data_dir, "04"))

            # video capture
            video_dir = os.path.join(
                "video", self._room_num, self._surgery_num, "attention"
            )
            cap1 = Capture(os.path.join(video_dir, f"{i + 1:02d}.mp4"))
            cap2 = Capture(os.path.join(video_dir, "04.mp4"))
            size = get_size(cap1.read()[1], self._field)
            cap1.set_pos_frame_count(0)

            # video writer
            out_path1 = os.path.join(data_dir, f"{i + 1:02d}", "video", f"{label1}.mp4")
            wrt1 = Writer(out_path1, cap1.fps, size)
            out_path2 = os.path.join(data_dir, "04", "video", f"{label2}.mp4")
            wrt2 = Writer(out_path2, cap2.fps, size)
            self._logger.info(f"=> writing video {out_path1}, {out_path2}")
            for frame_num in tqdm(range(frame_total)):
                frame_num += 1
                _, frame1 = cap1.read()
                _, frame2 = cap2.read()

                frame1 = self._write_frame(
                    frame_num, frame1, kps1, ind1, grp1, data1[frame_num], obj1, heatmap
                )
                frame2 = self._write_frame(
                    frame_num, frame2, kps2, ind2, grp2, data2[frame_num], obj2, heatmap
                )

                wrt1.write(frame1)
                wrt2.write(frame2)

            del kps1, ind1, grp1
            del kps2, ind2, grp2
            del cap1, cap2, wrt1, wrt2
            gc.collect()
