import datetime
import gc
import os
from glob import glob
from logging import Logger
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import openpyxl
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy import signal
from tqdm import tqdm
from utility.activity_loader import get_data_dirs, load_group
from utility.functions import moving_average
from utility.json_handler import load
from utility.video import Capture, Writer, concat_field_with_frame, get_size
from visualize.group import GroupVisualizer
from visualize.individual import write_field as ind_write_field
from visualize.keypoint import write_frame as kps_write_frame

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 40
plt.rcParams["xtick.direction"] = "in"  # x axis in
plt.rcParams["ytick.direction"] = "in"  # y axis in


class AttentionAnalyzer:
    def __init__(self, room_num: str, surgery_num: str, cfg_path: str, logger: Logger):
        self._room_num = room_num
        self._surgery_num = surgery_num
        with open(cfg_path, "r") as f:
            self._grp_cfg = yaml.safe_load(f)

        self._field = cv2.imread("image/field.png")

        self._logger = logger
        self._grp_vis = GroupVisualizer(["attention"])

        self._vertex_result: List[Tuple[int, float, str, int]] = []

    @property
    def vertex_result(self) -> List[Tuple[int, float, str, int]]:
        return self._vertex_result

    def _find_vertexs(
        self,
        max_val_ma: NDArray,
        prominence: float = 0.2,
        height_peak: float = 1.5,
        height_trough: float = 1.0,
    ) -> Tuple[NDArray, NDArray]:
        peaks = signal.find_peaks(
            max_val_ma, prominence=prominence, height=height_peak
        )[0]
        troughs = signal.find_peaks(
            max_val_ma * -1, prominence=prominence, height=(None, -height_trough)
        )[0]

        return peaks, troughs

    def _count_member(
        self,
        keypoints_data_lst: List[List[Dict[str, Any]]],
    ):
        self._logger.info("=> counting numbers of member")
        member_nums = []
        for keypoints_data in tqdm(keypoints_data_lst):
            frame_total = max([kps["frame"] for kps in keypoints_data])
            member_nums_per_data = [0 for _ in range(frame_total)]
            for kps in keypoints_data:
                frame_num = kps["frame"]
                member_nums_per_data[frame_num - 1] += 1
            member_nums += member_nums_per_data

        return member_nums

    def _save_plot(self, max_val_ma, peaks, troughs, member_nums_ma, fig_path):
        self._logger.info(f"=> saving plot figure to {fig_path}")

        fig = plt.figure(figsize=(20, 5))
        fig.subplots_adjust(left=0.05, right=0.94, bottom=0.23, top=0.95)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        ax1.plot(max_val_ma, label="max")
        ax1.scatter(peaks, max_val_ma[peaks], color="tab:orange", s=100)
        ax1.scatter(troughs, max_val_ma[troughs], color="tab:green", s=100)
        xticks = range(0, len(max_val_ma) + 1800, 1800 * 60)
        ax1.set_xticks(xticks, [t // (1800 * 60) for t in xticks])
        margin = len(max_val_ma) // 100
        ax1.set_xlim((-margin, len(max_val_ma) + margin))
        ax1.set_ylim((0, 4.0))
        ax1.set_xlabel("Hours")
        ax1.set_ylabel("Max of GA")

        ax2.plot(member_nums_ma, color="tab:red", linestyle=":")
        ax2.set_ylim((0, 10))
        ax2.set_ylabel("Member")

        plt.savefig(fig_path)

    def extract_results(
        self,
        ma_size: int = 1800,
        prominence: float = 0.2,
        height_peak: float = 1.5,
        height_trough: float = 1.0,
        fig_path: str = None,
    ):
        data_dirs = get_data_dirs(self._room_num, self._surgery_num)
        self._logger.info(f"=> data directories: {data_dirs}")

        heatmaps = []
        keypoints_data_lst = []
        for data_dir in data_dirs:
            self._logger.info(f"=> loading attention result from {data_dir}")
            json_path = os.path.join(data_dir, ".json", "group.json")
            group = load_group(
                json_path,
                self._grp_cfg,
                self._field,
                self._logger,
                only_data_loading=True,
            )
            attention_dict = group.attention
            heatmaps += list(attention_dict.values())

            self._logger.info(f"=> loading keypoint data from {data_dir}")
            json_path = os.path.join(data_dir, ".json", "keypoints.json")
            keypoints_data_lst.append(load(json_path))

            del group, attention_dict
            gc.collect()

        # max of group attention
        max_val = np.max(np.max(heatmaps, axis=1), axis=1)
        max_val_ma = moving_average(max_val, ma_size)

        # member number
        member_nums = self._count_member(keypoints_data_lst)
        member_nums_ma = moving_average(member_nums, ma_size)

        # find vertexs
        peaks, troughs = self._find_vertexs(
            max_val_ma,
            prominence,
            height_peak,
            height_trough,
        )
        vertexs = peaks.tolist() + troughs.tolist()

        self._save_plot(max_val_ma, peaks, troughs, member_nums_ma, fig_path)

        for vtx in vertexs:
            vertex_shape = "Peak" if vtx in peaks else "Trough"
            self._vertex_result.append(
                (vtx, max_val_ma[vtx], vertex_shape, member_nums_ma[vtx])
            )

    def _calc_video_position(
        self, margin_frame_num: int, frame_total: int = 54000
    ) -> List[Tuple[int, int, int]]:
        ret = []
        for (vertex_frame_num, _, _, _) in self.vertex_result:
            # add margin
            s_frame_num = max(1, vertex_frame_num - margin_frame_num)
            e_frame_num = vertex_frame_num + margin_frame_num

            # calc file num and frame num
            s_file_num = s_frame_num // frame_total + 1
            e_file_num = e_frame_num // frame_total + 1
            s_frame_num = s_frame_num % frame_total + 1
            e_frame_num = (e_frame_num % frame_total + 1) + (
                e_file_num - s_file_num
            ) * frame_total
            ret.append((s_file_num, s_frame_num, e_frame_num))

        return ret

    def save_excel(self, margin_frame_num: int, frame_total: int, excel_path: str):
        video_pos = self._calc_video_position(margin_frame_num, frame_total)

        cols = [
            "File Name",
            "Start Time",
            "End Time",
            "Vertex Shape",
            "Max GA-Value",
            "Number of Member",
            "Status",
            "Locations",
            "Events",
            "Remarkes",
        ]
        df = pd.DataFrame(columns=cols)

        sheet_name = f"{self._room_num}_{self._surgery_num}"
        self._logger.info(f"=> writing excel file to {excel_path}, sheet: {sheet_name}")
        for i, (_, ga_val, vertex_shape, member_num) in enumerate(
            tqdm(self.vertex_result)
        ):
            s_file_num, s_frame_num, e_frame_num = video_pos[i]

            # write dataframe
            file_name = f"{s_file_num:02d}_{s_frame_num:05d}_{e_frame_num:05d}.mp4"
            s_frame_num = ((s_file_num - 1) * frame_total + s_frame_num) // 30
            e_frame_num = ((s_file_num - 1) * frame_total + e_frame_num) // 30
            start_time = str(datetime.timedelta(seconds=s_frame_num)).format("%H:%M:%S")
            end_time = str(datetime.timedelta(seconds=e_frame_num)).format("%H:%M:%S")
            df.loc[i] = [
                file_name,  # File Name
                start_time,  # Start Time
                end_time,  # End Time
                vertex_shape,  # Vertex Shape
                ga_val,  # Max GA-Value
                member_num,  # Number of Member
                "",  # Status
                "",  # Locations
                "",  # Events
                "",  # Remarkes
            ]

        # write dataframe for excel
        if os.path.exists(excel_path):
            # delete sheet if same sheet exists
            workbook = openpyxl.load_workbook(filename=excel_path)
            if sheet_name in workbook.sheetnames:
                workbook.remove(workbook[sheet_name])
                workbook.save(excel_path)
            workbook.close()

            # over write excel file
            with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a") as writer:
                df.to_excel(writer, sheet_name, index=False, header=True)
        else:
            # create and write excel file
            df.to_excel(writer, sheet_name, index=False, header=True)

        del df

    def _load_jsons(self, data_dir):
        self._logger.info(f"=> load json files from {data_dir}")
        json_path = os.path.join(data_dir, ".json", "keypoints.json")
        kps_data = load(json_path)
        json_path = os.path.join(data_dir, ".json", "individual.json")
        ind_data = load(json_path)
        json_path = os.path.join(data_dir, ".json", "group.json")
        grp_data = load(json_path)
        return kps_data, ind_data, grp_data

    def _load_video(self, s_file_num):
        self._logger.info(f"=> load surgery {s_file_num:02d}.mp4")
        video_path = os.path.join(
            "video", self._room_num, self._surgery_num, f"{s_file_num:02d}.mp4"
        )
        return Capture(video_path)

    def crop_videos(self, margin_frame_num: int, frame_total: int):
        # delete previous files
        self._logger.info("=> delete files extracted previous process")
        for data_dir in sorted(
            glob(os.path.join("data", self._room_num, self._surgery_num, "*"))
        ):
            if data_dir.endswith("passing") or data_dir.endswith("attention"):
                continue
            for p in glob(os.path.join(data_dir, "video", "attention", "*.mp4")):
                if os.path.isfile(p):
                    os.remove(p)

        pre_s_file_num = 0
        video_pos = self._calc_video_position(margin_frame_num, frame_total)
        for s_file_num, s_frame_num, e_frame_num in video_pos:
            if pre_s_file_num < s_file_num:
                # load next video and json files
                data_dir = os.path.join(
                    "data", self._room_num, self._surgery_num, f"{s_file_num:02d}"
                )
                kps_data, ind_data, grp_data = self._load_jsons(data_dir)
                cap = self._load_video(s_file_num)
                # calc output size
                tmp_frame = cap.read()[1]
                size = get_size(tmp_frame, self._field)

            # create video writer
            out_path = os.path.join(
                data_dir,
                "video",
                "attention",
                f"{s_file_num:02d}_{s_frame_num}_{e_frame_num}.mp4",
            )
            wrt = Writer(out_path, cap.fps, size)

            # write video
            cap.set_pos_frame_count(s_frame_num - 1)
            self._logger.info(
                f"=> writing video file_num: {s_file_num}, frame: {s_frame_num}->{e_frame_num}"
            )
            for frame_num in tqdm(range(s_frame_num, e_frame_num)):
                ret, frame = cap.read()

                if not ret:
                    del kps_data, ind_data, grp_data, cap

                    # load next video and json files
                    s_file_num += 1
                    data_dir = os.path.join(
                        "data", self._room_num, self._surgery_num, f"{s_file_num:02d}"
                    )

                    if os.path.exists(data_dir):
                        kps_data, ind_data, grp_data = self._load_jsons(data_dir)
                        cap = self._load_video(s_file_num)
                        _, frame = cap.read()
                    else:
                        break

                frame_num %= frame_total
                frame = kps_write_frame(frame, kps_data, frame_num)
                field_tmp = ind_write_field(ind_data, self._field.copy(), frame_num)
                field_tmp = self._grp_vis.write_field(
                    "attention", frame_num, grp_data, field_tmp
                )
                frame = concat_field_with_frame(frame.copy(), field_tmp)
                wrt.write(frame)

            del wrt
            pre_s_file_num = s_file_num

        del kps_data, ind_data, grp_data, cap
        gc.collect()

    def crop_videos_random(
        self,
        margin_frame_num: int,
        frame_total: int,
        video_num: int = 10,
        seed: int = 128,
    ):
        random_data_dir = os.path.join(
            "data",
            self._room_num,
            self._surgery_num,
            "attention",
            "random",
        )
        # set random seed
        np.random.seed(seed)

        # delete previous files
        self._logger.info("=> delete files extracted previous process")
        for p in glob(os.path.join(random_data_dir, "*.mp4")):
            if os.path.isfile(p):
                os.remove(p)

        # extract not overlapped frame
        video_pos = self._calc_video_position(margin_frame_num, frame_total)
        not_overlapped_pos = []
        files = get_data_dirs(self._room_num, self._surgery_num)
        for file_path in files:
            file_num = int(os.path.basename(file_path))
            pos_lst = [pos for pos in video_pos if pos[0] == file_num]
            if len(pos_lst) == 0:
                continue

            pre_pos = pos_lst[0]
            for pos in pos_lst[1:-1]:
                pre_e_frame_num = pre_pos[2]
                s_frame_num = pos[1]
                if s_frame_num - pre_e_frame_num > margin_frame_num * 2:
                    middle_frame_num = (
                        s_frame_num - pre_e_frame_num
                    ) // 2 + pre_e_frame_num
                    not_overlapped_pos.append(
                        (
                            file_num,
                            middle_frame_num - margin_frame_num,
                            middle_frame_num + margin_frame_num,
                        )
                    )
                pre_pos = pos

        # random choice
        idx = np.random.choice(len(not_overlapped_pos), video_num, replace=False)
        random_pos = np.array(not_overlapped_pos)[idx]

        pre_s_file_num = 0
        for s_file_num, s_frame_num, e_frame_num in random_pos:
            if pre_s_file_num != s_file_num:
                # load next video and json files
                data_dir = os.path.join(
                    "data", self._room_num, self._surgery_num, f"{s_file_num:02d}"
                )
                kps_data, ind_data, grp_data = self._load_jsons(data_dir)
                cap = self._load_video(s_file_num)
                # calc output size
                tmp_frame = cap.read()[1]
                size = get_size(tmp_frame, self._field)

            # create video writer
            out_path = os.path.join(
                random_data_dir,
                f"random_{s_file_num:02d}_{s_frame_num}_{e_frame_num}.mp4",
            )
            wrt = Writer(out_path, cap.fps, size)

            # write video
            cap.set_pos_frame_count(s_frame_num - 1)
            self._logger.info(
                f"=> writing video file_num: {s_file_num}, frame: {s_frame_num}->{e_frame_num}"
            )
            for frame_num in tqdm(range(s_frame_num, e_frame_num)):
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num %= frame_total
                frame = kps_write_frame(frame, kps_data, frame_num)
                field_tmp = ind_write_field(ind_data, self._field.copy(), frame_num)
                field_tmp = self._grp_vis.write_field(
                    "attention", frame_num, grp_data, field_tmp
                )
                frame = concat_field_with_frame(frame.copy(), field_tmp)
                wrt.write(frame)

            del wrt
            pre_s_file_num = s_file_num

        del kps_data, ind_data, grp_data, cap
        gc.collect()
