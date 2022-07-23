import datetime
import gc
import os
from glob import glob
from logging import Logger
from typing import List, Tuple

import cv2
import numpy as np
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
plt.rcParams["font.size"] = 32
plt.rcParams["xtick.direction"] = "in"  # x axis in
plt.rcParams["ytick.direction"] = "in"  # y axis in


class AttentionAnalyzer:
    def __init__(self, cfg_path: str, logger: Logger):
        with open(cfg_path, "r") as f:
            self._grp_cfg = yaml.safe_load(f)

        self._field = cv2.imread("image/field.png")

        self._logger = logger
        self._grp_vis = GroupVisualizer(["attention"])

    def _find_peaks(
        self,
        heatmaps: List[NDArray],
        ma_size: int = 1800,
        prominence: float = 0.2,
        height: float = 1.5,
        height_inv: float = 1.0,
        fig_path: str = None,
    ) -> List[Tuple[int, float, str]]:
        max_val = np.max(np.max(heatmaps, axis=1), axis=1)
        max_val_ma = moving_average(max_val, ma_size)

        peaks = signal.find_peaks(max_val_ma, prominence=prominence, height=height)[0]
        peaks_inv = signal.find_peaks(
            max_val_ma * -1, prominence=prominence, height=(None, -height_inv)
        )[0]
        if len(peaks) == 0:
            return []
        peaks_all = sorted(peaks.tolist() + peaks_inv.tolist())

        self._save_plot(max_val_ma, peaks, peaks_inv, fig_path)

        result_lst: List[Tuple[int, float, str]] = []
        for peak in peaks_all:
            peak_shape = "Peak" if peak in peaks else "Trough"
            result_lst.append((peak, max_val_ma[peak], peak_shape))

        return result_lst

    def _save_plot(self, max_val_ma, peaks, peaks_inv, fig_path):
        self._logger.info(f"=> saving plot figure to {fig_path}")
        fig = plt.figure(figsize=(20, 5))
        fig.subplots_adjust(left=0.04, right=0.99, bottom=0.17, top=0.96)

        plt.plot(max_val_ma, label="max")
        plt.scatter(peaks, max_val_ma[peaks], color="tab:orange", s=64)
        plt.scatter(peaks_inv, max_val_ma[peaks_inv], color="tab:green", s=64)

        xticks = range(0, len(max_val_ma), 1800 * 30)
        plt.xticks(xticks, [t // 1800 for t in xticks])

        margin = len(max_val_ma) // 100
        plt.xlim((-margin, len(max_val_ma) + margin))
        plt.ylim((0, 4.0))

        plt.xlabel("Minutes")
        plt.ylabel("Max of GA")

        plt.savefig(fig_path)

    def extract_results(
        self,
        room_num: str,
        surgery_num: str,
        ma_size: int = 1800,
        prominence: float = 0.2,
        height: float = 1.5,
        height_inv: float = 1.0,
        fig_path: str = None,
    ) -> List[Tuple[int, float, str]]:
        data_dirs = get_data_dirs(room_num, surgery_num)
        self._logger.info(f"=> data directories: {data_dirs}")

        heatmaps = []
        for data_dir in data_dirs:
            self._logger.info(f"=> load attention result from {data_dir}")
            json_path = os.path.join(data_dir, ".json", "group.json")
            if os.path.exists(json_path):
                group = load_group(
                    json_path,
                    self._grp_cfg,
                    self._field,
                    self._logger,
                    only_data_loading=True,
                )
                attention_dict = group.attention
                heatmaps += list(attention_dict.values())

                del group, attention_dict
                gc.collect()

        return self._find_peaks(
            heatmaps, ma_size, prominence, height, height_inv, fig_path
        )

    def _load_jsons(self, data_dir):
        self._logger.info(f"=> load json files from {data_dir}")
        json_path = os.path.join(data_dir, ".json", "keypoints.json")
        kps_data = load(json_path)
        json_path = os.path.join(data_dir, ".json", "individual.json")
        ind_data = load(json_path)
        json_path = os.path.join(data_dir, ".json", "group.json")
        grp_data = load(json_path)
        return kps_data, ind_data, grp_data

    def _load_video(self, room_num, surgery_num, s_file_num):
        self._logger.info(f"=> load surgery {s_file_num:02d}.mp4")
        video_path = os.path.join(
            "video", room_num, surgery_num, f"{s_file_num:02d}.mp4"
        )
        return Capture(video_path)

    def crop_videos(
        self,
        room_num: str,
        surgery_num: str,
        peak_results: List[Tuple[int, float, str]],
        margin_frame_num: int,
        excel_path: str,
        frame_total: int = 54000,
    ):
        # delete previous files
        self._logger.info("=> delete files extracted previous process")
        for data_dir in sorted(glob(os.path.join("data", room_num, surgery_num, "*"))):
            if data_dir.endswith("passing") or data_dir.endswith("attention"):
                continue
            for p in glob(os.path.join(data_dir, "video", "attention", "*.mp4")):
                if os.path.isfile(p):
                    os.remove(p)

        # prepair dataframe for excel
        cols = ["ファイル名", "開始時間", "終了時間", "ピーク形状", "GA値", "場所", "事象", "備考"]
        df = pd.DataFrame(columns=cols)

        pre_s_file_num = 0
        for i, (frame_num, ga_val, peak_shape) in enumerate(peak_results):
            # add margin
            s_frame_num = max(1, frame_num - margin_frame_num)
            e_frame_num = frame_num + margin_frame_num

            # calc file num and frame num
            s_file_num = s_frame_num // frame_total + 1
            e_file_num = e_frame_num // frame_total + 1
            s_frame_num = s_frame_num % frame_total + 1
            e_frame_num = (e_frame_num % frame_total + 1) + (
                e_file_num - s_file_num
            ) * frame_total

            # write dataframe
            file_name = f"{s_file_num:02d}_{s_frame_num:05d}_{e_frame_num:05d}.mp4"
            start_time = str(datetime.timedelta(seconds=s_frame_num // 30)).format(
                "%H:%M:%S"
            )
            end_time = str(datetime.timedelta(seconds=e_frame_num // 30)).format(
                "%H:%M:%S"
            )
            df.loc[i] = [
                file_name,  # ファイル名
                start_time,  # 開始時間
                end_time,  # 終了時間
                peak_shape,  # ピーク形状
                ga_val,  # GA値
                "",  # 場所
                "",  # 事象
                "",  # 備考
            ]

            if pre_s_file_num < s_file_num:
                # load next video and json files
                data_dir = os.path.join(
                    "data", room_num, surgery_num, f"{s_file_num:02d}"
                )
                kps_data, ind_data, grp_data = self._load_jsons(data_dir)
                cap = self._load_video(room_num, surgery_num, s_file_num)
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
                        "data", room_num, surgery_num, f"{s_file_num:02d}"
                    )

                    if os.path.exists(data_dir):
                        kps_data, ind_data, grp_data = self._load_jsons(data_dir)
                        cap = self._load_video(room_num, surgery_num, s_file_num)
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

        # write dataframe for excel
        sheet_name = f"{room_num}_{surgery_num}"
        self._logger.info(f"=> writing excel file to {excel_path}, sheet: {sheet_name}")
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a") as writer:
            df.to_excel(writer, sheet_name, index=False, header=True)

        del kps_data, ind_data, grp_data, cap, df
        gc.collect()
