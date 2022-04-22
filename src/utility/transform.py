import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import json_handler

homo = {
    "02": [
        np.float32([[426, 553], [494, 510], [773, 638], [718, 692]]),
        np.float32([[336, 320], [400, 320], [400, 514], [336, 514]]),
    ],
    "05": [
        np.float32([[378, 264], [910, 357], [926, 609], [91, 444]]),
        np.float32([[3, 90], [581, 90], [581, 400], [3, 400]]),
    ],
    "08": [
        np.float32([[737, 517], [799, 537], [583, 698], [509, 656]]),
        np.float32([[336, 320], [400, 320], [400, 514], [336, 514]]),
    ],
    "09": [
        np.float32([[738, 277], [801, 300], [567, 462], [497, 430]]),
        np.float32([[336, 320], [400, 320], [400, 514], [336, 514]]),
    ],
    "09_0304": [
        np.float32([[738, 277], [801, 300], [567, 462], [497, 430]]),
        np.float32([[356, 342], [422, 342], [422, 548], [356, 548]]),
    ],
}


class Homography:
    def __init__(self, p_src, p_dst, dst_size):
        self.M = cv2.getPerspectiveTransform(p_src, p_dst)
        self.size = dst_size[1::-1]

    def transform_image(self, src):
        return cv2.warpPerspective(src, self.M, self.size)

    def transform_point(self, point):
        point = np.append(point, 1)
        result = np.dot(self.M, point)
        return np.array(
            [
                result[0] / result[2],
                result[1] / result[2],
            ]
        )


class CameraCalibration:
    def __init__(self, json_path=None):
        self.mtx = None
        self.dist = None
        if json_path is not None:
            self.from_json(json_path)

    def from_json(self, json_path):
        data = json_handler.load(json_path)
        self.mtx = np.array(data["mtx"])
        self.dist = np.array(data["dist"])

    def to_json(self, json_path):
        data = {"mtx": self.mtx, "dist": self.dist}
        json_handler.dump(data, json_path)

    def fit(
        self,
        images_folder_path,
        corner_pattern=(10, 7),
        square_size=6.8,
        is_verbose=False,
    ):
        # termination criteria
        subpix_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            1000,
            0.01,
        )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.01)

        # calibration flags
        calibration_flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            + cv2.fisheye.CALIB_CHECK_COND
            + cv2.fisheye.CALIB_FIX_SKEW
        )

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((np.prod(corner_pattern), 3), np.float32)
        objp[:, :2] = np.indices(corner_pattern).T.reshape(-1, 2)
        objp *= square_size

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = glob.glob(images_folder_path + "/*.jpg")

        for fname in tqdm(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, corner_pattern)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(
                    gray, corners, (3, 3), (-1, -1), subpix_criteria
                )
                imgpoints.append(corners2)

                if is_verbose:
                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(img, corner_pattern, corners2, ret)
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.show()

        # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        #     objpoints, imgpoints, gray.shape[::-1], None, None
        # )

        objpoints = np.expand_dims(np.asarray(objpoints), -2)
        ret, mtx, dist, _, _ = cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            None,
            None,
            None,
            None,
            calibration_flags,
            criteria,
        )

        self.mtx = mtx
        self.dist = dist

    def transform(self, img, alpha=0.0, is_crop=True):
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (w, h), alpha, (w, h)
        )

        dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)

        if is_crop:
            x, y, w, h = roi
            dst = dst[y : y + h, x : x + w]

        return dst
