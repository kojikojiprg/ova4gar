from glob import glob
import numpy as np
import cv2
import json


class Homography:
    def __init__(self, p_src, p_dst, dst_size):
        self.M = cv2.getPerspectiveTransform(p_src, p_dst)
        self.size = dst_size[1::-1]

    def transform_image(self, src):
        return cv2.warpPerspective(src, self.M, self.size)

    def transform_point(self, point):
        point = np.append(point, 1)
        result = np.dot(self.M, point)
        return np.array([
            result[0] / result[2],
            result[1] / result[2],
        ])


class CameraCalibration:
    def __init__(self, json_path=None):
        self.mtx = None
        self.dist = None
        if json_path is not None:
            data = json.load(json_path)
            self.mtx = data['mtx']
            self.dist = data['dist']

    def fit(self, images_folder_path):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = glob.glob(images_folder_path + '/*.jpg')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        self.mtx = mtx
        self.dist = dist

    def transform(self, img):
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.mtx,
            self.dist,
            (w, h), 1, (w, h))

        # undistort
        dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        return dst

    def to_json(self, json_path):
        data = {
            'mtx': self.mtx,
            'dist': self.dist
        }
        json.dump(data, json_path)
