from logging import Logger
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torchvision
from numpy.typing import NDArray
from PIL import Image
from torchvision import transforms

from hrnet.lib.config import cfg, update_config
from hrnet.lib.core.function import get_final_preds
from hrnet.lib.models import pose_hrnet
from hrnet.lib.utils.transforms import get_affine_transform


class HRNetDetecter:
    def __init__(self, cfg_path: str, logger: Logger, device: str, opts: list = []):
        # update config
        args = SimpleNamespace(**{"cfg": cfg_path, "opts": opts})
        args.modelDir = ""
        args.logDir = ""
        args.dataDir = ""
        args.prevModelDir = ""
        self.cfg = cfg
        update_config(self.cfg, args)

        self.logger: Logger = logger
        self.device = device

        # cudnn related setting
        torch.backends.cudnn.benchmark = self.cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = self.cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = self.cfg.CUDNN.ENABLED

        self.box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True, progress=False
        ).to(device)

        self.logger.info("=> loading model from {}".format(self.cfg.TEST.MODEL_FILE))
        self.pose_model = pose_hrnet.get_pose_net(self.cfg, is_train=False)
        self.pose_model.load_state_dict(torch.load(self.cfg.TEST.MODEL_FILE))
        self.pose_model.to(device)

        self.box_model.eval()
        self.pose_model.eval()

        self.box_transform = transforms.Compose([transforms.ToTensor()])
        self.pose_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __del__(self):
        # release memory
        del self.box_model, self.pose_model, self.logger, self.cfg

    def predict(self, images: NDArray):
        # object detection box
        pred_boxes_all_batch = self._get_person_detection_boxes(images, threshold=0.7)

        # pose estimation
        centers, scales = [], []
        pose_images = []
        box_indices = [0]
        for img, pred_boxes in zip(images, pred_boxes_all_batch):
            box_indices.append(box_indices[-1] + len(pred_boxes))
            if len(pred_boxes) >= 1:
                for box in pred_boxes:
                    center, scale = self._box_to_center_scale(
                        box, cfg.MODEL.IMAGE_SIZE[0], self.cfg.MODEL.IMAGE_SIZE[1]
                    )
                    centers.append(center)
                    scales.append(scale)
                pose_images.append(
                    (
                        img.copy()
                        if cfg.DATASET.COLOR_RGB
                        else img[:, :, [2, 1, 0]].copy()
                    )
                )

        pred_poses = self._get_pose_estimation_prediction(pose_images, centers, scales)

        return np.split(pred_poses, box_indices)

    def _get_person_detection_boxes(self, imgs, threshold):
        transformed_imgs = [
            self.box_transform(Image.fromarray(img)).to(self.device) for img in imgs
        ]
        preds = self.box_model(transformed_imgs)

        results = []
        for pred in preds:
            pred_classes = [
                i for i in list(pred["labels"].cpu().numpy())
            ]  # Get the Prediction Score
            pred_boxes = [
                [(i[0], i[1]), (i[2], i[3])]
                for i in list(pred["boxes"].detach().cpu().numpy())
            ]  # Bounding boxes
            pred_score = list(pred["scores"].detach().cpu().numpy())

            if not pred_score or max(pred_score) < threshold:
                continue

            # Get list of index with score greater than threshold
            pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
            pred_boxes = pred_boxes[: pred_t + 1]
            pred_classes = pred_classes[: pred_t + 1]

            person_boxes = []
            for idx, box in enumerate(pred_boxes):
                if pred_classes[idx] == 1:  # class is person
                    person_boxes.append(box)
            results.append(person_boxes)

        return results

    def _box_to_center_scale(self, box, model_image_width, model_image_height):
        center = np.zeros((2), dtype=np.float32)

        bottom_left_corner = box[0]
        top_right_corner = box[1]
        box_width = top_right_corner[0] - bottom_left_corner[0]
        box_height = top_right_corner[1] - bottom_left_corner[1]
        bottom_left_x = bottom_left_corner[0]
        bottom_left_y = bottom_left_corner[1]
        center[0] = bottom_left_x + box_width * 0.5
        center[1] = bottom_left_y + box_height * 0.5

        aspect_ratio = model_image_width * 1.0 / model_image_height
        pixel_std = 200

        if box_width > aspect_ratio * box_height:
            box_height = box_width * 1.0 / aspect_ratio
        elif box_width < aspect_ratio * box_height:
            box_width = box_height * aspect_ratio
        scale = np.array(
            [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
            dtype=np.float32,
        )
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def _get_pose_estimation_prediction(self, imgs, centers, scales):
        # pose estimation transformation
        model_inputs = []
        for img, center, scale in zip(imgs, centers, scales):
            trans = get_affine_transform(center, scale, 0, cfg.MODEL.IMAGE_SIZE)
            # Crop smaller image of people
            model_input = cv2.warpAffine(
                img,
                trans,
                (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
                flags=cv2.INTER_LINEAR,
            )

            # hwc -> 1chw
            model_input = self.pose_transform(model_input)  # .unsqueeze(0)
            model_inputs.append(model_input)

        # n * 1chw -> nchw
        model_inputs = torch.stack(model_inputs)

        with torch.no_grad():
            # compute output heatmap
            output = self.pose_model(model_inputs.to(self.device))
            coors, scores = get_final_preds(
                cfg,
                output.clone().cpu().numpy(),
                np.asarray(centers),
                np.asarray(scales),
            )

        return np.concatenate((coors, scores), axis=2)
