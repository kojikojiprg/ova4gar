from logging import Logger
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torchvision
from numpy.typing import NDArray

from hrnet.lib.config import cfg, update_config
from hrnet.lib.core.function import get_final_preds
from hrnet.lib.models import pose_hrnet
from hrnet.lib.utils.transforms import get_affine_transform


class HRNetDetecter:
    def __init__(self, cfg_path: str, logger: Logger, opts: list = []):
        # update config
        args = SimpleNamespace(**{"cfg": cfg_path, "opts": opts})
        args.modelDir = ""
        args.logDir = ""
        args.dataDir = ""
        args.prevModelDir = ""
        self.cfg = cfg
        update_config(self.cfg, args)

        self.logger: Logger = logger

        # cudnn related setting
        torch.backends.cudnn.benchmark = self.cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = self.cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = self.cfg.CUDNN.ENABLED

        self.box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True, progress=False
        )

        pose_model = pose_hrnet.get_pose_net(self.cfg, is_train=False)

        self.logger.info("=> loading model from {}".format(self.cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(self.cfg.TEST.MODEL_FILE))

        if torch.cuda.is_available():
            self.device = "cuda"
            self.box_model.cuda()
            gpus = [int(i) for i in self.cfg.GPUS.split(",")]
            self.pose_model = torch.nn.DataParallel(pose_model, device_ids=gpus)
            self.pose_model.cuda()
        else:
            self.device = "cpu"
            self.box_model.cpu()
            self.pose_model.cpu()

        self.box_model.eval()
        self.pose_model.eval()

    def __del__(self):
        # release memory
        del (
            self.box_model,
            self.pose_model,
            self.logger,
            self.cfg,
            self.transforms,
            self.parser,
        )

    def predict(self, image: NDArray):
        # object detection box
        image_t = (
            torch.from_numpy(image / 255.0).permute(2, 0, 1).float().to(self.device)
        )
        pred_boxes = self._get_person_detection_boxes(image_t, threshold=0.9)

        # pose estimation
        if len(pred_boxes) >= 1:
            preds = []
            for box in pred_boxes:
                center, scale = self._box_to_center_scale(
                    box, cfg.MODEL.IMAGE_SIZE[0], self.cfg.MODEL.IMAGE_SIZE[1]
                )
                image_pose = (
                    image.copy()
                    if cfg.DATASET.COLOR_RGB
                    else image[:, :, [2, 1, 0]].copy()
                )
                pose_preds = self._get_pose_estimation_prediction(
                    image_pose, center, scale
                )
                preds.append(pose_preds)

        return preds

    def _get_person_detection_boxes(self, img, threshold=0.5):
        pred = self.box_model(img)
        pred_classes = [
            i for i in list(pred[0]["labels"].cpu().numpy())
        ]  # Get the Prediction Score
        pred_boxes = [
            [(i[0], i[1]), (i[2], i[3])]
            for i in list(pred[0]["boxes"].detach().cpu().numpy())
        ]  # Bounding boxes
        pred_score = list(pred[0]["scores"].detach().cpu().numpy())
        if not pred_score or max(pred_score) < threshold:
            return []
        # Get list of index with score greater than threshold
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        pred_boxes = pred_boxes[: pred_t + 1]
        pred_classes = pred_classes[: pred_t + 1]

        person_boxes = []
        for idx, box in enumerate(pred_boxes):
            if pred_classes[idx] == 1:  # class is person
                person_boxes.append(box)

        return person_boxes

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

    def _get_pose_estimation_prediction(self, image, center, scale):
        # pose estimation transformation
        rotation = 0
        trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
        model_input = cv2.warpAffine(
            image,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR,
        )
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # pose estimation inference
        model_input = transform(model_input).unsqueeze(0)
        # switch to evaluate mode
        self.pose_model.eval()
        with torch.no_grad():
            # compute output heatmap
            output = self.pose_model(model_input)
            preds, _ = get_final_preds(
                cfg,
                output.clone().cpu().numpy(),
                np.asarray([center]),
                np.asarray([scale]),
            )

        return preds
