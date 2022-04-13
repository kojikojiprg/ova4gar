import os
import pprint
import sys
from types import SimpleNamespace

import torch

sys.path.append("./submodules/")
from hrnet.lib.config import cfg, check_config, update_config
from hrnet.lib.dataset import make_test_dataloader
from hrnet.lib.fp16_utils.fp16util import network_to_half
from hrnet.lib.utils.utils import create_logger, get_model_summary


class Extractor:
    def __init__(self, cfg_path: str, opts=None):
        args = SimpleNamespace(**{"cfg": cfg_path, "opts": opts})

        # update config
        update_config(cfg, args)
        check_config(cfg)

        self.logger, final_output_dir, tb_log_dir = create_logger(
            cfg, cfg_path, "valid"
        )

        self.logger.info(pprint.pformat(args))
        self.logger.info(cfg)

        # cudnn related setting
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

        model = eval("models." + cfg.MODEL.NAME + ".get_pose_net")(cfg, is_train=False)

        dump_input = torch.rand((1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE))
        self.logger.info(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))

        if cfg.FP16.ENABLED:
            model = network_to_half(model)

        if cfg.TEST.MODEL_FILE:
            self.logger.info("=> loading model from {}".format(cfg.TEST.MODEL_FILE))
            model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
        else:
            model_state_file = os.path.join(final_output_dir, "model_best.pth.tar")
            self.logger.info("=> loading model from {}".format(model_state_file))
            model.load_state_dict(torch.load(model_state_file))

        self.model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
        self.model.eval()

        data_loader, test_dataset = make_test_dataloader(cfg)
