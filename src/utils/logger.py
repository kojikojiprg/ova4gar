from __future__ import absolute_import, division, print_function

import logging
import os
import time


def setup_logger(log_dir: str):
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_file = "{}.log".format(time_str)
    final_log_file = os.path.join(log_dir, log_file)
    head = "%(asctime)-15s %(message)s"
    # logging.basicConfig(format=head)
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    return logger
