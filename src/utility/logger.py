from __future__ import absolute_import, division, print_function

import logging


def setup_logger():
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger
