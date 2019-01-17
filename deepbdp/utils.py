from functools import partial
import logging
from colorlog import ColoredFormatter

import torch
import torch.nn.functional as F

###############################################################################
# default device
###############################################################################
def set_device(device, logger):
    logger.warn('Attention!!! wrap torch.tensor(default device: [%s])', device)
    torch.tensor = partial(torch.tensor, device=torch.device(device))
    torch.eye = partial(torch.eye, device=torch.device(device))
    torch.zeros = partial(torch.zeros, device=torch.device(device))


###############################################################################
# logging
###############################################################################
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(name)s, %(asctime)s] %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'white,bold',
        'INFOV':    'cyan,bold',
        'WARNING':  'red',
        'ERROR':    'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
ch.setFormatter(formatter)


logging.addLevelName(logging.INFO + 1, 'INFOV')


def _infov(self, msg, *args, **kwargs):
    self.log(logging.INFO + 1, msg, *args, **kwargs)


logging.Logger.infov = _infov


def get_logger(alias):

    log = logging.getLogger(alias)
    log.setLevel(logging.DEBUG)
    log.handlers = []       # No duplicated handlers
    log.propagate = False   # workaround for duplicated logs in ipython
    log.addHandler(ch)
    return log


###############################################################################
# metric
###############################################################################
def seq_metric(input_seq, target_seq):
    """
    param:
        input_seq: [batch]
        target_seq: [batch]
    """
    l1_error = torch.abs(input_seq - target_seq).mean()
    l2_error = torch.sqrt(((input_seq - target_seq) * (input_seq - target_seq)).mean())
    return l1_error, l2_error