"""This file contains code to record an average over time.

Adapted from MaskBit (https://github.com/markweberdev/maskbit, utils/meter.py), licensed under the
Apache License 2.0. See THIRD_PARTY_NOTICES.md.
"""


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count