class AverageMeter(object):
    """Stores the summation and counts the number to compute the average value.
    """
    def __init__(self):
        self._sum = 0
        self._count = 0

    @property
    def avg(self):
        return self._sum / self._count if self._count != 0 else 0

    @property
    def count(self):
        return self._count

    @property
    def sum(self):
        return self._sum

    def update(self, value, n=1):
        self._sum += value * n
        self._count += n

    def reset(self):
        self._sum = 0
        self._count = 0
