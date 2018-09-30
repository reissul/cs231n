def is_working(losses, thresh=10.):
    if len(losses) <= 1:
        return False
    ln = losses[-1]
    l0 = losses[0]
    if l0 < 1e-6:
        return True
    perc_dec = 100. * (l0 - ln) / l0
    return perc_dec > thresh # working if improves by thresh%
