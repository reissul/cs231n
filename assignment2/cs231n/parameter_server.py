import numpy as np
import simplejson
from os.path import join, exists
from random import choice

class ParameterServer(object):
    def __init__(self, choices, logdir=None):
        self.choices = choices
        self.num_choices = np.prod(list(map(len, choices.values())))
        self.served = []
        self.logdir = logdir
        if logdir:
            self.fname = join(logdir, "served.json")
            self.load()
    def load(self):
        assert(self.logdir)
        if exists(self.fname):
            self.served = simplejson.load(open(self.fname))
    def save(self):
        assert(self.logdir)
        simplejson.dump(self.served, open(self.fname, "w"))
    def serve(self):
        assert(self.serving)
        while True:
            sample = dict([(k,choice(v)) for (k,v) in self.choices.items()])
            if self._is_new(sample):
                self.served.append(sample)
                self.save()
                return dict(sample.items())
    def _is_new(self, sample):
        for s in self.served:
            assert(sorted(sample.keys()) == sorted(s.keys()))
            dup = True
            for (k, v) in sample.items():
                o = s[k]
                if not isinstance(o, v.__class__) or \
                        (isinstance(o, (int, float)) and abs(o-v) > 1e-8) or \
                        (isinstance(o, str) and o != v):
                    dup = False
                    break
            # Found new differences, so it is a duplicate.
            if dup:
                return False
        # Found no duplicates, so it is new.
        return True
    @property
    def serving(self):
        return len(self.served) < self.num_choices
