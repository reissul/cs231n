import inspect
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import simplejson
from numpy.random import randint
from os.path import join, exists
from random import choice

import torch.optim as optim
from visdom import Visdom

from cs231n.train_utils2 import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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

class Visualizer(object):
    def __init__(self, logdir, port=6006):
        self.vis = Visdom(port=port)
        assert self.vis.check_connection(), 'No connection could be formed quickly'
        self.vis.close()
        plt.close('all')
        self.matplot_win = None
        self.logdir = logdir
        self.data = {}
        self.windows = {}
        self.table_cols = None
        self.table_data = {}
    def __del__(self):
        plt.close('all')
        del self.vis
    def update(self, table=False):
        for model in os.listdir(self.logdir):
            if "model" not in model: continue
            # Load *new* data from the model directory.
            if model not in self.data:
                self.data[model] = {LOSSES: [], TRAIN_ACCS: [], VAL_ACCS: []}
                self.windows[model] = dict(loss=None, accs=None)
            x, y = {}, {}
            for k in [LOSSES, TRAIN_ACCS, VAL_ACCS]:
                fname = join(self.logdir, model, k)
                if not exists(fname): continue
                d = list(map(float, open(fname).readlines()))
                x[k], y[k] = [], []
                for i in range(len(self.data[model][k]), len(d)):
                    x[k].append(i)
                    y[k].append(d[i])
                    self.data[model][k].append(d[i])
            # Continue if no new data or not working.
            if not y[LOSSES] or not is_working(self.data[model][LOSSES]):
                continue
            # Visualize plots for this model.
            if LOSSES in x and x[LOSSES]:
                opts = {
                    "title": model,
                    #"xlabel": "iteration",
                    "ylabel": "loss",
                    "xtickmin": 0,
                    "xtickmax": (1 + len(self.data[model][LOSSES]) // 1000) * 1000,
                    "ytickmin": 0,
                    "ytickmax": 5
                }
                w = self.windows[model]["loss"]
                if not w:
                    self.windows[model]["loss"] = self.vis.line(Y=self.data[model][LOSSES], opts=opts)
                else:
                    self.vis.line(X=x[LOSSES], Y=y[LOSSES], opts=opts,
                                  win=w, update="append")                    
            if TRAIN_ACCS in x and x[TRAIN_ACCS]:
                opts = {
                    "title": model,
                    #"xlabel": "iteration",
                    "ylabel": "accuracy",
                    "xtickmin": 0,
                    "xtickmax": (1 + len(self.data[model][TRAIN_ACCS]) // 100) * 100,
                    "ytickmin": 0,
                    "ytickmax": 100,
                    "legend": ["Train", "Val"]
                }
                update = "append" if self.windows[model]["accs"] else None
                w = self.windows[model]["accs"]
                if not w:
                    data = list(zip(self.data[model][TRAIN_ACCS], self.data[model][VAL_ACCS]))
                    self.windows[model]["accs"] = self.vis.line(Y=data, opts=opts)
                else:
                    self.vis.line(X=list(zip(x[TRAIN_ACCS], x[VAL_ACCS])),
                                  Y=list(zip(y[TRAIN_ACCS], y[VAL_ACCS])),
                                  opts=opts, win=w, update="append")
            # Load table data for this model.
            if table:
                if self.data[model][TRAIN_ACCS]:
                    fname = join(self.logdir, model, "arch.json")
                    parameters = simplejson.load(open(fname))
                    id = int(model.replace("model", ""))
                    p = sorted(parameters.items())
                    if not self.table_cols:
                        self.table_cols = ["id"] + sorted(parameters.keys()) + \
                            ["loss", "train acc", "val acc"]
                    row = [id]
                    row += [parameters[col] for col in self.table_cols if col in parameters]
                    row += [self.data[model][LOSSES][-1]]
                    row += [self.data[model][TRAIN_ACCS][-1]]
                    row += [self.data[model][VAL_ACCS][-1]]
                    self.table_data[model] = row
        # Visualize table data for all models.
        if table and self.table_data.values():
            # Sort by val acc.
            data = sorted(self.table_data.values(), key=lambda x: -x[-1])
            # Update table in visom.
            N, M = np.array(data).shape
            fig, ax = plt.subplots()
            #fig.set_figheight(2*N/8) # ewww
            fig.set_figheight(20) # ewww
            fig.set_figwidth(20)
            fig.patch.set_visible(False)
            ax.axis('off')
            ax.axis('tight')
            ax.table(cellText=data, colLabels=self.table_cols, loc='center')
            self.matplot_win = self.vis.matplot(plt, win=self.matplot_win)

class HyperOpt(object):
    
    def __init__(self, choices, constructor, trainer, loader_train, loader_val, logdir,
                 max_training=3, vis=True, verbose=True, coarse_its=None,
                 fine_epochs=10, device=torch.device('cpu')):
        self.parameter_server = ParameterServer(choices, logdir)
        self.constructor = constructor
        self.trainer = trainer
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.logdir = logdir
        self.max_training = max_training
        self.vis = Visualizer(logdir) if vis else None
        self.verbose = verbose
        self.coarse_its = coarse_its
        self.fine_epochs = fine_epochs
        self.device = device
        
        self.coarse_trained = []
        self.fine_trained = []
        
        self.load()

    def load(self):
        for model_fname in os.listdir(self.logdir):
            if "model" not in model_fname: continue
            model_dir = join(self.logdir, model_fname)
            # Load the losses and continue if not working.
            fname = join(model_dir, LOSSES)
            if exists(fname):
                losses = list(map(float, open(fname).readlines()))
                if not is_working(losses):
                    continue
            # Load last checkpoint
            check_fnames = [v for v in os.listdir(model_dir) if "check" in v]
            if not check_fnames: continue
            check_fname = join(model_dir, sorted(check_fnames)[-1])
            check = Checkpoint()
            check.load(check_fname)
            check.logdir = model_dir
            if check.epoch < self.fine_epochs:
                self.coarse_trained.append(check)
            else:
                self.fine_trained.append(check)
        
    def optimize(self):
        fname = inspect.stack()[0][3]
        logger.debug('%s: entering' % (fname))
        fine = len(self.coarse_trained) >= self.max_training
        while self.parameter_server.serving or self.coarse_trained:
            # If the coarse search is done, move to fine.
            if not fine and (len(self.coarse_trained) == self.max_training \
                                 or not self.parameter_server.serving):
                fine = True
            # If the fine search is done, move to coarse.
            elif fine and not self.coarse_trained:
                fine = False
            # Step
            if not fine:
                self.coarse_step()
            else:
                self.fine_step()

    def coarse_step(self):
        fname = inspect.stack()[0][3]
        logger.debug('%s: starting' % (fname))
        
        # Sample parameters.
        parameters = self.parameter_server.serve()
                
        # Find the most aggressive learning rate that works.
        for lr in [10**v for v in range(-1, -4, -1)]:
            model = self.constructor(parameters, (3,32,32), 10) # TODO: don't hard code?
            model_name = "model%03d" % (len(os.listdir(self.logdir)) + 1)
            model_logdir = join(self.logdir, model_name)
            logger.debug('%s: %s with lr=%.3f' % (fname, model_name, lr))
            if not exists(model_logdir):
                os.makedirs(model_logdir)
            parameters["Rate"] = lr
            simplejson.dump(parameters, open(join(model_logdir, "arch.json"), "w"))
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                  nesterov=True)
            losses, checkpoint = self.trainer(self.loader_train, self.loader_val, model, optimizer, 
                                              model_logdir, verbose=self.verbose, vis=self.vis, epochs=1,
                                              iterations=self.coarse_its, eval_iterations=16,
                                              device=self.device)
            checkpoint.logdir = model_logdir
            if is_working(losses):
                self.coarse_trained.append(checkpoint)
                break
    
    def fine_step(self):
        fname = inspect.stack()[0][3]
        logger.debug('%s: starting' % (fname))
        check = sorted(self.coarse_trained, key=lambda x: x.acc)[-1]
        check = self.trainer(self.loader_train, self.loader_val, check.model, check.optimizer,
                             check.logdir, verbose=self.verbose, vis=self.vis,
                             epochs=self.fine_epochs, eval_iterations=16, device=self.device)
        self.fine_trained.append(check)
