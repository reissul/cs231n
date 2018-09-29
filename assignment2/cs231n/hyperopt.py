import inspect
import logging
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import queue

import torch
import torch.optim as optim
from visdom import Visdom

from cs231n.train_utils import *

logger = logging.getLogger(__name__)

class HyperOpt(object):
    """
    A hyperparameter optimizer given data, trainer, etc.
    
    Example usage:
    choices = {
      "FilterSize": [3, 5],
      "FilterCount": [8, 32, 128],
    }
    ho = HyperOpt(choices, construct_model, train, loader_train, loader_val)
    ho.optimize()
    """
    def __init__(self, choices, constructor, trainer, loader_train, loader_val,
                 max_active=None, coarse_its=None, fine_epochs=10, 
                 verbose=False, visualize=True, port=6006, 
                 device=torch.device('cpu')):
        """
        Construct a HyperOpt object.
        
        Arguments:
        - choices: Dictionary from parameter name to parameter choices.
        - constructor: Model constructor function. Must have the following API:
          - constructor(parameters, image_shape)
        - trainer: Model training function. Must have the following API:
          - trainer(model, loader_train, loader_val)
        - trainer_val: PyTorch DataLoader for training set.
        - loader_val: PyTorch DataLoader for validation set.
        - max_active: Maximum number of active parameter sets.
        - coarse_its: Number of training iterations during coarse phases.
        - fine_epochs: Number of training epochs during fine phases.
        - ...
        """
        # Specified parameters and derivatives.
        self.choices = choices
        self.num_choices = np.prod(list(map(len, choices.values())))
        self.constructor = constructor
        self.trainer = trainer
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.max_active = max_active
        self.coarse_its = coarse_its
        self.fine_epochs = fine_epochs
        self.verbose = verbose
        self.device = device
        
        # 
        self.samples = set([])
        self.active = queue.PriorityQueue() # begin with no active
        self.trained = queue.PriorityQueue() # begin with no trained
        self.table_cols = ["id"] + list(choices.keys()) + \
            ["learn-rate", "train-acc", "val-acc"]
        self.table_data = []

        self.vis = Visdom(port=port)
        assert self.vis.check_connection(), 'No connection could be formed quickly'
        self.vis.close()
        plt.close('all')
        self.matplot_win = None
        #self.loss_wins = []

    def optimize(self):
        """
        Optimize hyperparameters.
        """
        fname = inspect.stack()[0][3]
        logger.debug('%s:' % (fname))
        fine = False
        sampled_all = False
        while not (sampled_all and not self.active.qsize()):
            # If the coarse search is done, move to fine.
            if not fine and (self.active.qsize() == self.max_active or sampled_all):
                fine = True
            # If the fine search is done, move to coarse.
            elif fine and not self.active.qsize():
                fine = False
            # Step.
            self.coarse_step() if not fine else self.fine_step()
            sampled_all = len(self.samples) == self.num_choices

    def coarse_step(self):
        fname = inspect.stack()[0][3]
        # Sample parameters and check if this was sampled previously.
        logger.debug('%s:' % (fname))
        lrs, parameter_inds, parameter_vals = [], {}, {}
        for (name, vals) in self.choices.items():
            ind = randint(0, len(vals))
            parameter_inds[name] = ind
            parameter_vals[name] = self.choices[name][ind]
        sample = tuple(sorted(parameter_inds.items()))
        if sample in self.samples:
            return
        self.samples.add(sample)
        
        # Find the right learning rate (if there is one).
        lrs = []
        while len(lrs) < 4:
            lr = 10 ** np.random.randint(-6, 1)
            if lr in lrs: continue
            lrs.append(lr)
            parameter_vals["learn-rate"] = lr
            model = self.constructor(parameter_vals, (3,32,32), 10) # TODO: don't hard code?
            optimizer = optim.SGD(model.parameters(), lr=lr,
                                  momentum=0.9, nesterov=True)
            history = self.trainer(model, optimizer, self.loader_train,
                                   self.loader_val, epochs=1,
                                   its=self.coarse_its, eval_its=16,
                                   verbose=self.verbose, device=self.device)
            # Add results to table.
            self._add_to_table(parameter_vals, history)
            # If it's working, add it to our active set and break.
            if history.working:
                logger.info("%s:\tlr = %.2E" % (fname, lr))
                logger.info("%s:\t\tworking = %s" % (fname, history.working))
                self.active.put((history, model, optimizer, parameter_vals))
                break
            
    def fine_step(self):
        fname = inspect.stack()[0][3]
        logger.debug('%s:' % (fname))
        history, model, optimizer, parameter_vals = self.active.get_nowait()
        history = self.trainer(model, optimizer, self.loader_train,
                               self.loader_val, epochs=self.fine_epochs,
                               eval_its=16, verbose=self.verbose,
                               device=self.device, history=history, vis=self.vis)
        self._add_to_table(parameter_vals, history)
        self.trained.put((history, model, parameter_vals))

    def _add_to_table(self, parameters, history):
        train_acc, val_acc = history.train_acc, history.val_acc
        # First add to the table
        row = [history.id]
        row += [parameters[col] for col in self.table_cols if col in parameters]
        row += [train_acc, val_acc]
        self.table_data.append(row)
        # Then sort by val acc
        self.table_data.sort(key=lambda x: -x[-1])
        # Update table in visom.
        N, M = np.array(self.table_data).shape
        fig, ax = plt.subplots()
        fig.set_figheight(2*N/8) # ewww
        fig.set_figwidth(20)
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ax.table(cellText=self.table_data, colLabels=self.table_cols, loc='center')
        self.matplot_win = self.vis.matplot(plt, win=self.matplot_win)

    @property
    def best_val_acc(self):
        if self.trained.qsize():
            return self.trained.queue[0][0].val_acc
        else:
            0.0
